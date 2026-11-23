"""Eight CPU workers compare the installed runner method with Gloo using actual local sockets/eventfds."""

import ast
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import unittest
from enum import IntEnum

candidate = os.environ.get("GLM53_TEST_ASCEND")
base = os.environ.get("GLM53_BASE_ASCEND")
if not (candidate and base):
    if __name__ == "__main__":
        raise SystemExit(
            "Set GLM53_TEST_ASCEND and GLM53_BASE_ASCEND to the candidate and reference package directories"
        )
    raise unittest.SkipTest("Fixed v0.30 worker packages are required")

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
import torch.distributed as dist

WORKER = Path(candidate).resolve() / "vllm_ascend/worker"
ORIGINAL = Path(base).resolve() / "vllm_ascend/worker/model_runner_v1.py"
IPC = WORKER / "glm53_mk_tp_control.py"
spec = importlib.util.spec_from_file_location("glm53_mk_local_dp_sync", WORKER / "glm53_mk_local_dp_sync.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class Comm(IntEnum):
    OTHER = 0
    ALLGATHER = 1
    MC2 = 2
    FUSED_MC2 = 3


def extracted(path):
    tree = ast.parse(path.read_text())
    f = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sync_metadata_across_dp")
    f.decorator_list = []
    ns = dict(
        torch=torch,
        dist=dist,
        CUDAGraphMode=Mode,
        MoECommType=Comm,
        get_dp_group=lambda: NS(cpu_group=dist.group.WORLD),
        should_skip_allreduce_across_dp_group=lambda c, d: c.skip,
        select_moe_comm_method=lambda n, c: c.comm,
        use_cann_megamoe=lambda c: c.mega,
        _post_process_cudagraph_mode=lambda t: int(t[1, :].min().item()),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])), str(path), "exec"), ns)
    return ns[f.name]


def make_runner(rank):
    ff = NS(
        oproj_tensor_parallel_size=1,
        lmhead_tensor_parallel_size=1,
        embedding_tensor_parallel_size=1,
        mlp_tensor_parallel_size=1,
    )
    return NS(
        dp_size=8,
        dp_rank=rank,
        parallel_config=NS(data_parallel_size_local=8, data_parallel_rank=rank),
        speculative_config=None,
        use_async_scheduling=False,
        model_config=NS(enforce_eager=True),
        max_num_reqs=1,
        input_batch=NS(num_reqs=1, req_ids=["r"], num_computed_tokens_cpu=[165], num_prompt_tokens=[128]),
        ascend_config=NS(finegrained_tp_config=ff),
        vllm_config=NS(skip=False, comm=Comm.OTHER, mega=False),
    )


def schedule(rid):
    return NS(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={rid: 1},
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        scheduled_cached_reqs=NS(resumed_req_ids=set()),
        preempted_req_ids=set(),
    )


def equal(left, right):
    assert left[0] == right[0] and left[2] == right[2]
    if left[1] is None:
        assert right[1] is None
    else:
        assert torch.equal(left[1], right[1]) and right[1].dtype == torch.int32 and right[1].device.type == "cpu"


def worker(rank, rendezvous, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, world_size=8, rank=rank)
    spec = importlib.util.spec_from_file_location("tested_tp_channel", IPC)
    ipc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ipc)

    class Gate:
        size = 8

        def __init__(self, rank):
            self.rank = rank

        def gather(self, phase, payload):
            rows = [None] * 8
            dist.all_gather_object(rows, (phase, payload))
            assert all(k == phase for k, v in rows)
            return [v for k, v in rows]

        def check(self, phase, error=None):
            rows = self.gather(phase, error)
            if any(rows):
                raise RuntimeError(f"{phase}: {rows}")

    raw = extracted(ORIGINAL)
    modified = extracted(WORKER / "model_runner_v1.py")
    runner = make_runner(rank)
    gate = Gate(rank)
    topo = (1, 1, 8, 1, 1)
    obj = sync.make_sync("on", gate, runner, topo, True, ipc.LocalTPControl.connect)
    assert obj is not None
    runner._glm53_mk_ep = NS(local_dp_sync=obj)
    raw_results = []

    def verify(n, graph, marker, draft=False):
        before = raw(runner, n, draft, Mode(graph))
        after = modified(runner, n, draft, Mode(graph), glm53_dp_step=marker)
        equal(before, after)

    # First native prefill has no previous completed MK round and no IPC scout.
    verify(128 if rank == 0 else 1, 0, schedule("p") if rank == 0 else True)
    assert obj.stats["local_hits"] == 0 and obj.stats["consensus_fallback"] == 0 and obj.stats["gloo_calls"] == 1
    steps = 36
    for i in range(steps):
        origin = i % 8
        tok = 1
        graph = 0
        draft = False
        req = "r"
        expected = i % 12
        runner.input_batch.num_computed_tokens_cpu[0] = 165 + i
        marker = schedule(req) if rank == origin else True
        runner.vllm_config.comm = (Comm.OTHER, Comm.ALLGATHER, Comm.MC2, Comm.FUSED_MC2)[i % 4]
        runner.vllm_config.mega = i % 2 == 0
        runner.ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size = 2 if i % 5 == 0 else 1
        obj.arm(i + 1, NS(owner=origin, request=req, position=164 + i))
        if expected == 0 and rank == origin:
            tok = 128
            marker = schedule("new-prefill")
        elif expected == 1 and rank == origin:
            marker.scheduled_new_reqs = [req]
        elif expected == 2 and rank == (origin + 1) % 8:
            marker = None
        elif expected == 3:
            graph = rank % 3
        elif expected == 4:
            tok = rank + 1
        elif expected == 5:
            draft = True
        elif expected == 6 and rank == origin:
            runner.input_batch.num_computed_tokens_cpu[0] -= 1
        verify(tok, graph, marker, draft)
    assert obj.stats["local_hits"] == 15 and obj.stats["consensus_fallback"] == 21, obj.stats
    before = dict(obj.stats)
    # Existing bypasses still return before any exchange.
    runner.vllm_config.skip = True
    verify(rank + 1, rank % 3, None)
    assert obj.stats == before
    runner.vllm_config.skip = False
    observe = sync.make_sync("observe", gate, runner, topo, True, ipc.LocalTPControl.connect)
    runner._glm53_mk_ep.local_dp_sync = observe
    observe.arm(55, NS(owner=3, request="r", position=199))
    runner.input_batch.num_computed_tokens_cpu[0] = 200
    verify(1, 0, schedule("r") if rank == 3 else True)
    assert observe.local is None and observe.stats["gloo_calls"] == observe.stats["gloo_local_candidate"] == 1
    runner.parallel_config.data_parallel_size_local = 7 if rank == 2 else 8
    assert sync.make_sync("on", gate, runner, topo, True, lambda *a: (_ for _ in ()).throw(AssertionError())) is None
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(obj.stats))
    obj.close()
    # Test a real second unique channel; close one follower before the next exchange.
    # Bound only this test's peer-death deadline.
    ipc._ROUND_TIMEOUT = 1
    channel = ipc.LocalTPControl.connect(gate, sync._FRAME.size)
    if rank == 7:
        channel.close()
    dist.barrier()
    if rank != 7:
        broken = sync.MetadataSync("on", rank, channel)
        broken.arm(69, NS(owner=0, request="r", position=199))
        original_reduce = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("unexpected Gloo after transport failure")
        )
        try:
            broken.packed(runner, schedule("r") if rank == 0 else True, 1, 0, False, torch, original_reduce, None)
        except (RuntimeError, OSError, TimeoutError):
            pass
        else:
            raise AssertionError("closed peer was accepted")
        broken.close()
    dist.barrier()
    dist.destroy_process_group()


class FakeTransport:
    def __init__(self, fn):
        self.fn = fn

    def all_gather(self, p):
        return self.fn(p)


def fail_cases():
    runner = make_runner(0)
    s = schedule("r")
    for fn in (lambda b: b[:3], lambda b: (b"BAD!" + b[4:]) * 8, lambda b: (_ for _ in ()).throw(TimeoutError())):
        obj = sync.MetadataSync("on", 0, FakeTransport(fn))
        obj.arm(1, NS(owner=0, request="r", position=164))
        try:
            obj.packed(runner, s, 1, 0, False, torch, lambda *a, **k: (_ for _ in ()).throw(AssertionError()), None)
        except (RuntimeError, TimeoutError):
            pass
        else:
            raise AssertionError("broken transport accepted")
    obj = sync.MetadataSync("observe", 0, None)
    obj.lock.acquire()
    try:
        obj.packed(runner, s, 1, 0, False, torch, lambda *a, **k: None, None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("concurrent call entered")


def test_eight_rank_local_dp_sync():
    fail_cases()
    with tempfile.TemporaryDirectory() as d:
        import torch.multiprocessing as tmp

        tmp.start_processes(worker, args=(str(Path(d, "gloo")), d), nprocs=8, join=True, start_method="fork")
        records = [json.loads(Path(d, f"rank{r}.json").read_text()) for r in range(8)]
    print(
        json.dumps(
            {
                "ranks": 8,
                "rounds_per_rank": 36,
                "local_hits_per_rank": [r["local_hits"] for r in records],
                "fallback_per_rank": [r["consensus_fallback"] for r in records],
                "checked": [
                    "native prefill",
                    "each owner",
                    "ragged",
                    "graph modes",
                    "draft",
                    "fine-grained TP",
                    "Gloo bypass",
                    "concurrent call",
                    "epoch/size mismatch",
                    "actual peer death",
                ],
            }
        )
    )


if __name__ == "__main__":
    test_eight_rank_local_dp_sync()
