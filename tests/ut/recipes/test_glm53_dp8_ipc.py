"""Exercise the local host control protocol on eight Gloo CPU ranks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist


WORKERS = Path(__file__).resolve().parents[3] / "examples/recipes/glm53_dp8/worker"


def load(name):
    spec = importlib.util.spec_from_file_location(name, WORKERS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Setup:
    def __init__(self, rank, size):
        self.rank, self.size = rank, size

    def gather(self, phase, value):
        result = [None] * self.size
        dist.all_gather_object(result, value)
        return result

    def check(self, phase, error=None):
        errors = self.gather(phase, error)
        if any(errors):
            raise RuntimeError(errors)


def exercise(rank, group):
    dist.init_process_group("gloo", init_method=group, world_size=8, rank=rank)
    startup = Setup(rank, 8)
    gate_class = load("glm53_dp8_control").FastRoundGate
    width = gate_class.MESSAGE_SIZE
    control = load("glm53_mk_tp_control").LocalTPControl.connect(startup, width)
    try:
        def exchange(payload):
            result = control.all_gather(payload)
            return [result[start:start + width] for start in range(0, 8 * width, width)]

        gate = gate_class(rank, 8, exchange=exchange, object_exchange=lambda value: startup.gather("object", value))
        for owner in range(8):
            request = f"request-{owner}"
            details = ("real", request, owner) if rank == owner else ("idle", None, None)
            assert gate.begin(*details) == (owner, request, owner)
            gate.check("ready")
            gate.check("complete")
        assert gate.begin("real", "conflict", 1) is None
    finally:
        control.close()
        dist.destroy_process_group()


class LocalControlTest(unittest.TestCase):
    def test_rotating_request_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            group = (Path(directory) / "group").as_uri()
            torch.multiprocessing.start_processes(exercise, args=(group,), nprocs=8, start_method="fork")


if __name__ == "__main__":
    unittest.main()
