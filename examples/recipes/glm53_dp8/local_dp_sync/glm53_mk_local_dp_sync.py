"""Opt-in, decode-only DP metadata exchange for a single-host eight-rank MK worker."""
import logging
import os
import struct
from threading import Lock
from time import perf_counter_ns

log = logging.getLogger(__name__)
# A private channel carries one row per DP rank. Epoch, origin and position refer
# to the last completed distributed MK round, common to the owner and companions.
_FRAME = struct.Struct("<4sQiiiii")
_STOP = (1, 8, 64, 128, 256, 512, 1024, 2048)


def make_sync(mode, gate, runner, topology, ipc, connector):
    if mode == "off":
        return None
    if mode not in ("observe", "on"):
        raise ValueError("MK_GLM53_LOCAL_DP_SYNC must be off, observe, or on")
    pc = runner.parallel_config
    ok = (ipc and topology == (1, 1, 8, 1, 1) and gate.size == runner.dp_size == 8
          and getattr(pc, "data_parallel_size_local", None) == 8 and gate.rank == runner.dp_rank == pc.data_parallel_rank
          and runner.speculative_config is None and not runner.use_async_scheduling
          and runner.model_config.enforce_eager and runner.max_num_reqs == 1)
    # Use the original object gate for bootstrap. A non-eligible member keeps
    # every member on the original Gloo path before any IPC token is sent.
    if not all(gate.gather("local_dp_shape", bool(ok))):
        log.info("GLM53 local DP metadata disabled for unsupported worker group")
        return None
    local = connector(gate, _FRAME.size) if mode == "on" else None
    return MetadataSync(mode, gate.rank, local)


class MetadataSync:
    def __init__(self, mode, rank, local):
        self.mode = mode
        self.rank = rank
        self.local = local
        self.previous = None
        self.lock = Lock()
        self.stats = dict(arms=0, local_hits=0, consensus_fallback=0, gloo_calls=0,
                          gloo_after_mk=0, gloo_local_candidate=0, gloo_ns=0,
                          candidate_gloo_ns=0, local_ns=0)
        self.candidate = False
        self.trace = os.getenv("MK_GLM53_LOCAL_DP_SYNC_TRACE") == "1"
        self.outcome = "native"

    def arm(self, sequence, active):
        # Called on all 8 workers only after success of both completion vote and barrier.
        if self.trace:
            log.info("GLM53 local DP step rank=%d sequence=%d owner=%d request=%s position=%d outcome=%s",
                     self.rank, sequence, active.owner, active.request, active.position, self.outcome)
        self.previous = (sequence, active.owner, active.request, active.position)
        self.stats["arms"] += 1

    def _role(self, runner, step, origin, rid, prev, tokens, mode, draft):
        if draft or mode != 0 or tokens != 1:
            return 0
        if self.rank != origin:
            return 2 if step is True else 0
        if step is None or step is True:
            return 0
        try:
            batch = runner.input_batch
            return int(batch.num_reqs == 1 and batch.req_ids[0] == rid
                       and int(batch.num_computed_tokens_cpu[0]) == prev + 1
                       and prev >= int(batch.num_prompt_tokens[0])
                       and step.total_num_scheduled_tokens == 1 and step.num_scheduled_tokens == {rid: 1}
                       and not step.scheduled_new_reqs and not step.scheduled_spec_decode_tokens
                       and not step.scheduled_encoder_inputs and not step.finished_req_ids
                       and not step.scheduled_cached_reqs.resumed_req_ids and not step.preempted_req_ids)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return 0

    def rows(self, runner, step, tokens, mode, draft):
        previous, self.previous = self.previous, None
        self.candidate = False
        if previous is None:
            return None
        sequence, origin, rid, prev = previous
        role = self._role(runner, step, origin, rid, prev, tokens, mode, draft)
        self.candidate = bool(role)
        self.stats["gloo_after_mk"] += 1
        if self.mode == "observe":
            return None
        start = perf_counter_ns()
        payload = _FRAME.pack(b"DPM1", sequence, origin, prev, role, tokens, mode)
        wire = self.local.all_gather(payload)
        if not isinstance(wire, (bytes, bytearray)) or len(wire) != 8 * _FRAME.size:
            raise RuntimeError("incomplete local DP metadata exchange")
        rows = list(_FRAME.iter_unpack(wire))
        # With a frame/epoch error the group cannot safely issue another collective.
        if any(row[:4] != (b"DPM1", sequence, origin, prev) for row in rows):
            raise RuntimeError("local DP metadata epoch or origin mismatch")
        if any(row[4] not in (0, 1, 2) for row in rows):
            raise RuntimeError("invalid local DP role")
        if any(row[4:] != (1 if index == origin else 2, 1, 0) for index, row in enumerate(rows)):
            self.stats["consensus_fallback"] += 1
            self._log("consensus_fallback")
            return None
        self.stats["gloo_after_mk"] -= 1
        self.stats["local_hits"] += 1
        self.stats["local_ns"] += perf_counter_ns() - start
        self._log("local_hits")
        return [[row[5] for row in rows], [row[6] for row in rows]]

    def packed(self, runner, step, tokens, mode, draft, torch, all_reduce, group):
        if not self.lock.acquire(False):
            raise RuntimeError("concurrent local DP metadata exchange")
        try:
            rows = self.rows(runner, step, tokens, mode, draft)
            if rows is not None:
                self.outcome = "local"
                return torch.tensor(rows, device="cpu", dtype=torch.int32)
            self.outcome = "gloo_candidate" if self.candidate else "gloo"
            tensor = torch.zeros(2, 8, device="cpu", dtype=torch.int32)
            tensor[0][self.rank] = tokens
            tensor[1][self.rank] = mode
            start = perf_counter_ns()
            all_reduce(tensor, group=group)
            self.recorded_gloo(start)
            return tensor
        finally:
            self.lock.release()

    def recorded_gloo(self, start):
        elapsed = perf_counter_ns() - start
        self.stats["gloo_calls"] += 1
        self.stats["gloo_ns"] += elapsed
        if self.candidate:
            self.stats["gloo_local_candidate"] += 1
            self.stats["candidate_gloo_ns"] += elapsed
            self._log("gloo_local_candidate")

    def _log(self, event):
        if self.stats[event] in _STOP:
            log.info("GLM53 local DP metadata rank=%d mode=%s event=%s counters=%s",
                     self.rank, self.mode, event, dict(self.stats))

    def close(self):
        log.info("GLM53 local DP metadata rank=%d mode=%s final=%s", self.rank, self.mode, dict(self.stats))
        if self.local is not None:
            self.local.close()

