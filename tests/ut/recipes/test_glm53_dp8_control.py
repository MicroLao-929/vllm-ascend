"""CPU tests for the optional compact DP/EP round gate.

No torch, NPU, network or installation is used.  Eight in-process ranks are
driven through a barrier-backed fake collective so the real three-phase
protocol, the identical-decision rule and the slow diagnostic path are
exercised concurrently.
"""

import os
import threading
import unittest
from pathlib import Path

import importlib.util

_WORKER = Path(__file__).resolve().parents[3] / "examples/recipes/glm53_dp8/worker/glm53_dp8_control.py"
_spec = importlib.util.spec_from_file_location("glm53_dp8_control", _WORKER)
fast_round = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fast_round)
FastRoundGate = fast_round.FastRoundGate
env_enabled = fast_round.env_enabled


class FakeCollective:
    """One lockstep fixed-size/object collective per rank."""

    def __init__(self, size, calls=128, timeout=5.0):
        self.size = size
        self.timeout = timeout
        self._lock = threading.Lock()
        self._barriers = [threading.Barrier(size) for _ in range(calls)]
        self._slots = [dict() for _ in range(calls)]
        self.payloads = []

    def exchange_for(self, rank):
        state = {"index": 0}

        def exchange(payload):
            index = state["index"]
            state["index"] += 1
            if index >= len(self._barriers):
                raise RuntimeError("fake collective exhausted")
            with self._lock:
                self._slots[index][rank] = payload
                self.payloads.append((index, rank, payload))
            try:
                self._barriers[index].wait(timeout=self.timeout)
            except threading.BrokenBarrierError as exc:
                raise RuntimeError("fake collective did not reach all ranks") from exc
            with self._lock:
                return [self._slots[index][peer] for peer in range(self.size)]

        return exchange


class Rehearsal:
    def __init__(self, size=8, *, probe=False):
        self.size = size
        self.fixed = FakeCollective(size)
        self.objects = FakeCollective(size)
        self.gates = [
            FastRoundGate(rank, size,
                          exchange=self.fixed.exchange_for(rank),
                          object_exchange=self.objects.exchange_for(rank),
                          probe=probe)
            for rank in range(size)
        ]

    def run(self, action):
        results = [None] * self.size
        errors = [None] * self.size

        def worker(rank):
            try:
                results[rank] = action(rank, self.gates[rank])
            except BaseException as exc:  # tests assert the exact failure mode
                errors[rank] = exc

        threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(self.size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assert_no_thread_leaks(threads)
        return results, errors

    @staticmethod
    def assert_no_thread_leaks(threads):
        if any(thread.is_alive() for thread in threads):
            raise AssertionError("a rank thread did not finish; the protocol diverged")


def roles_with_real(size, owner, request="cmpl-test-0-abc", position=17):
    return [(("real", request, position) if rank == owner else ("idle", None, None))
            for rank in range(size)]


class FastRoundTests(unittest.TestCase):
    def test_dp8_one_real_and_seven_idle_agree(self):
        rehearsal = Rehearsal(8)
        roles = roles_with_real(8, 3)
        results, errors = rehearsal.run(lambda rank, gate: gate.begin(*roles[rank]))
        self.assertTrue(all(error is None for error in errors), errors)
        expected = (3, "cmpl-test-0-abc", 17)
        self.assertTrue(all(result == expected for result in results), results)
        self.assertTrue(all(gate.stats["object"] == 0 for gate in rehearsal.gates))

        ready, ready_errors = rehearsal.run(lambda rank, gate: gate.check("ready"))
        complete, complete_errors = rehearsal.run(lambda rank, gate: gate.check("complete"))
        self.assertTrue(all(error is None for error in ready_errors + complete_errors))
        self.assertTrue(all(value is None for value in ready + complete))
        self.assertTrue(all(gate.stats["fixed"] == 3 and gate.stats["object"] == 0
                            for gate in rehearsal.gates))

    def test_all_native_returns_none(self):
        rehearsal = Rehearsal(8)
        results, errors = rehearsal.run(lambda rank, gate: gate.begin("native", None, None))
        self.assertTrue(all(error is None for error in errors), errors)
        self.assertTrue(all(result is None for result in results))

    def test_two_real_ranks_return_none_everywhere(self):
        rehearsal = Rehearsal(8)
        roles = [("idle", None, None)] * 8
        roles[2] = ("real", "cmpl-a", 9)
        roles[5] = ("real", "cmpl-b", 11)
        results, errors = rehearsal.run(lambda rank, gate: gate.begin(*roles[rank]))
        self.assertTrue(all(error is None for error in errors), errors)
        self.assertTrue(all(result is None for result in results), results)

    def test_mixed_real_native_idle_returns_none_everywhere(self):
        rehearsal = Rehearsal(8)
        roles = roles_with_real(8, 1)
        roles[6] = ("native", None, None)
        results, errors = rehearsal.run(lambda rank, gate: gate.begin(*roles[rank]))
        self.assertTrue(all(error is None for error in errors), errors)
        self.assertTrue(all(result is None for result in results), results)

    def test_unknown_role_fails_on_every_rank_without_slow_path(self):
        rehearsal = Rehearsal(8)
        roles = roles_with_real(8, 0)
        roles[4] = ("bogus", None, None)
        _, errors = rehearsal.run(lambda rank, gate: gate.begin(*roles[rank]))
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors), errors)
        self.assertTrue(all("unknown rank role" in str(error) for error in errors), errors)
        # The bad role is carried in the fixed message, so peers do not hang.
        self.assertTrue(all(gate.stats["object"] == 0 for gate in rehearsal.gates))

    def test_local_error_uses_one_object_round_and_fails_everywhere(self):
        rehearsal = Rehearsal(8)
        roles = roles_with_real(8, 2)
        local_errors = [None] * 8
        local_errors[6] = "ValueError: boom"
        _, errors = rehearsal.run(
            lambda rank, gate: gate.begin(*roles[rank], error=local_errors[rank]))
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors), errors)
        self.assertTrue(all("eligibility errors" in str(error) for error in errors), errors)
        self.assertTrue(all("boom" in str(error) for error in errors), errors)
        self.assertTrue(all(gate.stats["fixed"] == 1 and gate.stats["object"] == 1
                            and gate.stats["details"] == 1 for gate in rehearsal.gates))

    def test_oversized_request_uses_one_slow_round_and_still_decides(self):
        rehearsal = Rehearsal(8)
        long_request = "x" * (fast_round._REQUEST_BYTES + 1)
        roles = roles_with_real(8, 0, request=long_request, position=5)
        results, errors = rehearsal.run(lambda rank, gate: gate.begin(*roles[rank]))
        self.assertTrue(all(error is None for error in errors), errors)
        self.assertTrue(all(result == (0, long_request, 5) for result in results), results)
        self.assertTrue(all(gate.stats["object"] == 1 for gate in rehearsal.gates))

    def test_rank_or_sequence_mismatch_fails_on_every_rank(self):
        rehearsal = Rehearsal(8)
        fixed = FakeCollective(8)

        def corrupt_for(rank):
            inner = fixed.exchange_for(rank)

            def exchange(payload):
                rows = list(inner(payload))
                fields = list(fast_round._MESSAGE.unpack_from(rows[0], 0))
                fields[2] += 1  # corrupt the sequence of rank 0
                buffer = bytearray(fast_round._MESSAGE.size)
                fast_round._MESSAGE.pack_into(buffer, 0, *fields)
                rows[0] = bytes(buffer)
                return rows

            return exchange

        gates = [FastRoundGate(rank, 8, exchange=corrupt_for(rank),
                               object_exchange=rehearsal.objects.exchange_for(rank),
                               probe=False) for rank in range(8)]
        results = [None] * 8
        errors = [None] * 8

        def worker(rank):
            try:
                results[rank] = gates[rank].begin(*roles_with_real(8, 0)[rank])
            except BaseException as exc:
                errors[rank] = exc

        threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors), errors)
        self.assertTrue(all("rank/order mismatch" in str(error) for error in errors), errors)

    def test_startup_gather_is_object_and_does_not_advance_sequence(self):
        rehearsal = Rehearsal(8)
        results, errors = rehearsal.run(
            lambda rank, gate: gate.gather("enable", (rank, "config")))
        self.assertTrue(all(error is None for error in errors), errors)
        for rank, gate in enumerate(rehearsal.gates):
            self.assertEqual(gate.sequence, 0)
            self.assertEqual(gate.stats["object"], 1)
            self.assertEqual(results[rank][rank], (rank, "config"))

    def test_probe_validates_fixed_layout_without_advancing_sequence(self):
        rehearsal = Rehearsal(8)
        _, errors = rehearsal.run(lambda rank, gate: gate.probe())
        self.assertTrue(all(error is None for error in errors), errors)
        self.assertTrue(all(gate.stats["fixed"] == 1 for gate in rehearsal.gates))
        self.assertTrue(all(gate.sequence == 0 for gate in rehearsal.gates))

    def test_ready_error_slow_path_reports_details_everywhere(self):
        rehearsal = Rehearsal(8)
        rehearsal.run(lambda rank, gate: gate.begin(*roles_with_real(8, 0)[rank]))

        def action(rank, gate):
            return gate.check("ready", "RuntimeError: bad metadata") if rank == 5 else gate.check("ready")

        _, errors = rehearsal.run(action)
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors), errors)
        self.assertTrue(all("errors" in str(error) for error in errors), errors)
        self.assertTrue(all("bad metadata" in str(error) for error in errors), errors)

    def test_fast_begin_tp_is_rejected(self):
        gate = FastRoundGate(0, 2, exchange=lambda payload: [payload, payload],
                             object_exchange=lambda payload: [payload, payload], probe=False)
        with self.assertRaises(RuntimeError):
            gate.begin_tp("r", 1, 7)

    def test_message_round_trip_is_fixed_size(self):
        rehearsal = Rehearsal(2)
        request = "cmpl-unicode-测试-0"
        roles = roles_with_real(2, 1, request=request, position=23)
        results, errors = rehearsal.run(lambda rank, gate: gate.begin(*roles[rank]))
        self.assertTrue(all(error is None for error in errors), errors)
        self.assertTrue(all(result == (1, request, 23) for result in results), results)
        self.assertEqual(fast_round._MESSAGE.size, 171)

    def test_env_enabled_spellings(self):
        name = "MK_FAST_ROUND_TEST_SWITCH"
        previous = os.environ.pop(name, None)
        try:
            self.assertFalse(env_enabled(name))
            for value in ("1", "true", "YES", " on "):
                os.environ[name] = value
                self.assertTrue(env_enabled(name), value)
            for value in ("0", "false", "off", ""):
                os.environ[name] = value
                self.assertFalse(env_enabled(name), value)
        finally:
            os.environ.pop(name, None)
            if previous is not None:
                os.environ[name] = previous


if __name__ == "__main__":
    unittest.main()
