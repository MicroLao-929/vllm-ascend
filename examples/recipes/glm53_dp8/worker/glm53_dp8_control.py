"""Optional compact B1 round gate for TP1 DP/EP MK decode.

The proven ``RoundGate`` in ``glm53_mk_ep.py`` performs three
``torch.distributed.all_gather_object`` calls per decode token (``select``,
``ready`` and ``complete``).  Each call pickles/unpickles Python tuples on the
CPU process group.  This module keeps the identical three-phase protocol and
the identical decision rules, but carries the control words in one fixed-size
packed message and uses a compact uint8 ``all_gather``.

Design constraints (deliberately enforced here):

* The fast path never changes the process role.  Exactly one rank may be
  ``real`` and every other rank must be ``idle``; all other combinations make
  every rank return ``None`` exactly like ``RoundGate.begin``.
* Every rank makes the *same* decision from the *same* gathered rows.  No rank
  may take a fast path while a peer takes a slow path: the ``_NEEDS_DETAILS``
  sentinel and the error flag are visible to all rows, so all ranks enter the
  one ``all_gather_object`` slow path together.
* No fallback happens after a cache write.  ``select``/``ready`` validate
  before the MK graph executes; ``complete`` is fail-stop and never falls back
  to native decode.
* Phases that are not on the per-token path (``enable``, ``bootstrap``,
  ``loaded`` and friends) keep the original object payload, so startup
  diagnostics are unchanged.

This module has no torch import at module scope, which keeps the CPU tests
independent of the NPU stack.  ``FixedCpuExchange`` imports torch lazily.
"""

from __future__ import annotations

from typing import NamedTuple

import struct


# Fixed message layout.  ``<`` removes padding, so the size is stable on every
# architecture (8+8+8+1+1+8+8+1+128 = 171 bytes).
_REQUEST_BYTES = 128
_MESSAGE = struct.Struct(f"<qqqBbqqB{_REQUEST_BYTES}s")
_MAGIC = 0x474C4D35334D4B01  # "GLM53MK" + protocol revision
_PHASE_IDS = {"select": 0, "ready": 1, "complete": 2, "probe": 3}
_ROLE = {"native": 0, "real": 1, "idle": 2}
_ROLE_NAMES = ("native", "real", "idle")
_NO_REQUEST = -1
_NEEDS_DETAILS = -2
_FLAG_ERROR = 1
_HOT_PHASES = frozenset(("select", "ready", "complete"))
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


class _Row(NamedTuple):
    magic: int
    rank: int
    sequence: int
    phase: int
    role: int
    length: int
    position: int
    flags: int
    request: bytes


def _require(condition, message):
    if not condition:
        raise RuntimeError(f"GLM53 fast round: {message}")


def env_enabled(name):
    """Read the same on/off spellings used by the stage-only env switches."""
    import os

    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


class FixedCpuExchange:
    """One compact ``all_gather`` of fixed-size byte rows on a CPU group.

    ``torch.distributed.all_gather`` on a small uint8 tensor is a single
    collective with no Python object serialization.  The receive buffers are
    allocated once; only the payload bytes are copied per round.
    """

    def __init__(self, group, size, message_size=_MESSAGE.size):
        _require(type(size) is int and size >= 2, "invalid fast round group size")
        _require(type(message_size) is int and message_size > 0, "invalid fast round message size")
        import torch
        import torch.distributed as dist

        self._torch = torch
        self._dist = dist
        self._group = group
        self._message_size = message_size
        self._send_bytes = bytearray(message_size)
        self._recv_bytes = [bytearray(message_size) for _ in range(size)]
        # Gloo fills the receive tensors from its worker threads; inference
        # mode must be off while the frombuffer views are created.
        with torch.inference_mode(False):
            self._send = torch.frombuffer(self._send_bytes, dtype=torch.uint8)
            self._recv = [torch.frombuffer(buffer, dtype=torch.uint8)
                          for buffer in self._recv_bytes]

    def exchange(self, payload):
        if len(payload) != self._message_size:
            raise ValueError(f"fast round payload has {len(payload)} bytes, expected {self._message_size}")
        self._send_bytes[:] = payload
        self._dist.all_gather(self._recv, self._send, group=self._group)
        return [bytes(buffer) for buffer in self._recv_bytes]


class FastRoundGate:
    """Drop-in hot-path replacement for ``RoundGate`` on TP1 DP/EP groups.

    ``exchange(payload) -> list[bytes]`` performs the compact fixed-size
    collective.  ``object_exchange(payload) -> list`` performs the original
    ``all_gather_object`` and is used for startup phases and for the one
    diagnostic retry when a payload cannot be represented in the fixed message.
    """

    MESSAGE_SIZE = _MESSAGE.size

    def __init__(self, rank, size, *, exchange, object_exchange, probe=True):
        _require(size in (2, 6, 8) and 0 <= rank < size, "invalid DP group size/rank")
        _require(callable(exchange) and callable(object_exchange), "fast round exchanges must be callable")
        self.rank = rank
        self.size = size
        self.sequence = 0
        self._exchange = exchange
        self._object_exchange = object_exchange
        self.stats = {"fixed": 0, "object": 0, "details": 0}
        self.mode = "fast"
        if probe:
            self.probe()

    # -- framing ---------------------------------------------------------
    def _pack(self, *, sequence, phase, role=0, length=_NO_REQUEST, position=0,
              flags=0, request=b""):
        _require(phase in _PHASE_IDS, f"unknown fast round phase {phase!r}")
        _require(len(request) <= _REQUEST_BYTES, "fast round request exceeds fixed field")
        buffer = bytearray(_MESSAGE.size)
        _MESSAGE.pack_into(buffer, 0, _MAGIC, self.rank, sequence, _PHASE_IDS[phase],
                           role, length, position, flags, request)
        return bytes(buffer)

    def _decode(self, raw_rows, sequence, phase):
        _require(len(raw_rows) == self.size,
                 f"received {len(raw_rows)} rows for group size {self.size}")
        rows = []
        for raw in raw_rows:
            _require(isinstance(raw, (bytes, bytearray)) and len(raw) == _MESSAGE.size,
                     "fast round row is not one fixed-size message")
            rows.append(_Row(*_MESSAGE.unpack_from(raw, 0)))
        _require(all(row.magic == _MAGIC for row in rows), "fast round magic mismatch")
        expected = [(rank, sequence, _PHASE_IDS[phase]) for rank in range(self.size)]
        actual = [(row.rank, row.sequence, row.phase) for row in rows]
        _require(actual == expected, f"rank/order mismatch at round {sequence} {phase}")
        return rows

    # -- startup / diagnostic object path --------------------------------
    def gather(self, phase, payload):
        """Exactly ``RoundGate.gather``: no sequence increment, object rows."""
        self.stats["object"] += 1
        rows = self._object_exchange((self.rank, self.sequence, phase, payload))
        _require([(rank, sequence, item_phase) for rank, sequence, item_phase, _ in rows]
                 == [(rank, self.sequence, phase) for rank in range(self.size)],
                 f"rank/order mismatch at round {self.sequence} {phase}")
        return [data for _, _, _, data in rows]

    def probe(self):
        """One fixed-size round trip that validates encoding and group ABI."""
        self.stats["fixed"] += 1
        raw = self._exchange(self._pack(sequence=-1, phase="probe"))
        self._decode(raw, -1, "probe")

    # -- per-token phases ------------------------------------------------
    def begin(self, role, request_id, position, error=None):
        """Mirror ``RoundGate.begin`` with a compact fixed-size payload."""
        self.sequence += 1
        if isinstance(role, str):
            role_id = _ROLE.get(role, -1)
        elif type(role) is int and role in (0, 1, 2):
            role_id = role
        else:
            role_id = -1

        length, packed_position, request = _NO_REQUEST, 0, b""
        flags = _FLAG_ERROR if error is not None else 0
        # A bad role code is carried in the fixed message and rejected by every
        # rank after the gather; only payloads that cannot be encoded need the
        # object diagnostic round.
        needs_details = error is not None
        if not needs_details and role_id >= 0 and role == "real":
            if (isinstance(request_id, str) and type(position) is int
                    and _INT64_MIN <= position <= _INT64_MAX):
                encoded = request_id.encode("utf-8", errors="surrogatepass")
                if len(encoded) <= _REQUEST_BYTES:
                    length, packed_position, request = len(encoded), position, encoded
                else:
                    needs_details = True
            else:
                needs_details = True
        elif (not needs_details and role_id >= 0 and role == "idle"
              and (request_id is not None or position is not None)):
            # Preserve the original global validation instead of raising on a
            # single rank while its peers wait in the collective.
            needs_details = True

        if needs_details:
            length, packed_position, request = _NEEDS_DETAILS, 0, b""

        self.stats["fixed"] += 1
        raw = self._exchange(self._pack(sequence=self.sequence, phase="select", role=role_id,
                                        length=length, position=packed_position,
                                        flags=flags, request=request))
        rows = self._decode(raw, self.sequence, "select")
        if (needs_details
                or any(row.flags & _FLAG_ERROR for row in rows)
                or any(row.length == _NEEDS_DETAILS for row in rows)):
            return self._begin_slow(role, request_id, position, error)

        _require(all(0 <= row.role < len(_ROLE_NAMES) for row in rows), "unknown rank role")
        roles = [_ROLE_NAMES[row.role] for row in rows]
        if roles.count("real") != 1 or roles.count("idle") != self.size - 1:
            return None
        owner = roles.index("real")
        row = rows[owner]
        _require(0 <= row.length <= _REQUEST_BYTES, "fast round owner request length")
        return owner, row.request[:row.length].decode("utf-8", errors="surrogatepass"), row.position

    def _begin_slow(self, role, request_id, position, error):
        """One object round trip, used only for diagnostics or oversized ids."""
        self.stats["details"] += 1
        payload = (role, request_id, position, error)
        rows = self.gather("select", payload)  # object path; counts one object round
        # Identical rules and messages as RoundGate.begin.
        _require(all(not row[3] for row in rows), f"eligibility errors {rows}")
        roles = [row[0] for row in rows]
        _require(all(role in ("native", "real", "idle") for role in roles), "unknown rank role")
        if roles.count("real") != 1 or roles.count("idle") != self.size - 1:
            return None
        owner = roles.index("real")
        return owner, rows[owner][1], rows[owner][2]

    def check(self, phase, error=None):
        """Mirror ``RoundGate.check`` with a compact fixed-size payload."""
        if phase not in _HOT_PHASES:
            return self.gather(phase, error)
        flags = _FLAG_ERROR if error is not None else 0
        self.stats["fixed"] += 1
        raw = self._exchange(self._pack(sequence=self.sequence, phase=phase, flags=flags))
        rows = self._decode(raw, self.sequence, phase)
        if flags or any(row.flags & _FLAG_ERROR for row in rows):
            self.stats["details"] += 1
            errors = self.gather(phase, error)
            _require(not any(errors), f"round {self.sequence} {phase} errors {errors}")
        else:
            _require(not any(row.flags & _FLAG_ERROR for row in rows),
                     f"round {self.sequence} {phase} flags unexpectedly set")

    def begin_tp(self, request_id, position, token, error=None):
        raise RuntimeError("GLM53 fast round: fast gate supports TP1 DP/EP groups only")
