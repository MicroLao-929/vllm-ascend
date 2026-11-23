"""Pre-native-builder continuation for ordinary GLM-5.3 MK decode (TP1/DP8/EP8).

Scope
-----
This module is the decision-and-fallback core of a narrowly scoped fast path
that removes the *native* per-layer attention metadata build from the live
decode rank, before the 45-layer MK graph runs.  It is pure Python: it must be
importable without torch, vllm or an NPU so the admission rules can be
exercised on CPU.

What happens on a fast step (candidate ``pre_builder_deferral``)
---------------------------------------------------------------
1. ``execute_model`` has already run ``_update_states`` and ``_prepare_inputs``
   for the step.  The live rank asks this module whether the step is a
   *proven* single-token continuation of the request whose previous MK round
   completed on all eight DP/EP ranks.
2. If admitted, the module asks the injected ``plan_reader`` for the CPU cache
   addresses of exactly this token and *defers* the native metadata build
   instead of running it.  The MK bridge consumes the CPU plan.
3. Before the MK graph touches any cache, the normal round gate runs
   ``select``.  If the eight ranks do not report exactly one ``real`` owner and
   seven ``idle`` companions, the live rank calls ``materialize()`` -- the
   deferred native metadata is built *before* the first MK write -- and the
   step runs native.  A step that ran the native model is never recorded as MK.
4. ``shadow`` mode builds the native metadata anyway and cross-checks the CPU
   plan against the native addresses.  ``on`` mode skips the build and is only
   allowed after this process has produced a clean, sufficiently long shadow
   record.  Any shadow mismatch disables ``on`` for the rest of the process.

Candidate ``pre_replay_intercept``
----------------------------------
The runner may eventually run native prefill/dummy work with ACL graphs
(``FULL_DECODE_ONLY``) while the MK hook stays eager and intercepts before the
ACL replay.  That path is modelled here by ``allow_pre_replay`` but is *not*
the production default: it changes graph-mode admission and cannot be verified
without hardware.  Every check that makes it fail closed is still enforced.

Nothing in this module captures a graph, schedules a dummy rank, samples, or
writes a cache tile.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

NATIVE = "native"
MK = "mk"

ROLE_REAL = "real"
ROLE_IDLE = "idle"
ROLE_NATIVE = "native"
_ROLES = (ROLE_NATIVE, ROLE_REAL, ROLE_IDLE)

CANDIDATE_DEFER = "pre_builder_deferral"
CANDIDATE_INTERCEPT = "pre_replay_intercept"
_CANDIDATES = (CANDIDATE_DEFER, CANDIDATE_INTERCEPT)

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ON = "on"
_MODES = (MODE_OFF, MODE_SHADOW, MODE_ON)

# The only topology this module admits today.  Anything else is native.
TP1_DP8_EP8 = (1, 1, 8, 1, 1)

EAGER_RUNTIME_MODES = frozenset({"NONE"})
GRAPH_RUNTIME_MODES = frozenset(
    {"PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}
)

DEFAULT_MIN_SHADOW = 8

_MISSING = object()


class _Declined(Exception):
    """Internal: the fast path cannot be proven for this step."""


def _attr(obj, name, default=_MISSING):
    """Read an attribute without letting a property/exception abort a step."""
    if obj is None:
        return default
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _as_bool(value, default=None):
    if value is _MISSING or value is None:
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _as_int(value, default=None):
    if value is _MISSING or value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _sequence(value, length=None):
    """Return a tuple view of a tensor/numpy/list-like value, or None."""
    if value is _MISSING or value is None:
        return None
    view = getattr(value, "np", value)  # vllm_ascend pinned buffers expose .np
    try:
        items = tuple(view)
    except TypeError:
        return None
    if length is not None:
        if len(items) < length:
            return None
        return items[:length]
    return items


def _first(value):
    if value is _MISSING or value is None:
        return None
    try:
        return value[0]
    except Exception:
        return None


def _token_items(value):
    """Normalise ``SchedulerOutput.num_scheduled_tokens`` to ((req_id, n), ...)."""
    if value is _MISSING or value is None:
        return None
    items = getattr(value, "items", None)
    if callable(items):
        try:
            return tuple((str(k), int(v)) for k, v in items())
        except Exception:
            return None
    seq = _sequence(value)
    if seq is None:
        return None
    out = []
    for item in seq:
        try:
            req_id, count = item
            out.append((str(req_id), int(count)))
        except Exception:
            return None
    return tuple(out)


def _optional_sequence(value):
    """Return a tuple for optional id collections, else None."""
    if value is _MISSING or value is None:
        return None
    seq = _sequence(value)
    return seq


def _flag_len(obj, name):
    value = _attr(obj, name)
    if value is _MISSING or value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _tuple_or_none(value):
    if value is _MISSING or value is None:
        return None
    try:
        return tuple(value)
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# Runner facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """The single-token continuation this step claims to be."""

    request_id: str
    position: int
    prompt: int
    count: int

    @property
    def phase(self):
        return self.position - self.prompt


@dataclass(frozen=True)
class Facts:
    """CPU-visible state used by admission.

    ``None`` means "not available"; every ``None`` that a rule depends on makes
    the rule fail closed.  Building this object never mutates the runner.

    The field names mirror the objects inspected in the live runner so the
    wiring diff stays mechanical.  ``from_runner`` is deliberately defensive:
    a missing or raising attribute must degrade to native, never crash a step.
    """

    # topology / configuration
    topology: tuple | None = None
    speculative: bool | None = None
    async_scheduling: bool | None = None
    async_spec_decode: bool | None = None
    dcp: bool | None = None
    dynamic_eplb: bool | None = None
    metadata_executor_in_flight: bool | None = None
    metadata_executor_capture_producers: bool | None = None
    long_seq_metadata: bool | None = None
    sparse_kv_offload: bool | None = None
    mm_prefix_lm: bool | None = None
    kvpp_scheduler: bool | None = None
    prefix_caching: bool | None = None
    mamba_cache_mode: str | None = None
    # Stateful features the caller must report explicitly; None means the
    # wiring did not supply the probe and the rule is skipped (see from_runner).
    deferred_state_corrections: bool | None = None
    use_compress: bool | None = None
    decoder_replay: bool | None = None
    replay_active: bool | None = None
    with_prefill: bool | None = None
    encoder_inputs: int | None = None
    offload_active: bool | None = None
    kv_transfer: bool | None = None
    ec_transfer: bool | None = None

    # MK hook / bridge state
    hook_present: bool = False
    local_rank: int | None = None
    hook_invalid: bool | None = None
    hook_active: bool | None = None
    hook_tp_size: int | None = None
    plan_size: int | None = None
    plan_depth: int | None = None
    verified_graph: bool | None = None
    last_completed_round: tuple | None = None
    hook_request: str | None = None
    hook_next_position: int | None = None
    bridge_request: str | None = None
    bridge_position: int | None = None
    bridge_poisoned: bool | None = None
    max_context_tokens: int | None = None

    # batch / schedule
    batch_num_reqs: int | None = None
    batch_req_ids: tuple | None = None
    batch_num_computed: int | None = None
    batch_num_prompt: int | None = None
    optimistic_seq_len: int | None = None
    scheduler_total_tokens: int | None = None
    scheduler_tokens: tuple | None = None
    scheduler_spec_tokens: int | None = None
    scheduler_new_reqs: int | None = None
    scheduler_finished_ids: tuple | None = None
    scheduler_resumed_ids: tuple | None = None
    scheduler_preempted_ids: tuple | None = None
    scheduler_sync_kv_loads: bool | None = None

    # execution
    num_tokens: int | None = None
    runtime_mode: str | None = None
    capturing: bool | None = None
    query_start_loc: tuple | None = None
    gdn_query_start_loc: tuple | None = None
    has_gdn: bool | None = None

    @classmethod
    def from_runner(cls, runner, scheduler_output, *, num_tokens, runtime_mode,
                    capturing, deferred_state_corrections=None,
                    use_compress=None, decoder_replay=None, replay_active=None,
                    with_prefill=None, encoder_inputs=None, offload_active=None,
                    kv_transfer=None, ec_transfer=None):
        """Snapshot the live objects; every probe is exception-safe.

        The nine keyword-only arguments above cannot be read portably from CPU
        fakes, so the runner wiring passes them explicitly.  Left as ``None``
        they are treated as "not tripped"; the production call site supplies
        every one of them from the exact objects the handoff checks.
        """
        parallel = _attr(runner, "parallel_config")
        cache = _attr(runner, "cache_config")
        hook = _attr(runner, "_glm53_mk_ep")
        bridge = _attr(hook, "bridge")
        executor = _attr(runner, "device_metadata_executor")
        kvpp = _attr(runner, "kvpp")
        batch = _attr(runner, "input_batch")
        rec = _attr(scheduler_output, "scheduled_cached_reqs")

        topology = None
        if parallel is not _MISSING and parallel is not None:
            values = tuple(
                _as_int(_attr(parallel, name))
                for name in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            )
            if all(v is not None for v in values):
                topology = values

        def opt_bool(obj, name):
            return _as_bool(_attr(obj, name))

        gdn = _as_bool(_attr(runner, "_has_gdn"))
        query = _sequence(_attr(runner, "query_start_loc"), 2)
        gdn_query = (
            _sequence(_attr(runner, "gdn_query_start_loc"), 2) if gdn else None
        )

        return cls(
            topology=topology,
            speculative=_as_bool(_attr(runner, "speculative_config") is not None),
            async_scheduling=opt_bool(runner, "use_async_scheduling"),
            async_spec_decode=opt_bool(runner, "use_async_spec_decode"),
            dcp=opt_bool(runner, "use_dcp"),
            dynamic_eplb=opt_bool(runner, "dynamic_eplb"),
            metadata_executor_in_flight=opt_bool(executor, "submission_in_flight"),
            metadata_executor_capture_producers=opt_bool(executor, "capture_producers"),
            long_seq_metadata=_as_bool(_attr(runner, "long_seq_metadata") is not None),
            sparse_kv_offload=opt_bool(runner, "sparse_kv_offload_enabled"),
            mm_prefix_lm=opt_bool(runner, "is_mm_prefix_lm"),
            kvpp_scheduler=_as_bool(_attr(kvpp, "scheduler") is not None),
            prefix_caching=opt_bool(cache, "enable_prefix_caching"),
            mamba_cache_mode=(
                None
                if _attr(cache, "mamba_cache_mode") is _MISSING
                else _attr(cache, "mamba_cache_mode")
            ),
            deferred_state_corrections=deferred_state_corrections,
            use_compress=use_compress,
            decoder_replay=decoder_replay,
            replay_active=replay_active,
            with_prefill=with_prefill,
            encoder_inputs=encoder_inputs,
            offload_active=offload_active,
            kv_transfer=kv_transfer,
            ec_transfer=ec_transfer,
            hook_present=hook is not _MISSING and hook is not None,
            local_rank=_as_int(_attr(_attr(hook, "gate"), "rank")),
            hook_invalid=opt_bool(hook, "invalid"),
            hook_active=_as_bool(_attr(hook, "active") is not None),
            hook_tp_size=_as_int(_attr(hook, "tp_size")),
            plan_size=_as_int(_attr(_attr(hook, "plan"), "size")),
            plan_depth=_as_int(_attr(_attr(hook, "plan"), "depth")),
            verified_graph=_as_bool(_attr(hook, "verified_graph") is not None),
            last_completed_round=_tuple_or_none(_attr(hook, "_last_completed_round")),
            hook_request=_attr(hook, "request_id"),
            hook_next_position=_as_int(_attr(hook, "next_position")),
            bridge_request=_attr(bridge, "_request"),
            bridge_position=_as_int(_attr(bridge, "_position")),
            bridge_poisoned=opt_bool(bridge, "_poisoned"),
            max_context_tokens=_as_int(_attr(bridge, "max_context_tokens")),
            batch_num_reqs=_as_int(_attr(batch, "num_reqs")),
            batch_req_ids=_sequence(_attr(batch, "req_ids")),
            batch_num_computed=_as_int(_first(_attr(batch, "num_computed_tokens_cpu"))),
            batch_num_prompt=_as_int(_first(_attr(batch, "num_prompt_tokens"))),
            optimistic_seq_len=_as_int(_first(_attr(runner, "optimistic_seq_lens_cpu"))),
            scheduler_total_tokens=_as_int(
                _attr(scheduler_output, "total_num_scheduled_tokens")
            ),
            scheduler_tokens=_token_items(_attr(scheduler_output, "num_scheduled_tokens")),
            scheduler_spec_tokens=_flag_len(
                scheduler_output, "scheduled_spec_decode_tokens"
            ),
            scheduler_new_reqs=_flag_len(scheduler_output, "scheduled_new_reqs"),
            scheduler_finished_ids=_optional_sequence(
                _attr(scheduler_output, "finished_req_ids")
            ),
            scheduler_resumed_ids=_optional_sequence(
                _attr(rec, "resumed_req_ids")
            ),
            scheduler_preempted_ids=_optional_sequence(
                _attr(scheduler_output, "preempted_req_ids")
            ),
            scheduler_sync_kv_loads=_as_bool(
                _attr(scheduler_output, "has_sync_kv_loads", False)
            ),
            num_tokens=num_tokens,
            runtime_mode=runtime_mode,
            capturing=capturing,
            query_start_loc=tuple(int(v) for v in query) if query else None,
            gdn_query_start_loc=(
                tuple(int(v) for v in gdn_query) if gdn_query else None
            ),
            has_gdn=gdn,
        )


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rollout:
    """Feature rollout state, normally loaded from the environment once."""

    mode: str = MODE_OFF
    candidate: str = CANDIDATE_DEFER
    min_shadow: int = DEFAULT_MIN_SHADOW
    allow_unverified_on: bool = False
    allow_pre_replay: bool = False

    @classmethod
    def from_env(cls):
        def flag(name):
            return os.environ.get(name, "0").strip().lower() in (
                "1", "true", "yes", "on"
            )

        mode = os.environ.get("MK_GLM53_NATIVE_BUILDER_DEFERRAL", MODE_OFF).strip().lower()
        if mode not in _MODES:
            mode = MODE_OFF
        candidate = os.environ.get("MK_GLM53_EARLY_CANDIDATE", CANDIDATE_DEFER).strip().lower()
        if candidate not in _CANDIDATES:
            candidate = CANDIDATE_DEFER
        try:
            min_shadow = max(0, int(os.environ.get(
                "MK_GLM53_EARLY_MIN_SHADOW", str(DEFAULT_MIN_SHADOW))))
        except ValueError:
            min_shadow = DEFAULT_MIN_SHADOW
        return cls(
            mode=mode,
            candidate=candidate,
            min_shadow=min_shadow,
            allow_unverified_on=flag("MK_GLM53_EARLY_ALLOW_UNVERIFIED_ON"),
            allow_pre_replay=flag("MK_GLM53_EARLY_ALLOW_PRE_REPLAY"),
        )

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(f"unknown deferral mode {self.mode!r}")
        if self.candidate not in _CANDIDATES:
            raise ValueError(f"unknown candidate {self.candidate!r}")
        if self.min_shadow < 0:
            raise ValueError("min_shadow must be non-negative")


@dataclass(frozen=True)
class Admission:
    allowed: bool
    reason: str
    schedule: Schedule | None
    shadow: bool
    candidate: str


def _decline(reason, schedule=None, *, shadow=False, candidate=CANDIDATE_DEFER):
    return Admission(False, reason, schedule, shadow, candidate)


def evaluate(facts, rollout, *, shadow_steps=0, shadow_clean=True,
             forced_miss=None):
    """Fail-closed admission for one step.

    ``forced_miss`` is a deterministic test/diagnostic switch matching the
    upstream ``MK_GLM53_NATIVE_BUILDER_MISS`` convention ("rank:phase").
    """
    sched = _schedule(facts)

    if rollout.mode == MODE_OFF:
        return _decline("deferral mode off", sched)
    if not facts.hook_present:
        return _decline("no MK EP hook", sched)
    if facts.topology != TP1_DP8_EP8:
        return _decline(f"topology {facts.topology} is not TP1/DP8/EP8", sched)
    if facts.plan_size != 8 or facts.plan_depth != 45:
        return _decline(
            f"plan is {facts.plan_depth}xEP{facts.plan_size}, not 45xEP8", sched
        )
    if facts.hook_tp_size != 1:
        return _decline(f"hook tp_size {facts.hook_tp_size} is not 1", sched)

    # Structural disqualifiers that make the native metadata build matter or
    # make the CPU block tables ambiguous.
    disqualifiers = (
        ("hook invalid after an earlier failure", facts.hook_invalid),
        ("a previous EP round is still active", facts.hook_active),
        ("MK graph not verified", not facts.verified_graph),
        ("no completed previous MK round", facts.last_completed_round is None),
        ("bridge poisoned", facts.bridge_poisoned),
        ("speculative decoding enabled", facts.speculative),
        ("async scheduling enabled", facts.async_scheduling),
        ("async spec decode enabled", facts.async_spec_decode),
        ("DCP enabled", facts.dcp),
        ("dynamic EPLB enabled", facts.dynamic_eplb),
        ("device metadata submission in flight", facts.metadata_executor_in_flight),
        ("device metadata capture producers present",
         facts.metadata_executor_capture_producers),
        ("long-sequence metadata present", facts.long_seq_metadata),
        ("sparse KV offload enabled", facts.sparse_kv_offload),
        ("multimodal prefix LM enabled", facts.mm_prefix_lm),
        ("kvpp scheduler enabled", facts.kvpp_scheduler),
        ("prefix caching enabled", facts.prefix_caching),
        ("mamba cache mode is not none", facts.mamba_cache_mode != "none"),
        ("deferred state corrections pending", facts.deferred_state_corrections),
        ("compress attention enabled", facts.use_compress),
        ("decoder replay present", facts.decoder_replay is not None),
        ("decoder replay active", facts.replay_active),
        ("prefill in progress", facts.with_prefill),
        ("encoder inputs scheduled", (facts.encoder_inputs or 0) > 0),
        ("CPU offload active", facts.offload_active),
        ("KV transfer active", facts.kv_transfer),
        ("EC transfer active", facts.ec_transfer),
    )
    for reason, tripped in disqualifiers:
        if tripped:
            return _decline(reason, sched)

    # Execution geometry: one unpadded eager token (or a proven pre-replay
    # intercept for candidate A).
    if facts.num_tokens != 1:
        return _decline(f"num_tokens {facts.num_tokens} is not 1", sched)
    if facts.capturing:
        return _decline("inside graph capture", sched)
    if facts.runtime_mode not in EAGER_RUNTIME_MODES:
        if not (rollout.candidate == CANDIDATE_INTERCEPT
                and rollout.allow_pre_replay
                and facts.runtime_mode in GRAPH_RUNTIME_MODES):
            return _decline(
                f"runtime mode {facts.runtime_mode} requires pre-replay intercept",
                sched,
            )

    if sched is None:
        return _decline("not a single-token continuation", sched)
    if facts.query_start_loc is not None and facts.query_start_loc[:2] != (0, 1):
        return _decline(f"query boundaries {facts.query_start_loc[:2]}", sched)
    if facts.has_gdn:
        if facts.gdn_query_start_loc is None:
            return _decline("missing GDN query boundaries", sched)
        if facts.gdn_query_start_loc[:2] != (0, 1):
            return _decline(
                f"GDN query boundaries {facts.gdn_query_start_loc[:2]}", sched
            )
    if facts.optimistic_seq_len is not None and facts.optimistic_seq_len != sched.position + 1:
        return _decline(
            f"optimistic seq len {facts.optimistic_seq_len} != {sched.position + 1}",
            sched,
        )
    if forced_miss in (
        f"{facts.local_rank}:{sched.phase}",
        f"{facts.bridge_request}:{sched.phase}",
        sched.phase,
    ):
        return _decline("forced local miss", sched)

    # Rollout verification gate.  ``on`` cannot skip verification silently:
    # until this process has accumulated ``min_shadow`` clean shadow steps it
    # keeps building the native metadata and cross-checking the plan, then
    # flips to the deferring fast path.  A single mismatch disables the fast
    # path for the rest of the process.
    shadow = rollout.mode == MODE_SHADOW
    reason = "proven single-token continuation"
    if rollout.mode == MODE_ON:
        if not shadow_clean:
            return _decline("a shadow mismatch disabled the fast path", sched)
        if shadow_steps >= rollout.min_shadow or rollout.allow_unverified_on:
            shadow = False
        else:
            shadow = True
            reason = (
                f"shadow warmup {shadow_steps}/{rollout.min_shadow} "
                "clean steps before deferral"
            )
    return Admission(True, reason, sched, shadow, rollout.candidate)


def _schedule(facts):
    """Return the claimed Schedule, or None when any identity is unproven."""
    if facts.batch_num_reqs != 1 or facts.batch_req_ids is None:
        return None
    if len(facts.batch_req_ids) != 1:
        return None
    rid = facts.batch_req_ids[0]
    if not isinstance(rid, str) or not rid:
        return None
    if facts.scheduler_total_tokens != 1:
        return None
    if facts.scheduler_tokens != ((rid, 1),):
        return None
    if facts.scheduler_spec_tokens or facts.scheduler_new_reqs:
        return None
    if rid in (facts.scheduler_finished_ids or ()):
        return None
    if rid in (facts.scheduler_resumed_ids or ()):
        return None
    if rid in (facts.scheduler_preempted_ids or ()):
        return None
    if facts.scheduler_sync_kv_loads:
        return None
    position, prompt = facts.batch_num_computed, facts.batch_num_prompt
    if position is None or prompt is None:
        return None
    if not 0 < position < (facts.max_context_tokens or 0):
        return None
    if position < prompt:
        return None
    if (facts.hook_request, facts.hook_next_position) != (rid, position):
        return None
    if facts.last_completed_round != (rid, position - 1):
        return None
    if (facts.bridge_request, facts.bridge_position) != (rid, position - 1):
        return None
    return Schedule(rid, position, prompt, 1)


# ---------------------------------------------------------------------------
# Global role proof
# ---------------------------------------------------------------------------


def prove_roles(rows):
    """Prove one live owner and seven idle companions from the select rows."""
    rows = tuple(rows)
    if len(rows) != 8:
        return False, f"select returned {len(rows)} rows, expected 8"
    bad = [r for r in rows if r not in _ROLES]
    if bad:
        return False, f"unknown rank role(s) {bad}"
    if rows.count(ROLE_REAL) != 1:
        return False, f"{rows.count(ROLE_REAL)} real owners, expected 1"
    if rows.count(ROLE_IDLE) != 7:
        return False, f"{rows.count(ROLE_IDLE)} idle companions, expected 7"
    if rows.count(ROLE_NATIVE) != 0:
        return False, "a rank selected native"
    return True, "one live one-token owner and seven idle companions"


def owner_rank(rows):
    rows = tuple(rows)
    return rows.index(ROLE_REAL) if rows.count(ROLE_REAL) == 1 else None


# ---------------------------------------------------------------------------
# Deferred native metadata
# ---------------------------------------------------------------------------


class DeferredNativeMetadata:
    """One-shot deferred build with the guarantees the runner needs.

    * ``materialize()`` builds at most once and returns the same result on a
      repeated call.
    * A failed build cannot be retried in the same round; it releases the
      previous producer submission before re-raising.
    * ``finish()`` releases exactly once after the step, including the case
      where the metadata was never needed.

    ``build`` must do the full native job (``_build_attention_metadata`` plus
    ``_prepare_device_metadata_for_forward``).  ``join`` receives the producer
    stream when one is active; ``release`` releases the submission.
    """

    def __init__(self, build, *, join=None, release=None, meter=None):
        if not callable(build):
            raise TypeError("build must be callable")
        self._build = build
        self._join = join
        self._release = release
        self._meter = meter
        self._result = None
        self._attempted = False
        self._finished = False
        self.builds = 0

    @property
    def attempted(self):
        return self._attempted

    @property
    def built(self):
        return self._result is not None

    def materialize(self):
        if self._result is not None:
            return self._result
        if self._attempted:
            raise RuntimeError(
                "A failed native metadata build cannot be retried in this round"
            )
        self._attempted = True
        started = _now()
        try:
            result = self._build()
            if result is None:
                raise RuntimeError("deferred native metadata build returned None")
            if self._join is not None:
                self._join()
            self._result = result
            self.builds += 1
            if self._meter is not None:
                self._meter.record("deferred_materialize", _now() - started)
            return result
        except BaseException:
            # The submission may have been armed before the producer callbacks
            # ran.  Release it before propagating so the next round can build.
            if self._release is not None and not self._finished:
                self._finished = True
                try:
                    self._release()
                except Exception:
                    pass
            raise

    def finish(self):
        if self._finished:
            return
        self._finished = True
        if self._release is not None:
            self._release()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _now():
    return time.perf_counter_ns()


@dataclass
class PhaseMeter:
    """Nanosecond phase totals for the host timeline around the fast path."""

    phase_ns: dict = field(default_factory=dict)

    def record(self, name, delta_ns):
        self.phase_ns[name] = self.phase_ns.get(name, 0) + int(delta_ns)

    def snapshot(self):
        return dict(self.phase_ns)


@dataclass
class StepLedger:
    """Exact per-step accounting; native and MK can never both be counted."""

    admitted: int = 0
    declined: int = 0
    deferred: int = 0
    shadow_checks: int = 0
    shadow_matches: int = 0
    shadow_misses: int = 0
    shadow_clean: bool = True
    shadow_steps: int = 0
    materialized: int = 0
    fast_missed: int = 0
    mk_steps: int = 0
    native_steps: int = 0
    idle_steps: int = 0
    invalid_steps: int = 0
    native_build_calls: int = 0
    native_build_skipped: int = 0
    selected_rounds: int = 0
    role_failures: int = 0

    def on_admission(self, prepared):
        if prepared.action == "native":
            self.declined += 1
            return
        self.admitted += 1
        if prepared.action == "defer":
            self.deferred += 1

    def on_select(self, rows):
        self.selected_rounds += 1
        ok, _ = prove_roles(rows)
        if not ok:
            self.role_failures += 1
        return ok

    def on_shadow(self, equal):
        self.shadow_steps += 1
        self.shadow_checks += 1
        if equal:
            self.shadow_matches += 1
        else:
            self.shadow_misses += 1
            self.shadow_clean = False

    def on_materialize(self):
        self.materialized += 1

    def on_native_build(self):
        self.native_build_calls += 1

    def on_build_skipped(self):
        self.native_build_skipped += 1

    def on_step(self, *, native_forward_ran, mk_step_ran,
                deferred_this_step=False):
        if native_forward_ran and mk_step_ran:
            self.invalid_steps += 1
            raise RuntimeError(
                "a step cannot record both a native forward and an MK step"
            )
        if mk_step_ran:
            self.mk_steps += 1
        elif native_forward_ran:
            self.native_steps += 1
            if deferred_this_step:
                # A deferred (would-be-fast) step that selected native is a
                # real miss; a shadow step is expected to run native.
                self.fast_missed += 1
        else:
            self.idle_steps += 1

    def snapshot(self):
        return {
            "admitted": self.admitted,
            "declined": self.declined,
            "deferred": self.deferred,
            "shadow_checks": self.shadow_checks,
            "shadow_matches": self.shadow_matches,
            "shadow_misses": self.shadow_misses,
            "shadow_clean": self.shadow_clean,
            "shadow_steps": self.shadow_steps,
            "materialized": self.materialized,
            "fast_missed": self.fast_missed,
            "mk_steps": self.mk_steps,
            "native_steps": self.native_steps,
            "idle_steps": self.idle_steps,
            "invalid_steps": self.invalid_steps,
            "native_build_calls": self.native_build_calls,
            "native_build_skipped": self.native_build_skipped,
            "selected_rounds": self.selected_rounds,
            "role_failures": self.role_failures,
        }


# ---------------------------------------------------------------------------
# Plan / orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Validated CPU addresses for exactly one token.

    ``addresses`` is the tuple the bridge consumes (KDA slots, MLA pages,
    indexer addresses).  ``reader_plan`` is the upstream ``StepAddressPlan``
    retained for shadow comparison and logging.
    """

    request_id: str
    position: int
    addresses: object
    reader_plan: object = None

    def matches(self, schedule):
        return (self.request_id, self.position) == (
            schedule.request_id, schedule.position
        )


@dataclass(frozen=True)
class Prepared:
    """Outcome for one step.  ``action`` decides what the runner does next."""

    action: str          # "native" | "shadow" | "defer"
    reason: str
    admission: Admission
    plan: Plan | None = None
    deferred: DeferredNativeMetadata | None = None

    @property
    def admitted(self):
        return self.action != "native"

    @property
    def skips_native_build(self):
        return self.action == "defer"

    def materialize(self):
        if self.deferred is None:
            raise RuntimeError("no deferred native metadata to materialize")
        return self.deferred.materialize()

    def finish(self):
        if self.deferred is not None:
            self.deferred.finish()


class EarlyRunner:
    """Admission + deferral orchestrator used by the runner wiring.

    ``plan_reader(facts, schedule) -> Plan`` must read only CPU state and fail
    with an exception when it cannot prove the addresses.  ``build_native`` is
    the same thunk the runner would otherwise call; it is invoked lazily.
    """

    def __init__(self, rollout, *, plan_reader, meter=None, ledger=None,
                 clock=_now):
        if not callable(plan_reader):
            raise TypeError("plan_reader must be callable")
        self.rollout = rollout
        self._plan_reader = plan_reader
        self.meter = meter or PhaseMeter()
        self.ledger = ledger or StepLedger()
        self._clock = clock
        self._last = None

    @property
    def last(self):
        return self._last

    def prepare(self, facts, *, build_native=None, forced_miss=None):
        started = self._clock()
        admission = evaluate(
            facts,
            self.rollout,
            shadow_steps=self.ledger.shadow_steps,
            shadow_clean=self.ledger.shadow_clean,
            forced_miss=forced_miss,
        )
        self.meter.record("admit", self._clock() - started)
        if not admission.allowed:
            prepared = Prepared("native", admission.reason, admission)
            self.ledger.on_admission(prepared)
            self._last = prepared
            return prepared

        started = self._clock()
        try:
            plan = self._plan_reader(facts, admission.schedule)
            if plan is None:
                raise _Declined("plan reader returned no plan")
            if not plan.matches(admission.schedule):
                raise _Declined(
                    f"plan identity {(plan.request_id, plan.position)} does not "
                    f"match {(admission.schedule.request_id, admission.schedule.position)}"
                )
        except Exception as exc:
            self.meter.record("cpu_plan", self._clock() - started)
            reason = f"CPU plan miss: {type(exc).__name__}: {exc}"
            prepared = Prepared("native", reason, admission)
            self.ledger.on_admission(prepared)
            self._last = prepared
            return prepared
        self.meter.record("cpu_plan", self._clock() - started)

        if admission.shadow:
            prepared = Prepared("shadow", admission.reason, admission, plan)
        else:
            if build_native is None:
                # Without a build thunk the deferred fallback cannot exist, so
                # the only safe outcome is a normal native step.
                prepared = Prepared(
                    "native",
                    "no native build thunk for deferred fallback",
                    admission,
                )
                self.ledger.on_admission(prepared)
                self._last = prepared
                return prepared
            deferred = DeferredNativeMetadata(
                build_native,
                join=None,
                release=None,
                meter=self.meter,
            )
            prepared = Prepared("defer", admission.reason, admission,
                                plan, deferred)
        self.ledger.on_admission(prepared)
        self._last = prepared
        return prepared


# ---------------------------------------------------------------------------
# Helpers used by the wiring diff
# ---------------------------------------------------------------------------


def addresses_equal(cpu_addresses, native_addresses):
    """Compare the CPU plan with the native addresses; return (equal, detail)."""
    try:
        cpu = tuple(cpu_addresses)
        native = tuple(native_addresses)
    except TypeError:
        return False, "address tuples are not iterable"
    if len(cpu) != len(native):
        return False, f"address arity {len(cpu)} != {len(native)}"
    for label, left, right in zip(("KDA", "MLA", "Indexer"), cpu, native):
        if left.keys() != right.keys():
            return False, f"{label} layer keys differ"
        for layer in right:
            if left[layer] != right[layer]:
                return False, f"{label}[{layer}] differs"
    return True, "CPU plan equals native addresses"


def bind_forward_context(context, prepared):
    """Expose the plan to the MK hook without importing runner internals."""
    if prepared.action == "native":
        return
    schedule = prepared.admission.schedule
    if schedule is None:
        return
    context.glm53_cpu_addresses = (
        (schedule.request_id, schedule.position),
        prepared.plan.addresses,
    )
    context.glm53_early_action = prepared.action


def check_forward_context(context):
    """Return the bound fast-path plan, or None when the hook must go native."""
    bound = _attr(context, "glm53_cpu_addresses")
    if bound is None or bound is _MISSING:
        return None
    return bound


def pre_replay_intercept_allowed(facts, rollout):
    """Candidate A predicate; fail-closed unless explicitly opted in.

    Returns ``(allowed, reason)``.  Even when allowed, the caller must still
    run ``evaluate`` so every other invariant is enforced.
    """
    if rollout.candidate != CANDIDATE_INTERCEPT or not rollout.allow_pre_replay:
        return False, "pre-replay intercept is not opted in"
    if facts.runtime_mode not in GRAPH_RUNTIME_MODES:
        return False, f"runtime mode {facts.runtime_mode} is not a graph mode"
    if facts.capturing:
        return False, "inside graph capture"
    if facts.num_tokens != 1:
        return False, "not a single-token step"
    return True, "graph-mode intercept permitted by rollout"
