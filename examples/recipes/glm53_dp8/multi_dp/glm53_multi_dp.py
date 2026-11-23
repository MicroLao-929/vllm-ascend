"""Native-prefill, multi-origin B1 ordinary decode on eight local EP/DP ranks."""
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import struct

LOG = logging.getLogger(__name__)
CAPABILITY = "glm53.tp1.dp8.ep8.multi_source_b1.v1"
# Share the existing strictly ordered rank-local IPC stream. Each protocol record
# has its own magic, rank, epoch and explicit phase. Long UTF-8 identities use an
# object exchange by unanimous agreement; IDs are never truncated or hashed.
_HEADER = struct.Struct("<4sQBBBiH")
_PHASE = {"select":1, "ready":2, "complete":3}
_NATIVE, _REAL, _IDLE, _MIGRATION, _ERROR, _DETAIL, _POISONED = range(7)
_OK, _DECLINE, _FATAL = range(3)


class BrokenRound(RuntimeError):
    """No safe native retry after protocol damage, an NPU error, or launch."""


class _Native(Exception):
    """All ranks declined before the first device graph launch."""


@dataclass(frozen=True)
class Source:
    rank: int
    request: str
    position: int


@dataclass(frozen=True)
class Plan:
    sources: tuple
    reason: str = ""
    rearm: bool = False

    @property
    def owners(self):
        return tuple(item.rank for item in self.sources)


@dataclass
class Active:
    plan: Plan
    readiness_started: bool = False
    launch_allowed: bool = False


def _validate(rows):
    if len(rows) != 8 or any(not isinstance(row, (tuple, list)) or len(row)!=3 for row in rows):
        raise BrokenRound("invalid multi-DP selection frame count or arity")
    for role,rid,pos in rows:
        if role not in (_NATIVE,_REAL,_IDLE,_MIGRATION,_ERROR,_POISONED):
            raise BrokenRound(f"invalid multi-DP selection role {role!r}")
        if role == _REAL and (not isinstance(rid,str) or not rid or type(pos) is not int or pos<=0):
            raise BrokenRound("invalid source request or position")
    return tuple(Source(rank,rid,pos) for rank,(role,rid,pos) in enumerate(rows) if role==_REAL)


class Group:
    def __init__(self,gate):
        self.gate = gate
        self.rank = gate.rank
        self.channel = getattr(gate,"local_control",None)
        if gate.size != 8:
            raise ValueError("multi-origin decode requires an eight-rank data parallel group")
        if self.channel is not None and (self.channel.size!=8 or self.channel.rank!=self.rank
                                         or self.channel.payload_size<=_HEADER.size):
            raise ValueError("multi-origin IPC topology or message width mismatch")
        self.last_sources = {}

    @property
    def epoch(self):
        return self.gate.sequence

    def _exchange(self,phase,local):
        if self.channel is None:
            return self.gate.gather("multi_"+phase,local)
        role,rid,pos = local
        encoded = rid.encode("utf-8", errors="surrogatepass") if isinstance(rid,str) else b""
        width = self.channel.payload_size
        if len(encoded)>width-_HEADER.size or (pos is not None and not 0 <= pos < (1<<31)):
            role,encoded,pos = _DETAIL,b"",None
        raw = bytearray(width)
        _HEADER.pack_into(raw,0,b"GDM1",self.epoch,_PHASE[phase],self.rank,role,
                          -1 if pos is None else pos,len(encoded))
        raw[_HEADER.size:_HEADER.size+len(encoded)] = encoded
        wire = self.channel.all_gather(bytes(raw))
        if not isinstance(wire,(bytes,bytearray)) or len(wire)!=8*width:
            raise BrokenRound("truncated multi-origin collective")
        rows = []
        for rank in range(8):
            off=rank*width
            magic,epoch,phase_id,sender,role,pos,n=_HEADER.unpack_from(wire,off)
            if (magic,epoch,phase_id,sender)!=(b"GDM1",self.epoch,_PHASE[phase],rank) or n>width-_HEADER.size:
                raise BrokenRound("multi-origin magic/epoch/phase/rank mismatch")
            start=off+_HEADER.size
            value=wire[start:start+n].decode("utf-8",errors="surrogatepass") if n else None
            rows.append((role,value,None if pos==-1 else pos))
        if any(r[0]==_DETAIL for r in rows):
            return self.gate.gather("multi_"+phase+"_details",local)
        return rows

    def select(self,local):
        self.gate.sequence += 1
        rows=self._exchange("select",local)
        sources=_validate(rows)
        roles=[r[0] for r in rows]
        if _POISONED in roles:
            raise BrokenRound("a peer reported a failed or overlapping generation")
        if _ERROR in roles:
            return Plan((),"admission error: "+", ".join(f"rank{rank}={r[1]}" for rank,r in enumerate(rows) if r[0]==_ERROR))
        if _MIGRATION in roles:
            return Plan((),"request migration or resumed lifetime")
        ids=[src.request for src in sources]
        if len(set(ids))!=len(ids):
            return Plan((),"same request appears on multiple ranks")
        if any(self.last_sources.get(src.request,src.rank)!=src.rank for src in sources):
            # A native step re-establishes the destination worker's cache ownership.
            self.last_sources={}
            return Plan((),"request changed origin")
        if _NATIVE in roles or not sources:
            return Plan((),"native participant or no real tokens",rearm=True)
        return Plan(sources)

    def voted(self,phase,code,reason=None):
        # Success needs one fixed-size exchange. Details on failures are cold path.
        rows=self._exchange(phase,(code,None,None))
        if any(r[0] not in (_OK,_DECLINE,_FATAL) or r[1:] not in ((None,None),[None,None]) for r in rows):
            raise BrokenRound("invalid multi-origin vote")
        severity=max(r[0] for r in rows)
        if severity:
            reasons=self.gate.gather("multi_"+phase+"_reason",reason)
            return severity, "; ".join(f"rank{rank}={v}" for rank,v in enumerate(reasons) if v)
        return _OK,""


def load(hook, runner, *, receipts, trace, synchronize):
    """Bootstrap validates the selected backend on every rank before accepting work."""
    error=None
    backend=None
    try:
        from inspect import signature
        from blockrt.models.glm_5_3_flash import multi_dp as backend
        if (backend.CAPABILITY!=CAPABILITY or not callable(backend.source_receipts)
                or "active_owners" not in signature(type(hook.companion).run).parameters
                or not hasattr(type(hook.companion),"receipt_graphs") or hook.tp_size!=1 or hook.plan.size!=8 or hook.plan.depth!=45
                or runner.parallel_config.data_parallel_size!=8
                or runner.parallel_config.data_parallel_size_local!=8 or runner.max_num_reqs<1
                or runner.speculative_config is not None or runner.use_async_scheduling
                or not runner.model_config.enforce_eager):
            raise ValueError("unsupported multi-origin graph ABI or runner geometry")
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"
    # FastRoundGate.check only raises on its known hot phases; use gather explicitly.
    errors=hook.gate.gather("multi_capabilities",error)
    if any(errors):
        raise RuntimeError(f"unavailable multi-origin execution capability: {errors}")
    return Driver(Group(hook.gate),backend,receipts,trace,synchronize)


class Driver:
    def __init__(self,group,backend,receipts,trace,synchronize):
        self.group,self.backend,self.receipts,self.trace,self.synchronize=group,backend,receipts,trace,synchronize
        self.disabled=False
        self.pending_native=None
        self.log_counts={}
        audit=os.getenv("GLM53_MULTI_AUDIT")
        self.audit_file=(Path(audit)/f"multi-dp-{os.getpid()}-rank{group.rank}.jsonl"
                         if audit and (trace or receipts) else None)

    def _vote(self,hook,runner,output,idle,context,num_tokens):
        if output is not None:
            # A finished request different from our current one may be reported
            # normally. Only explicit local migration invalidates this selection.
            rid=runner.input_batch.req_ids[0] if runner.input_batch.num_reqs==1 else None
            resumed=output.scheduled_cached_reqs.resumed_req_ids or ()
            preempted=getattr(output,"preempted_req_ids",None) or ()
            if rid is not None and (rid in resumed or rid in preempted or rid in output.finished_req_ids):
                return _MIGRATION,None,None
        role,rid,pos=hook._vote(runner,output,idle,context,num_tokens)
        if role=="real" and runner.input_batch.req_id_to_index.get(rid)!=0:
            return _NATIVE,None,None
        return {"real":_REAL,"idle":_IDLE,"native":_NATIVE}[role],rid,pos

    def _event(self,outcome,plan,local,source=None,receipts=None):
        key=(outcome,len(plan.sources))
        count=self.log_counts.get(key,0)+1
        self.log_counts[key]=count
        if not (self.trace or self.receipts or count in (1,64,256,1024)):
            return
        event={"epoch":self.group.epoch,"rank":self.group.rank,"outcome":outcome,"owners":plan.owners,
               "sources":[(s.rank,s.request,s.position) for s in plan.sources],
               "role":"real" if local[0]==_REAL else "companion" if local[0]==_IDLE else "native",
               "request":local[1],"position":local[2],"count":count}
        if plan.reason:event["reason"]=plan.reason
        if source is not None:event["local_graph"]="full" if source else "companion"
        if receipts is not None:event["receipts"]=receipts
        line=json.dumps(event,separators=(",",":"),ensure_ascii=True)
        if self.audit_file is not None:
            try:
                with self.audit_file.open("a",encoding="utf8") as stream:
                    stream.write(line+"\n")
            except OSError as exc:
                LOG.error("GLM53 cannot save per-rank multi-origin audit %s: %s",self.audit_file,exc)
                self.audit_file=None
        LOG.info("GLM53 multi DP step %s",line)

    def _ready_vote(self,hook,code,reason=None):
        active=hook.active
        if active is None or active.readiness_started:
            raise BrokenRound("duplicate or out-of-phase multi-origin launch")
        active.readiness_started=True
        try:
            verdict,why=self.group.voted("ready",code,reason)
        except Exception as exc:
            hook.invalid=True
            raise BrokenRound("multi-origin readiness transport failed") from exc
        if verdict==_FATAL:
            hook.invalid=True
            raise BrokenRound("pre-launch device error: "+why)
        if verdict==_DECLINE:raise _Native(why)
        try:hook.runtime.barrier()
        except Exception as exc:
            hook.invalid=True
            raise BrokenRound("multi-origin start barrier failed") from exc
        active.launch_allowed=True

    def ready(self,hook):
        try:self.synchronize()
        except Exception as exc:return self._ready_vote(hook,_FATAL,str(exc))
        return self._ready_vote(hook,_OK)

    def observed_native(self,hook,runner,output):
        pending,self.pending_native=self.pending_native,None
        if pending is None or self.disabled:return
        scheduled=hook._schedule(runner,output)
        if scheduled is not None and scheduled[:2]==pending and scheduled[3]==1:
            rid,position=pending
            # The native forward wrote its own VA caches. Drop private views so
            # the next admitted token imports the complete native state again.
            hook.bridge.release(rid)
            hook.request_id,hook.next_position=rid,position+1

    def _native(self,hook,context,local,plan):
        if plan.rearm and local[0]==_REAL:
            self.pending_native=(local[1],local[2])
        if hasattr(hook,"_materialize_early_metadata"):
            hook._materialize_early_metadata(context)
        self._event("native",plan,local)
        return None

    def forward(self,hook,runner,output,idle,context,num_tokens,input_ids,positions):
        self.pending_native=None
        hook.result=None
        if self.disabled:
            if hasattr(hook,"_materialize_early_metadata"):hook._materialize_early_metadata(context)
            return None
        if hook.invalid or hook.active is not None:
            local=(_POISONED,None,None)
        else:
            try:local=self._vote(hook,runner,output,idle,context,num_tokens)
            except Exception as exc:local=(_ERROR,f"{type(exc).__name__}: {exc}",None)
        try:plan=self.group.select(local)
        except Exception:
            hook.invalid=True
            raise
        if not plan.sources:
            if plan.rearm and output is not None and local[0]==_NATIVE:
                scheduled=hook._schedule(runner,output)
                if scheduled is not None:
                    rid,p,prompt,count=scheduled
                    if (count==1 and 0<prompt<=p<hook.bridge.max_context_tokens
                            and runner.input_batch.req_id_to_index.get(rid)==0
                            and rid not in output.finished_req_ids
                            and not output.scheduled_cached_reqs.resumed_req_ids
                            and not (getattr(output,"preempted_req_ids",None) or ())):
                        self.pending_native=(rid,p)
            return self._native(hook,context,local,plan)
        active=Active(plan)
        hook.active=active
        result=local_receipts=None
        error=None
        try:
            if idle:
                hook.companion.check_ready()
                self.ready(hook)
                receipt=hook.companion.run(active_owners=plan.owners)
                if tuple(r.layer_id for r in receipt)!=tuple(hook.plan.moe):
                    raise BrokenRound("companion layer receipt mismatch")
                result=hook.idle_hidden
            else:
                hook.result=hook.try_decode(runner,output,context,positions,input_ids)
                if hook.result is None:
                    raise ValueError("owner declined its selected token")
                if not active.launch_allowed:
                    raise BrokenRound("decoder returned without the pre-launch rendezvous")
                result=hook.result.hidden_states
            self.synchronize()
            if self.receipts:
                graphs=hook.companion.receipt_graphs if idle else (hook.bridge._model._last_graph,)
                local_receipts=self.backend.source_receipts(graphs,rank=self.group.rank,active_owners=plan.owners)
                if tuple(item["layer_id"] for item in local_receipts)!=tuple(hook.plan.moe):
                    raise ValueError("source receipt layer set/order mismatch")
        except _Native as exc:
            error=exc
        except Exception as exc:
            if active.launch_allowed:
                error=exc
            elif not active.readiness_started:
                try:self.synchronize()
                except Exception as device_exc:
                    self._ready_vote(hook,_FATAL,str(device_exc))
                try:self._ready_vote(hook,_DECLINE,f"{type(exc).__name__}: {exc}")
                except _Native as cancelled:error=cancelled
            else:
                hook.invalid=True
                raise
        finally:
            hook.active=None
        if isinstance(error,_Native):
            # The vote preceded every device graph. Native destinations were not
            # written. Some private bridges may have been aborted half-preparation;
            # retire MK on all ranks for this instance and never reuse them.
            self.disabled=True;hook.result=None
            return self._native(hook,context,local,Plan((),str(error)))
        try:
            verdict,why=self.group.voted("complete",_FATAL if error else _OK,str(error) if error else None)
            if verdict:raise BrokenRound("post-launch failure; native replay would double-write a token: "+why)
            hook.runtime.barrier()
        except Exception:
            hook.invalid=True;hook.result=None
            raise
        self.group.last_sources={s.request:s.rank for s in plan.sources}
        self._event("mk",plan,local,not idle,local_receipts)
        return result

