"""Real strict worker methods + real eight-process IPC/Gloo; device graph data are CPU test tensors."""
import ast
import importlib.util
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
os.environ.setdefault("OMP_NUM_THREADS","1")
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
ROOT=RAW=WORKER=OLD=md=None


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def checked(ok,msg):
    if not ok:raise RuntimeError(msg)


def framework():
    classes={}
    for path in [RAW/"glm53_mk_hook.py",WORKER/"glm53_mk_ep.py"]:
        classes.update({n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)})
    selected={"_Step":None,"GLM53MKHook":{"_release","update_requests","_schedule","observe_native","try_decode","process_logits"},
              "_CoordinatedDecodeState":None,"GLM53MKEPHook":{"_vote","observe_native","forward_round","ready_to_launch","take_step"}}
    body=[]
    for name,methods in selected.items():
        cls=classes[name]
        if methods:cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
        body.append(cls)
    ns={"torch":torch,"_require":checked,"dataclass":dataclass,"_LOG":logging.getLogger("test"),
        "logger":logging.getLogger("test"),"_log_hbm":lambda *args:None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"production-hook-methods","exec"),ns)
    return ns["GLM53MKEPHook"],ns["_CoordinatedDecodeState"]


class Gate:
    size=8
    def __init__(self,rank):self.rank=rank;self.sequence=0
    def gather(self,phase,payload):
        msgs=[None]*8
        dist.all_gather_object(msgs,(self.rank,self.sequence,phase,payload))
        assert [(r,e,p) for r,e,p,_ in msgs]==[(r,self.sequence,phase) for r in range(8)]
        return [row[-1] for row in msgs]
    def check(self,phase,error=None):
        checked(not any(self.gather(phase,error)),phase)


class Graph:
    """Each rank enters exactly one actual Gloo all-to-all pair for the whole epoch."""
    def __init__(self,rank,hook,counts):
        self.rank,self.hook,self.counts=rank,hook,counts
        self.nodes=[NS(layer_id=f"layer{layer}",op_name=x) for layer in range(3,45)
                    for x in ("moe_metadata_exchange","moe_dispatch","moe_combine")]
        self.last=None
    def run(self):
        plan=self.hook.active.plan
        ids=plan.owners
        s=next((s for s in plan.sources if s.rank==self.rank),None)
        self.counts["graph"]+=1
        if s:self.counts["full"]+=1
        else:self.counts["idle"]+=1
        tag=(s.rank+1)*10000+s.position if s else 0
        outgoing=torch.tensor([tag if tag else 0 for _ in range(8)],dtype=torch.int64)
        incoming=torch.empty_like(outgoing)
        dist.all_to_all_single(incoming,outgoing)
        assert [i for i,n in enumerate(incoming.tolist()) if n]==list(ids)
        returned=torch.where(incoming!=0,incoming+self.rank,0)
        owner_rows=torch.empty_like(outgoing)
        dist.all_to_all_single(owner_rows,returned)
        self.last=int(owner_rows.sum())
        if s:assert self.last==8*tag+28
        else:assert self.last==0


class Bridge:
    max_context_tokens=2048
    def __init__(self,hook,graph,proxy):
        self.hook,self.graph,self.proxy=hook,graph,proxy
        self._model=NS(_last_graph=graph)
        self.owned=None
        self.imports=[]
        self.committed=[]
        self.before=self.after=False
    def release(self,rid):
        if self.owned==rid:self.owned=None
    def step(self,context,*,request_id,position,input_ids,kda,mla,indexer):
        if self.before:raise ValueError("test prelaunch descriptor rejected")
        fresh=self.owned!=request_id
        if fresh:self.imports.append((request_id,position))
        self.proxy.run(self.graph)
        self.owned=request_id;self.committed.append((request_id,position))
        if self.after:raise ValueError("test postlaunch writeback failure")
        return NS(hidden_states=torch.tensor([[self.graph.last]]),logits=None,imported=fresh,graph_reused=not fresh)


def schedule(rank,rid,pos,prompt=128,count=1,resume=False):
    request=NS(mm_features=[],prompt_embeds=None,lora_request=None,sampling_params=NS(prompt_logprobs=None))
    runner=NS(max_num_reqs=2,input_batch=NS(num_reqs=1,req_ids=[rid],req_id_to_index={rid:0},num_computed_tokens_cpu=[pos],num_prompt_tokens=[prompt]),
              requests={rid:request},input_ids=NS(gpu=torch.tensor([11+rank])),
              model=NS(compute_logits=lambda h:h))
    output=NS(num_scheduled_tokens={rid:count},total_num_scheduled_tokens=count,scheduled_new_reqs=[],
              finished_req_ids=set(),scheduled_cached_reqs=NS(resumed_req_ids={rid} if resume else set()),
              preempted_req_ids=set(),scheduled_spec_decode_tokens={})
    return runner,output


def one_hook(cls,proxy_cls,gate,rank,proof_dir,audit=True):
    hook=cls.__new__(cls)
    hook.request_id=hook.next_position=hook.active=hook.verified_graph=hook.result=None
    hook.invalid=False;hook.tp_size=1;hook.plan=NS(moe=tuple(range(3,45)),depth=45,size=8)
    hook.gate=gate;hook.text=NS(config=NS(hidden_size=1,vocab_size=1));hook.idle_hidden=torch.tensor([[-1]])
    hook.runtime=NS(barrier=dist.barrier)
    hook._cache_descriptors=lambda runner:({},{},{})
    counts={"graph":0,"full":0,"idle":0}
    graph=Graph(rank,hook,counts)
    proxy=proxy_cls(NS(run=lambda g:g.run()),hook)
    hook.bridge=Bridge(hook,graph,proxy)
    def companion_run(active_owners):
        assert tuple(active_owners)==hook.active.plan.owners;graph.run()
        return tuple(NS(layer_id=layer) for layer in hook.plan.moe)
    hook.companion=NS(check_ready=lambda:None,run=companion_run,receipt_graphs=(graph,))
    def receipts(graphs,rank,active_owners):
        assert graphs==(graph,)
        return tuple({"layer_id":layer,"source_rows":[8 if i in active_owners else 0 for i in range(8)],
                      "received_by_source":[1 if i in active_owners else 0 for i in range(8)],
                      "received_rows":len(active_owners)} for layer in hook.plan.moe)
    hook.multi_dp=md.Driver(md.Group(gate),NS(source_receipts=receipts),audit,True,lambda:None)
    return hook,counts


def literal_step(hook,rank,owners,names,positions,*,prefill=None,resume=None,two_on=None):
    idle=rank not in owners and rank!=prefill
    if idle:
        runner,out=NS(),None;pos=token=None
    else:
        pos=positions[rank]
        count=128 if rank==prefill else 1
        runner,out=schedule(rank,names[rank],pos,128,count,rank==resume)
        if rank==two_on:
            rid2="actual-second-row"
            runner.input_batch.num_reqs=2;runner.input_batch.req_ids.append(rid2);runner.input_batch.req_id_to_index[rid2]=1
            out.num_scheduled_tokens[rid2]=1;out.total_num_scheduled_tokens=2
        hook.update_requests(out);token=runner.input_ids.gpu;pos=torch.tensor([pos])
    ctx=NS(cudagraph_runtime_mode=NS(name="NONE"),capturing=False)
    hidden=hook.forward_round(runner,out,idle,ctx,128 if rank==prefill else 2 if rank==two_on else 1,token,pos)
    if out is not None:
        if hidden is None:hook.observe_native(runner,out)
        else:
            step=hook.take_step();logits=hook.process_logits(runner,out,step,torch.tensor([0]))
            assert hidden.item()==logits.item()==8*((rank+1)*10000+positions[rank])+28
    elif hidden is not None:assert hidden.item()==-1
    return hidden is not None


def run(rank,rendezvous,directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+rendezvous,world_size=8,rank=rank)
    gate=Gate(rank);ctl=imported("ipc_real_"+str(rank),OLD/"glm53_mk_tp_control.py")
    gate.local_control=ctl.LocalTPControl.connect(gate,200)
    cls,proxy=framework()
    os.environ["GLM53_MULTI_AUDIT"]=directory
    fh=logging.FileHandler(Path(directory,f"trace-{rank}.jsonl"))
    md.LOG.addHandler(fh);md.LOG.setLevel(logging.INFO)
    hook,counts=one_hook(cls,proxy,gate,rank,directory)
    names=[f"request-{i}" for i in range(8)]
    positions=[128+i for i in range(8)]
    # Native prefills are independently committed by every rank.
    for r in range(8):
        if r==rank:
            runner,out=schedule(rank,names[r],0,positions[r],positions[r])
            hook.observe_native(runner,out)
    assert literal_step(hook,rank,range(8),names,positions)
    assert counts=={"graph":1,"full":1,"idle":0}
    positions=[p+1 for p in positions]
    assert literal_step(hook,rank,(1,4,7),names,positions)
    positions=[p+(i in (1,4,7)) for i,p in enumerate(positions)]
    assert literal_step(hook,rank,(3,),names,positions)
    positions[3]+=1
    # Rank 6's new prefill must collectively select native. Other active ranks
    # complete their native decode and re-import authoritative KV on resumption.
    mixed_names=names.copy();mixed_names[6]="prefill-new"
    p=positions.copy();p[6]=0
    assert not literal_step(hook,rank,range(8),mixed_names,p,prefill=6)
    positions=[v+1 for v in positions];positions[6]=128;names=mixed_names
    assert literal_step(hook,rank,range(8),names,positions)
    assert hook.bridge.imports[-1]==(names[rank],positions[rank])
    positions=[p+1 for p in positions]
    # A duplicated request identity never launches eight unrelated expert rounds.
    dup=names.copy();dup[2]=dup[1]
    if rank==2:hook.request_id=dup[2]
    before=counts["graph"]
    assert not literal_step(hook,rank,range(8),dup,positions)
    assert counts["graph"]==before
    # Restore fresh native lifetimes before migration and wide-ID cold protocol.
    names=[("long-request-"*30)+str(r) for r in range(8)];positions=[160+r for r in range(8)]
    hook.request_id,hook.next_position=names[rank],positions[rank]
    assert literal_step(hook,rank,range(8),names,positions)
    positions=[p+1 for p in positions]
    moved=names.copy();moved[0],moved[1]=moved[1],moved[0]
    hook.request_id,hook.next_position=moved[rank],positions[rank]
    before=counts["graph"]
    assert not literal_step(hook,rank,range(8),moved,positions)
    assert counts["graph"]==before
    hook.request_id,hook.next_position=names[rank],positions[rank]
    assert not literal_step(hook,rank,range(8),names,positions,resume=5)
    # Capacity 2 was used throughout. An actual second scheduled row on just one rank
    # forces all peers onto the ordinary native batch, with zero device graph calls.
    hook.request_id,hook.next_position=names[rank],positions[rank]
    before=counts["graph"]
    assert not literal_step(hook,rank,range(8),names,positions,two_on=4)
    assert counts["graph"]==before
    # One Python preparation failure before any launch returns native on all ranks.
    hook.request_id,hook.next_position=names[rank],positions[rank]
    hook.bridge.before=(rank==2);before=counts["graph"]
    assert not literal_step(hook,rank,range(8),names,positions)
    assert hook.multi_dp.disabled and counts["graph"]==before
    assert not literal_step(hook,rank,range(8),names,positions)
    # A writeback failure after grant is fatal on every rank; never rerun natively.
    hook2,c2=one_hook(cls,proxy,gate,rank,directory)
    hook2.request_id,hook2.next_position=names[rank],positions[rank];hook2.bridge.after=rank==5
    try:literal_step(hook2,rank,range(8),names,positions)
    except md.BrokenRound:pass
    else:raise AssertionError("postlaunch error replayed or continued")
    assert c2["graph"]==1 and hook2.invalid
    Path(directory,f"rank{rank}.json").write_text(json.dumps(counts))
    gate.local_control.close();dist.destroy_process_group()


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("prepared",type=Path,help="output of prepare.py --multi-dp")
    args.add_argument("artifacts",type=Path,help="directory for CPU-only receipts")
    options=args.parse_args()
    WORKER=options.prepared.resolve()/"vllm_ascend/worker"
    RAW=OLD=WORKER
    ROOT=options.artifacts.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    md=imported("glm53_multi_dp_cpu",WORKER/"glm53_multi_dp.py")
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        mp.start_processes(run,args=(str(Path(d,"bootstrap")),d),nprocs=8,join=True,start_method="fork")
        stats=[json.loads(Path(d,f"rank{i}.json").read_text()) for i in range(8)]
        import subprocess
        with (ROOT/"cpu-merged-receipts.log").open("w") as output:
            subprocess.run([sys.executable,str(Path(__file__).with_name("merge_audit.py")),d],stdout=output,check=True)
        with (ROOT/"cpu-trace.jsonl").open("w") as output:
            for i in range(8):output.write(Path(d,f"trace-{i}.jsonl").read_text())
    print(json.dumps({"device":"CPU/Gloo","world_size":8,"graph_calls":stats,
                      "tested":["C8 distinct positions/id/local logits","C3 mixed B1","C1 B1","prefill+native KV reimport",
                                "duplicate request","long UTF8 fallback","resumed lifetime","capacity2 actual B1 and B2 fallback","pregrant rollback","postgrant fail-stop"]}))
