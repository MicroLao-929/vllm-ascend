"""Exercise production load_ep_hook and both gate implementations with eight CPU ranks."""
import ast
import builtins
import importlib.util
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace as NS

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
os.environ.setdefault("OMP_NUM_THREADS","1")
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import argparse
WORKER=None


def local_module(name,file):
    spec=importlib.util.spec_from_file_location(name,file)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def run(rank,path):
    dist.init_process_group("gloo",init_method="file://"+path,rank=rank,world_size=8)
    torch.set_num_threads(1)
    for name in ["vllm_ascend","vllm_ascend.worker","blockrt","blockrt.dist","blockrt.models","blockrt.models.glm_5_3_flash"]:
        m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
    ctrl=local_module("vllm_ascend.worker.glm53_dp8_control",WORKER/"glm53_dp8_control.py")
    mu=local_module("vllm_ascend.worker.glm53_multi_dp",WORKER/"glm53_multi_dp.py")
    back=types.ModuleType("blockrt.models.glm_5_3_flash.multi_dp")
    back.CAPABILITY=mu.CAPABILITY;back.source_receipts=lambda *a,**k:()
    sys.modules[back.__name__]=back
    class Companion:
        @property
        def receipt_graphs(self):return ()
        def run(self,active_owners=None):return ()
    class Runtime:
        closed=0
        def __init__(self,who,*args):
            if case=="init failure" and who==3:raise RuntimeError("injected runtime init rank 3")
        @staticmethod
        def load_bootstrap():return NS(mk_ep_init=1,mk_ep_pointer=2,mk_ep_barrier=3,mk_ep_close=4)
        def close(self):Runtime.closed+=1
    runtime_module=types.ModuleType("blockrt.dist.ep_runtime");runtime_module.ExpertParallelRuntime=Runtime
    sys.modules[runtime_module.__name__]=runtime_module
    group=NS(rank_in_group=rank,world_size=8,ranks=list(range(8)),cpu_group=dist.group.WORLD)
    envs=NS(VLLM_ASCEND_GLM53_MK_EP_ENDPOINT="cpu-test",VLLM_ASCEND_GLM53_MK_CHECKPOINT="test",
            VLLM_ASCEND_GLM53_MK_DECODE=False,VLLM_ASCEND_GLM53_MK_MTP=False,
            VLLM_ASCEND_GLM53_MK_NATIVE_EXPERTS=True,VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS=2048)
    pc=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=8,data_parallel_size_local=8,
          prefill_context_parallel_size=1,decode_context_parallel_size=1,enable_expert_parallel=True)
    runner=NS(parallel_config=pc,dp_rank=rank,max_num_reqs=2,model_config=NS(enforce_eager=True),
              scheduler_config=NS(max_num_seqs=2),speculative_config=None,broadcast_pp_output=False,
              use_aux_hidden_state_outputs=False,use_async_scheduling=False,dynamic_eplb=False,vllm_config=NS(kv_transfer_config=None))
    def construct(r,q,rt,gate):
        if rank==5 and case=="construct failure":raise ValueError("injected constructor detail rank 5")
        if rank==4 and case=="returned none":return None
        return NS(gate=gate,companion=Companion(),plan=NS(depth=45,size=8,moe=tuple(range(3,45))),
                  tp_size=1,close=lambda:rt.close())
    def check(ok,message):
        if not ok:raise RuntimeError("GLM53 MK EP: "+message)
    fnames={"_topology","_group_rows","_check_runner_config","load_ep_hook","RoundGate"}
    tree=ast.parse((WORKER/"glm53_mk_ep.py").read_text())
    body=[n for n in tree.body if getattr(n,"name",None) in fnames]
    # Only NPU synchronization is injected; startup never calls it in this CPU test.
    ns=dict(os=os,envs=envs,builtins=builtins,_require=check,torch=NS(distributed=dist,npu=NS(synchronize=lambda:None)),
            get_dp_group=lambda:group,get_ep_group=lambda:group,logger=logging.getLogger("test"),_construct=construct)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"production-startup","exec"),ns)
    load=ns["load_ep_hook"]
    for compact in (False,True):
        os.environ["GLM53_PLAIN_COMPACT_GATE"]=str(int(compact))
        os.environ["GLM53_DP8_LOCAL_CONTROL"]="0"
        for case,needle in [
                ("disabled",None),("strict capacity two","synchronous eager single-request"),
                ("rank gate mismatch","inconsistent worker configuration"),("bootstrap failure","synchronous eager"),
                ("init failure","injected runtime init rank 3"),("construct failure","constructor detail rank 5"),
                ("returned none","returned no GLM53 EP hook"),("capacity two multi ready",None)]:
            envs.VLLM_ASCEND_GLM53_MK_EP=case!="disabled"
            os.environ["MK_GLM53_MULTI_DP"]=str(int(case not in ("disabled","strict capacity two") and not (case=="rank gate mismatch" and rank==4)))
            runner.model_config.enforce_eager=not(case=="bootstrap failure" and rank==2)
            before=Runtime.closed
            try:
                h=load(runner)
                assert needle is None
                if case=="disabled":assert h is None
                else:assert h is not None and isinstance(h.multi_dp,mu.Driver)
            except RuntimeError as exc:
                if not needle or needle not in str(exc) or "NoneType" in str(exc):raise
                if case in ("construct failure","returned none"):assert Runtime.closed==before+1
        dist.barrier()
    dist.destroy_process_group()


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("prepared",type=Path,help="output of prepare.py --multi-dp")
    WORKER=args.parse_args().prepared.resolve()/"vllm_ascend/worker"
    with tempfile.TemporaryDirectory() as temp:
        mp.start_processes(run,args=(str(Path(temp,"rendezvous")),),nprocs=8,join=True,start_method="fork")
    print("PASS: 8 ranks x Gloo and compact gates x disabled, legacy shape, mode mismatch, single-rank bootstrap/init/construct/None, multi capacity two.")
