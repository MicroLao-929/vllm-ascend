"""Run the actual reference admission, request lifecycle and auditor on CPU. No NPU imports."""
import ast
from collections import Counter
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest import TestCase, defaultTestLoader, TextTestRunner
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
BUILD=HERE/"build"
REF=Path((BUILD/"original-root").read_text().strip())
EP=REF/"decode_255k/overlay/glm53_mk_ep.py"
NEW=BUILD/"overlay/glm53_mk_ep.py"
BASE=EP.with_name("glm53_mk_hook.py")
WARM=EP.with_name("glm53_mk_warm_prefix.py")


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def methods(path,name,names,bases,namespace):
    node=next(x for x in ast.parse(path.read_text()).body if getattr(x,"name",None)==name)
    node.bases=[ast.Name(x,ast.Load()) for x in bases]
    node.body=[x for x in node.body if getattr(x,"name",None) in names]
    assert len(node.body)==len(names)
    prog=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(prog,str(path),'exec'),namespace)
    return namespace[name]


def require(x,msg):
    if not x:raise ValueError(msg)


def ns():
    quiet=logging.getLogger("ref-cold-cpu")
    return {"_require":require,"os":os,"_LOG":quiet,"logger":quiet,"Path":Path,
            "ctypes":ctypes,"hashlib":hashlib,"BLOCK_SIZE":640,"_Step":lambda *args:args}


def request_objects(rid="cmpl-cpu-0-engine",prompt=128,max_tokens=128):
    req=NS(mm_features=None,prompt_embeds=None,lora_request=None,
           sampling_params=NS(prompt_logprobs=None,max_tokens=max_tokens),num_computed_tokens=0)
    batch=NS(num_reqs=1,req_ids=[rid],num_computed_tokens_cpu=[0],num_prompt_tokens=[prompt])
    r=NS(requests={rid:req},input_batch=batch)
    new=NS(req_id=rid,num_computed_tokens=0,prompt_token_ids=[9]*prompt)
    out=NS(num_scheduled_tokens={rid:prompt},total_num_scheduled_tokens=prompt,scheduled_new_reqs=[new],
           scheduled_cached_reqs=NS(req_ids=[],resumed_req_ids=set()), scheduled_spec_decode_tokens={},
           kv_connector_metadata=None,ec_connector_metadata=None,has_sync_kv_loads=False,scheduled_encoder_inputs={},
           preempted_req_ids=set(),finished_req_ids=set())
    return rid,r,out


def hookref(frozen,warm=False,cold=False):
    side=ns()
    methods(BASE,"GLM53MKHook",{"_schedule","observe_native","_release","try_decode","update_requests"},[],side)
    Warm=methods(WARM,"WarmPrefixAttach",{"clear","observe","_local_schedule"},[],side)
    Hook=methods(frozen,"GLM53MKEPHook",{"observe_native","_release","_vote","_offline_schedule_signature","_check_ready"},
                 ["GLM53MKHook"],side)
    hook=Hook()
    hook._reference_cold_short=cold
    hook._warm_prefix=Warm() if warm else None
    if warm:hook._warm_prefix.clear()
    hook.request_id=hook.next_position=hook.active=None; hook.invalid=False
    hook.gate=NS(rank=0);hook.bridge=NS(max_context_tokens=262080,release=lambda rid:None)
    return hook


def branch_guards(path,runner,replace=None):
    tree=ast.parse(path.read_text())
    fn=next(x for x in tree.body if getattr(x,'name',None)=='load_ep_hook')
    fields={"cold_config","combine"}
    env=dict(os=os,runner=runner,_require=require,enabled=True,topology=(8,1,1,1,1),shared=True,capacity="262080",
             single_origin=False,replicated_owner=True,fuse_tp_admission=True,batch_prefix_copies=True,
             warm_prefix=False,warm_address_mode="off",deferral_mode="off",long_li="1",
             combined_li="0",typed_owner="1",private_input="1",cold_short=True)
    env.update(replace or {})
    count=0
    for stmt in fn.body:
        if isinstance(stmt,ast.Assign) and getattr(stmt.targets[0],"id",None) in fields:
            exec(compile(ast.Module([stmt],[]),str(path),"exec"),env)
        if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and getattr(stmt.value.func,'id',None)=='_require':
            msg=stmt.value.args[-1]
            if isinstance(msg,ast.Constant) and msg.value in {
                "reference cold-short requires 45L TP8/DP1/EP8, full native metadata, fixed capacity and typed owner",
                "typed owner requires the cumulative replicated-owner TP8 service",
                "combined warm-align and LI require TP8, long capacity and batched prefix import",
                "warm-align CPU addresses require warm-prefix TP admission, replicated owner and a reviewed LI route",
                "typed owner and private broadcast must be selected together"}:
                exec(compile(ast.Module([stmt],[]),str(path),"exec"),env)
                count+=1
    return count


class Scalar:
    shape=(1,)
    def __init__(self,p):self.p=p
    def __eq__(self,n):return NS(all=lambda:NS(item=lambda:self.p==n))


class Checks(TestCase):
    def test_materialization_and_unmodified_overlay(self):
        prep=module("coldprep",HERE/"prepare.py")
        manifest=json.loads((BUILD/"manifest.json").read_text())
        self.assertEqual({n:prep.digest(REF/n) for n in prep.SOURCES},prep.SOURCES)
        changed=[k for k,v in manifest["original_overlay"].items() if v!=manifest["adapted_overlay"][k]]
        self.assertEqual(changed,["glm53_mk_ep.py"])
        with tempfile.TemporaryDirectory() as temp:
            tmp=Path(temp)/"rebuild"
            result=prep.prepare(REF,tmp)
            self.assertEqual(prep.digest(BUILD/"manifest.json"),result["manifest"])
            self.assertEqual((BUILD/"serve-mk.sh").read_bytes(),(tmp/"serve-mk.sh").read_bytes())
        for f in BUILD.glob("*.sh"):subprocess.run(["bash","-n",str(f)],check=True)
        subprocess.run(["sha256sum","--check","--strict","overlay.sha256"],cwd=BUILD,check=True,stdout=subprocess.DEVNULL)
        orig=(REF/"decode_255k/serve.sh").read_text()
        cmd=(BUILD/"serve-mk.sh").read_text()
        self.assertEqual(orig[orig.index("argv=("):].replace("--enable-prefix-caching","--no-enable-prefix-caching"),cmd[cmd.index("argv=("):])
        self.assertIn('sha256sum --check --strict launch-shared.sha256 launch-git.sha256',cmd)
        self.assertIn("verify_registry.py",cmd)
        self.assertIn('export MK_GLM53_REFERENCE_COLD_SHORT=1 MK_GLM53_LAZY_RUNNER_INPUTS=off',cmd)

    def test_real_warm_attach_rejects_cold(self):
        rid,r,o=request_objects()
        h=hookref(EP,warm=True);ctx=NS(cudagraph_runtime_mode=NS(name="NONE"))
        with self.assertLogs("ref-cold-cpu",level="INFO") as capture:
            h.observe_native(r,o)
        self.assertTrue(any('no matching block-aligned local prefix hit' in line for line in capture.output))
        self.assertIsNone(h.next_position)
        for n in range(128,255):
            r.input_batch.num_computed_tokens_cpu=[n]
            o.num_scheduled_tokens={rid:1};o.total_num_scheduled_tokens=1;o.scheduled_new_reqs=[]
            self.assertEqual(h._vote(r,o,False,ctx,1),("native",None,None))
            h.observe_native(r,o)
        self.assertIsNone(h.next_position)

    def test_actual_cold_vote_and_full_native_metadata_handoff(self):
        for rank in range(8):
            rid,r,o=request_objects()
            h=hookref(NEW,cold=True);h.gate.rank=rank;h.tp_size=8
            h.text=NS(config=NS(hidden_size=4096,vocab_size=100))
            fakehidden=NS(shape=(1,4096));calls=[]
            h._cache_descriptors=lambda runner:("K","M","I")
            def bridge_step(ctx,**kw):
                self.assertEqual((kw["kda"],kw["mla"],kw["indexer"]),("K","M","I"))
                self.assertIsNone(kw["cpu_addresses"]);self.assertIsNone(kw["cpu_reads"])
                self.assertNotIn("warm_prefix_destinations",kw)
                calls.append((kw['request_id'],kw['position']))
                return NS(hidden_states=fakehidden,logits=None,graph_reused=bool(len(calls)>1),imported=len(calls)==1)
            h.bridge.step=bridge_step
            ctx=NS(cudagraph_runtime_mode=NS(name="NONE"),attn_metadata={"explicit-native":"descriptor"})
            with self.assertLogs("ref-cold-cpu",level="INFO") as log:h.observe_native(r,o)
            self.assertIn(f"cold prefill rank={rank}",log.output[-1]);self.assertEqual(h.next_position,128)
            for pos in range(128,255):
                r.input_batch.num_computed_tokens_cpu=[pos]
                o.scheduled_new_reqs=[];o.num_scheduled_tokens={rid:1};o.total_num_scheduled_tokens=1
                self.assertEqual(h._vote(r,o,False,ctx,1),("real",rid,pos))
                self.assertIs(h.try_decode(r,o,ctx,Scalar(pos),Scalar(200))[2],fakehidden)
            self.assertEqual(calls,[(rid,p) for p in range(128,255)])
            o.finished_req_ids={rid};h.update_requests(o);self.assertIsNone(h.next_position)

    def test_wrong_request_shapes_never_arm(self):
        for prompt,max_tokens,start,chunk in [(129,128,0,129),(128,127,0,128),(128,128,1,127),(128,128,0,64)]:
            rid,r,o=request_objects(prompt=prompt,max_tokens=max_tokens)
            r.input_batch.num_computed_tokens_cpu=[start]
            o.scheduled_new_reqs[0].num_computed_tokens=start
            o.total_num_scheduled_tokens=chunk;o.num_scheduled_tokens={rid:chunk}
            h=hookref(NEW,cold=True);h.observe_native(r,o);self.assertIsNone(h.next_position)
        rid,r,o=request_objects()
        o.scheduled_new_reqs[0].req_id="other";h=hookref(NEW,cold=True);h.observe_native(r,o)
        self.assertIsNone(h.next_position)

    def test_boot_guards_voted_and_fail_closed(self):
        runner=NS(cache_config=NS(enable_prefix_caching=False,mamba_cache_mode="none"),
                  model_config=NS(enforce_eager=True),speculative_config=None,scheduler_config=NS(max_num_seqs=1),
                  model=NS(config=NS(num_hidden_layers=45)))
        env={"MK_GLM53_LAZY_RUNNER_INPUTS":"off","MK_GLM53_CPU_METADATA_PREFETCH":"0",
             "MK_GLM53_MHC_FP32_SPLITK":"1","MK_GLM53_SFA_POOLED_B1":"1","SCHEDULING_MODE":"static"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError,"cumulative replicated-owner"):branch_guards(EP,runner)
            self.assertEqual(branch_guards(NEW,runner),5)
            for field,v in [("topology",(1,1,8,1,1)),("capacity","256"),("private_input","0"),("typed_owner","0"),
                            ("warm_prefix",True),("warm_address_mode","on"),("deferral_mode","on"),("combined_li","1"),
                            ("long_li","0"),("batch_prefix_copies",False),("fuse_tp_admission",False),
                            ("replicated_owner",False),("single_origin",True),("shared",False)]:
                with self.subTest(field=field),self.assertRaises(ValueError):branch_guards(NEW,runner,{field:v})
            warm=dict(cold_short=False,warm_prefix=True,warm_address_mode="on",combined_li="1")
            self.assertEqual(branch_guards(EP,runner,warm),4);self.assertEqual(branch_guards(NEW,runner,warm),5)
            for obj,key,v in [(runner.cache_config,"enable_prefix_caching",True),(runner.cache_config,"mamba_cache_mode","align"),
                              (runner.scheduler_config,"max_num_seqs",2),(runner.model.config,"num_hidden_layers",4),
                              (runner,"speculative_config",{}),(runner.model_config,"enforce_eager",False)]:
                with patch.object(obj,key,v), self.assertRaises(ValueError):branch_guards(NEW,runner)
            for var,v in [("MK_GLM53_LAZY_RUNNER_INPUTS","on"),("SCHEDULING_MODE","dynamic"),("MK_GLM53_SFA_POOLED_B1","0")]:
                with patch.dict(os.environ,{var:v}), self.assertRaises(ValueError):branch_guards(NEW,runner)
        self.assertRegex(NEW.read_text(),r"private_input,\s+cold_short, cold_config\)\)")
        self.assertIn("all(v == voted[0] for v in voted)",NEW.read_text())

    def test_real_offline_flag_check_and_rank_consensus(self):
        h=hookref(NEW,cold=True)
        p=REF/"decode_255k/offline-plan.txt";checksum=hashlib.sha256(p.read_bytes()).hexdigest()
        h.gate=NS(rank=3,gather=lambda key,rec:[rec]*8)
        h.verified_graph=NS(nodes=[1,2,3])
        setting={"MEGA_KERNEL_HOME":"/unused","MK_GLM53_OFFLINE_STATIC_COSTS":str(p)}
        for hasfile,applied,ok in ((True,1,True),(True,0,False),(False,1,False),(False,0,True)):
            env=dict(setting) if hasfile else {"MEGA_KERNEL_HOME":"/unused"}
            fn=lambda:applied
            with patch.dict(os.environ,env,clear=True), patch.object(ctypes,"CDLL",return_value=NS(Glm53OfflineScheduleApplied=fn)):
                active=NS(admission_token=None)
                if ok:
                    self.assertEqual(h._offline_schedule_signature(),checksum if hasfile else "stock")
                    h._check_ready(active)
                else:
                    with self.assertRaisesRegex(ValueError,"capture disagrees"):h._check_ready(active)
        h.gate.gather=lambda key,rec:[rec]*7+[("wrong",None)]
        h._offline_schedule_signature=lambda:checksum
        with self.assertRaisesRegex(ValueError,"capture disagrees"):h._check_ready(NS(admission_token=None))

    def test_proof_requires_true_steps_prefill_and_first_import(self):
        audit=module("colda",BUILD/"audit.py");root,mod=audit.original();r={"seed":12810,"id":"cmpl-test","cached_tokens":0}
        rid="cmpl-test-0-eng"
        lines=[f"GLM53 EP round={p+9} rank={k} origin_rank=0 request={rid} position={p} role=real origin_tokens=1 backend=mk cache_writeback=complete layer_receive=None tp_size=8"
               for p in range(128,255) for k in range(8)]
        pre=[(k,rid,0,128,128) for k in range(8)]
        imp=[(rid,p,p==128) for p in range(128,255) for _ in range(8)]
        text="\n".join(lines)
        check=lambda t,p,i: audit.per_request(t,r,mod,p,i)["ok"]
        self.assertTrue(check(text,pre,imp))
        for variant in [(text,pre[:-1],imp),(text,pre,imp[:-1]),("\n".join(lines[:-1]),pre,imp),
                        (text+"\n"+lines[0],pre,imp),(text,pre,[*imp[:-1],(rid,254,True)])]:
            self.assertFalse(check(*variant))
        r["cached_tokens"]=64;self.assertFalse(check(text,pre,imp))
        typed=[f"GLM53 typed owner capture rank={k} layers=42 compact_endpoints=42" for k in range(8)]
        for which,signature in (("required",audit.OFFLINE_SHA),("stock","stock")):
            logs=[*typed,*[f"GLM53 reference cold ready rank={k} schedule={signature} python_ops=2100" for k in range(8)]]
            if which=="required":logs.extend([audit.PLAN_OK]*8)
            valid="\n".join(logs)
            self.assertTrue(audit.schedule(valid,which)["ok"])
            self.assertFalse(audit.schedule(valid,"stock" if which=="required" else "required")["ok"])
            self.assertFalse(audit.schedule(valid.replace("cold ready rank=7","cold ready rank=5"),which)["ok"])
            self.assertFalse(audit.schedule(valid.replace("layers=42","layers=41"),which)["ok"])
            self.assertFalse(audit.schedule(valid+"\nconfigured offline schedule differs from the built graph",which)["ok"])
        # The separately retained original long board confirms these three marker grammars.
        old=(HERE.parents[1]/"bench/reference255-original-h184-110006-183903/board/mk-final/service.log")
        if old.is_file():
            s=mod.ESC.sub("",old.read_text(errors="replace"))
            self.assertEqual(len(audit.TYPED.findall(s)),8);self.assertEqual(s.count(audit.PLAN_OK),8)
            self.assertGreater(len(audit.IMPORT.findall(s)),8)
        historical=HERE.parents[1]/"bench"
        self.assertTrue(historical.is_dir())


if __name__=="__main__":
    out=HERE/"evidence";out.mkdir(exist_ok=True)
    suite=defaultTestLoader.loadTestsFromTestCase(Checks)
    result=TextTestRunner(verbosity=2).run(suite)
    report={"passed":result.wasSuccessful(),"cases":result.testsRun,
            "production_functions":["GLM53MKEPHook.observe_native","WarmPrefixAttach.observe","GLM53MKHook._schedule",
            "GLM53MKEPHook._vote","GLM53MKHook.try_decode","GLM53MKHook.update_requests","load_ep_hook admission expressions",
            "_offline_schedule_signature","_check_ready"],"operator_execution":"none; synthetic scalar and bridge return",
            "standard_output_comparator":"true MK logs plus rank-specific plan, prefill and cold import; 7112 required",
            "manifest":hashlib.sha256((BUILD/"manifest.json").read_bytes()).hexdigest()}
    dest=out/"cpu-results.json"
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({**report,"evidence_sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}))
    sys.exit(0 if result.wasSuccessful() else 1)
