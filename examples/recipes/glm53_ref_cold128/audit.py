"""Require native cold prefill, first import, 127 true TP8 steps and the selected graph schedule."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from statistics import median
import sys

HERE=Path(__file__).resolve().parent
OFFLINE_SHA="5ee116026e720cdd348a5ed164713dfcbd6edfdf3d69b297c3bb55d6dbea03a5"
PREFILL=re.compile(r"GLM53 reference cold prefill rank=(\d+) request=(\S+) start=(\d+) end=(\d+) prompt=(\d+)")
READY=re.compile(r"GLM53 reference cold ready rank=(\d+) schedule=(\S+) python_ops=(\d+)")
TYPED=re.compile(r"GLM53 typed owner capture rank=(\d+) layers=(\d+) compact_endpoints=(\d+)")
IMPORT=re.compile(r"GLM53 MK decode request=(\S+) position=(\d+) graph=(?:capture|reuse) imported=(True|False) cache_writeback=complete")
PLAN_OK="offline schedule enabled graph_ops=2122 nodes=20806 tasks=20231"
PLAN_REJECT=("configured offline schedule differs from the built graph","GLM53 B16 offline schedule: ")
RANKS=set(range(8))
REAL="all_decode_positions_eight_real_mk_ranks"


def need(condition,msg):
    if not condition:
        raise ValueError(msg)


def save(p,data):
    temp=p.with_suffix(".pending")
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    temp.replace(p)


def original():
    root=Path((HERE/"original-root").read_text().strip())
    spec=importlib.util.spec_from_file_location("original_short128",(root/"short128/study.py"))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return root,mod


def schedule(text,plan):
    rows=[(int(r),v,int(n)) for r,v,n in READY.findall(text)]
    typed=[(int(r),int(m),int(n)) for r,m,n in TYPED.findall(text)]
    seen={r for r,_,_ in rows}
    expected=OFFLINE_SHA if plan=="required" else "stock"
    markers=text.count(PLAN_OK)
    failures=[msg for msg in PLAN_REJECT if msg in text]
    ok=(seen==RANKS and all(v==expected and n>0 for _,v,n in rows)
        and len({n for _,_,n in rows})==1 and {r for r,_,_ in typed}==RANKS
        and all(m==42 and n==42 for _,m,n in typed)
        and (markers>=8 if plan=="required" else markers==0) and not failures)
    if ok:
        state="original_plan_confirmed" if plan=="required" else "explicit_stock_no_offline_plan"
    elif failures:
        state="original_plan_rejected" if plan=="required" else "unexpected_plan_in_stock_arm"
    else:
        state="insufficient_or_inconsistent_plan_evidence"
    return {"mode":plan,"state":state,"ok":ok,"expected_schedule":expected,"rank_ready":rows,
            "typed_owner":typed,"exact_native_plan_markers":markers,"explicit_errors":failures}


def per_request(text,r,mod,prefill,imports):
    rid=r["id"]
    routing=mod.classify(text,rid)
    pr=[(rk,s,e,p) for rk,request,s,e,p in prefill if mod.match(rid,request)]
    ir=Counter((p,imp) for request,p,imp in imports if mod.match(rid,request))
    prefill_ok=Counter(pr)==Counter((rank,0,128,128) for rank in RANKS)
    imports_ok=ir==Counter({(pos,pos==128):8 for pos in range(128,255)})
    cached_ok=r.get("cached_tokens") in (None,0)
    ok=(routing["status"]==REAL and routing["real_mk_rank_events"]==1016 and prefill_ok and imports_ok and cached_ok)
    return {"seed":r["seed"],"id":rid,"ok":ok,"rank_prefills":pr,"native_prefill_0_to_128":prefill_ok,
            "first_import_then_reuse":imports_ok,"first_imports":ir[128,True],"unexpected_imports":sum(
                n for (p,flag),n in ir.items() if flag!=(p==128)),"cached_tokens":r.get("cached_tokens"),"routing":routing}


def inputs(data,root):
    measured=[x for x in data if x["phase"]=="measure"]
    warm=[x for x in data if x["phase"]=="warm"]
    need(len(warm)==1 and warm[0]["seed"]==12800 and len(data)==8,"expected one warm and seven measured requests")
    need([x["seed"] for x in measured]==list(range(12810,12871,10)),"expected the frozen seven measured seeds")
    hashes=json.loads((root/"short128/input-seed-digests.json").read_text())["seed_and_realized_sha256"]
    for r in data:
        digest=hashlib.sha256(json.dumps(r["input_ids"],separators=(",",":")).encode()).hexdigest()
        need(hashes[str(r["seed"])]==digest and len(r["input_ids"])==len(r["ids"])==len(r['arrivals_ns'])==128,
             "input hash or response length mismatch")
        need(all(a<=b for a,b in zip(r['arrivals_ns'],r['arrivals_ns'][1:])),"non-monotonic response")
        need(r.get("cached_tokens") in (0,None),"unexpected public cached tokens")
    return measured


def audit(mode,arm,plan):
    root,mod=original()
    log=arm/"service.log"
    check={"mode":mode,"requested_plan":plan,"passed":False,"failures":[]}
    text=mod.ESC.sub("",log.read_text(errors="replace")) if log.is_file() else ""
    if mode=="mk":check["schedule"]=schedule(text,plan)
    try:
        command=(arm/"command.sh").read_text()
        need("--no-enable-prefix-caching" in command and "--enable-prefix-caching" not in command,
             "actual command did not disable prefix caching")
        cov=json.loads((arm/"short-coverage.json").read_text())
        check["host"]=cov["host"]
        need(cov["mode"]==mode and check["host"],"unexpected worker mode or machine")
        raw=[json.loads(line) for line in (arm/"requests.jsonl").read_text().splitlines() if line]
        measured=inputs(raw,root)
        need((arm/"controller.exit").read_text().strip()=="0","controller did not complete")
        if mode=="native":
            check["native_graph_replay_markers"]=text.lower().count("acl graph replay is active")
            need("FULL_DECODE_ONLY" in command and check["native_graph_replay_markers"]>0,"missing native graph replay")
            need(not mod.official.ROUND.findall(text),"native service emitted a true MK decode")
        else:
            env=dict(line.split('=',1) for line in (arm/"cold-env.txt").read_text().splitlines())
            need(env["GLM53_REF_COLD_PLAN"]==plan and env["MK_GLM53_REFERENCE_COLD_SHORT"]=="1"
                 and env["MK_GLM53_LAZY_RUNNER_INPUTS"]=="off","unexpected cold flags")
            path=env['MK_GLM53_OFFLINE_STATIC_COSTS']
            need(path==str(root/"decode_255k/offline-plan.txt") if plan=="required" else path=="",
                 "unexpected offline plan input")
            prefill=[(int(r),req,int(s),int(e),int(p)) for r,req,s,e,p in PREFILL.findall(text)]
            imports=[(req,int(p),flag=="True") for req,p,flag in IMPORT.findall(text)]
            check["requests"]=[per_request(text,r,mod,prefill,imports) for r in measured]
            check["rank_events"]=sum(req["routing"]["real_mk_rank_events"] for req in check["requests"])
            need(all(req["ok"] for req in check["requests"]) and check["rank_events"]==7112,
                 "incomplete cold prefill, true MK trajectory or first-KV-import evidence")
            need(check["schedule"]["ok"],"offline plan route unsupported or incompletely evidenced")
        check["passed"]=True
    except (ValueError,KeyError,OSError,AssertionError) as e:
        check["failures"].append(f"{type(e).__name__}: {e}")
    return check


def compare(native,mk):
    root,mod=original()
    rec=json.loads((mk/"cold-audit.json").read_text())
    plan=rec['requested_plan']
    left=audit("native",native,plan)
    right=audit("mk",mk,plan)
    need(left["passed"] and right["passed"],f"unverified arms; see {native} and {mk} cold-audit.json")
    need(left["host"]==right["host"],"comparison requires the same host")
    prev=mod.compare(native,mk)
    need(prev['all_reference_decode_mk'] and prev['observed_real_mk_rank_steps']==7112,"incomplete reference coverage")
    # Keep measured distributions so the small fixed sample remains inspectable.
    sets=[]
    for arm in (native,mk):
        raw=[json.loads(l) for l in (arm/'requests.jsonl').read_text().splitlines() if l]
        rs=inputs(raw,root)
        sets.append({"tpot_ms":[x['tpot_ms'] for x in rs],"ttft_ms":[x['ttft_ms'] for x in rs]})
    return {"qualification":right["schedule"]["state"],"variant":"reference-plus-cold-host-adaptation",
            "host":left["host"],"tp":8,"dp":1,"ep":8,"mtp":False,"input_output":[128,128],
            "native_graph_tpot_ms":prev["native_graph_tpot_ms"],"cold_mk_tpot_ms":prev["mk_tpot_ms"],
            "speedup_native_over_mk":prev["mk_speedup_vs_native"],"matched_token_positions":prev["overall_precision_positions"],
            "total_positions":896,"first_divergence_one_based":prev["first_divergence_one_based"],
            "real_mk_rank_steps":7112,"native":sets[0],"mk":sets[1],"native_audit":str(native/'cold-audit.json'),
            "mk_audit":str(mk/'cold-audit.json')}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="act",required=True)
    a=s.add_parser("arm");a.add_argument("mode", choices=("native","mk"));a.add_argument("directory",type=Path);a.add_argument("plan",choices=("required","stock"))
    c=s.add_parser("compare");c.add_argument("native",type=Path);c.add_argument("mk",type=Path);c.add_argument("output",type=Path)
    v=p.parse_args()
    if v.act=="arm":
        result=audit(v.mode,v.directory,v.plan)
        save(v.directory/"cold-audit.json",result)
        print(json.dumps({k:result[k] for k in ("mode","requested_plan","passed","failures")}))
        sys.exit(0 if result["passed"] else 3)
    else:
        save(v.output,compare(v.native,v.mk))
