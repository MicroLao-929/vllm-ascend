"""CPU checks for the shipped multi-origin protocol and pinned worker patches."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

RECIPE=Path(__file__).resolve().parents[3]/"examples/recipes/glm53_dp8"
FEATURE=RECIPE/"multi_dp"


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


class Exchange:
    def __init__(self,rows):
        self.rank,self.size,self.sequence,self.rows=0,8,0,rows
    def gather(self,phase,item):
        return self.rows


class MultiSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=module("tested_glm53_multi_dp",FEATURE/"glm53_multi_dp.py")
        cls.prepare=module("tested_glm53_multi_prepare",RECIPE/"prepare.py")

    def test_distinct_real_rows_and_idle_ranks(self):
        m=self.m
        rows=[(m._REAL,"r"+str(i),128+i) if i%2==0 else (m._IDLE,None,None) for i in range(8)]
        gate=m.Group(Exchange(rows))
        plan=gate.select(rows[0])
        self.assertEqual(plan.owners,(0,2,4,6))
        self.assertEqual(tuple(s.position for s in plan.sources),(128,130,132,134))
        rows[1]=(m._REAL,"r0",150)
        self.assertFalse(gate.select(rows[0]).sources)

    def test_rejected_rank_and_migration(self):
        m=self.m
        rows=[(m._REAL,"request"+str(i),130) for i in range(8)]
        gate=m.Group(Exchange(rows));rows[4]=(m._NATIVE,None,None)
        self.assertFalse(gate.select(rows[0]).sources)
        rows[4]=(m._MIGRATION,None,None)
        self.assertFalse(gate.select(rows[0]).rearm)
        rows[4]=(m._POISONED,None,None)
        with self.assertRaises(m.BrokenRound):gate.select(rows[0])

    def test_abi_installation_manifest(self):
        m=json.loads((FEATURE/"manifest.json").read_text())
        self.assertEqual(m["capability"],self.m.CAPABILITY)
        self.assertEqual(set(m["runners"].values()),{"stock","deployed"})
        self.assertEqual(hashlib.sha256((FEATURE/"glm53_multi_dp.py").read_bytes()).hexdigest(),m["module"])
        with tempfile.TemporaryDirectory() as d:
            for name,item in m["workers"].items():
                before=(RECIPE/"worker"/name).read_bytes()
                self.assertEqual(hashlib.sha256(before).hexdigest(),item["input"])
                target=Path(d,name);target.write_bytes(before)
                with (FEATURE/item["patch"]).open("rb") as patch:
                    subprocess.run(["patch","--quiet","--batch","--fuzz=0",str(target)],stdin=patch,check=True)
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),item["output"])

    def test_unsupported_runner_and_exclusive_epoch_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            t=Path(d);(t/"worker").mkdir();(t/"worker/model_runner_v1.py").write_text("# unrelated")
            with self.assertRaisesRegex(ValueError,"Unsupported v0.30"):self.prepare.install_multi_dp(t)
            for opts in ({"lazy_tables":True},{"local_dp_sync":True}):
                with self.assertRaisesRegex(ValueError,"Multi-source decode owns"):
                    self.prepare.prepare(t,t/"output",multi_dp=True,**opts)
                self.assertFalse((t/"output").exists())


if __name__=="__main__":unittest.main(verbosity=2)
