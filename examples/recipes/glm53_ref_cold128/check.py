"""Build the actual cold reference recipe and replay seven CPU tests with an attested device-log fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("official_reference",type=Path)
    p.add_argument("--result",type=Path)
    args=p.parse_args()
    recipe=Path(__file__).resolve().parent
    receipt=json.loads((recipe/"fixtures/184-plan-events.json").read_text())
    fixture=recipe/"fixtures/184-plan-events.log"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==receipt["fixture_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        work=root/"source"/"candidate";work.mkdir(parents=True)
        for name in ("prepare.py","audit.py","cpu_checks.py"):
            shutil.copyfile(recipe/name,work/name)
        dest=root/receipt["source"]
        dest.parent.mkdir(parents=True);shutil.copyfile(fixture,dest)
        subprocess.run([sys.executable,"-IB",str(work/"prepare.py"),str(args.official_reference),str(work/"build")],check=True)
        subprocess.run([sys.executable,"-IB",str(work/"cpu_checks.py")],check=True)
        proof=(work/"evidence/cpu-results.json").read_bytes()
        if args.result:
            args.result.write_bytes(proof)
        print("Validated",hashlib.sha256(proof).hexdigest())


if __name__=="__main__":
    main()
