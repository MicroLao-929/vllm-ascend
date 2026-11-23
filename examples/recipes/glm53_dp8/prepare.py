"""Prepare an isolated vLLM Ascend v0.30 worker package for GLM-5.3 DP8."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


RECIPE = Path(__file__).resolve().parent
PIN = "cd87230b21d74bbc1958dacd57d3b496da6f8919"


def install_lazy_tables(dest):
    candidate = RECIPE / "lazy_tables"
    versions = json.loads((candidate / "versions.json").read_text())
    for name, options in versions.items():
        worker = dest / "worker" / name
        original = hashlib.sha256(worker.read_bytes()).hexdigest()
        if original not in options:
            raise ValueError(f"Unsupported worker for lazy tables: {name} at {original}")
        version = options[original]
        with (candidate / version["file"]).open("rb") as delta:
            subprocess.run(["patch", "--quiet", "--batch", "--fuzz=0", str(worker)], stdin=delta, check=True)
        if hashlib.sha256(worker.read_bytes()).hexdigest() != version["result"]:
            raise ValueError(f"Unexpected lazy tables worker: {name}")
    for path in sorted((candidate / "worker").glob("*.py")):
        output = dest / "worker" / path.name
        if output.exists():
            raise FileExistsError(f"Unexpected existing worker: {path.name}")
        shutil.copy2(path, output)


def prepare(source, output, *, lazy_tables=False):
    """Publish a complete package atomically after checking the pinned inputs."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    package = source / "vllm_ascend"
    if package in output.parents or output == package:
        raise ValueError("Output must be outside the source package")
    inputs = json.loads((RECIPE / "base.json").read_text())
    for name, digest in inputs.items():
        path = package / "worker" / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != digest:
            raise ValueError(f"Incompatible worker {name}. Expected source commit {PIN}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        stage = Path(scratch) / "package"
        dest = stage / "vllm_ascend"
        shutil.copytree(package, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in sorted((RECIPE / "worker").glob("*.py")):
            if path.name not in inputs:
                raise ValueError(f"Worker missing from base manifest: {path.name}")
            shutil.copy2(path, dest / "worker" / path.name)
        if lazy_tables:
            install_lazy_tables(dest)
        os.rename(stage, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="pinned checkout or prepared v0.30 runtime")
    parser.add_argument("output", type=Path, help="a new directory for PYTHONPATH")
    parser.add_argument("--lazy-tables", action="store_true", help="install the opt-in early-runner experiment")
    args = parser.parse_args()
    print(prepare(args.source, args.output, lazy_tables=args.lazy_tables))
