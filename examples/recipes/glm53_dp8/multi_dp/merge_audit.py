"""Validate the shape of per-worker JSONL and emit the backend receipt checker's log format."""
import argparse
import json
from pathlib import Path


def merge(folder):
    files=sorted(folder.glob("multi-dp-*-rank*.jsonl"))
    if len(files)!=8:
        raise ValueError(f"expected eight worker logs for one service instance, found {len(files)}")
    ranks=set()
    for path in files:
        rank=int(path.stem.rsplit("rank",1)[1])
        if rank in ranks or not 0<=rank<8:
            raise ValueError(f"unexpected rank file {path}")
        ranks.add(rank)
        previous=-1
        with path.open() as stream:
            for line in stream:
                item=json.loads(line)
                if item["rank"]!=rank or item["epoch"]<=previous:
                    raise ValueError(f"wrong/duplicate rank or epoch in {path}")
                previous=item["epoch"]
                print("GLM53 multi DP step "+json.dumps(item,separators=(",",":")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    merge(p.parse_args().directory)
