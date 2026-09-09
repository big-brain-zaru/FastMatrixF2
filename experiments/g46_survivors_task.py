"""pool_runner task: full solves of the Gamma rank-46 census survivors (outcome 'pass' in G46_filter.jsonl),
fewest slots first, E1+E2 + CaDiCaL 1.9.5 at --conf conflicts. Any SAT = verified Gamma-invariant rank-46.
  python pool_runner.py --task g46_survivors_task --workers 8 --wall 3600 --log ../results/G46_full.jsonl -- --conf 5000000 --max-slots 11
"""
import argparse, json, time
import gamma46_task as G46

def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=int, default=5_000_000)
    ap.add_argument("--max-slots", type=int, default=11)
    ap.add_argument("--filter-only", action="store_true", default=False)
    ap.add_argument("--filter-conf", type=int, default=100_000)
    return ap.parse_args(argv)

def instances(a):
    out = []
    for l in open("../results/G46_filter.jsonl"):
        r = json.loads(l)
        if r.get("outcome") != "pass" or len(r["sizes"]) > a.max_slots: continue
        out.append({"key": r["key"], "class_key": r["class_key"], "sizes": r["sizes"], "cost": -len(r["sizes"])})
    return out

def run_instance(inst, a):
    return G46.run_instance(inst, a)
