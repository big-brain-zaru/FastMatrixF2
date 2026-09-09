"""Reproduce every headline count of docs/SCIENTIFIC-REPORT.md from results/ and print a checklist.
Usage: python reproduce.py   (runs report_figures.py, dissect_b2.py and the checks below)"""
import json, glob, subprocess, sys, collections
import symgroup as SG, fp_filter as FP
from flip_graph import verify

def jl(p):
    out = []
    for f in sorted(glob.glob(p)):
        for l in open(f):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def key(r): return (tuple(r["deleted_orbits"]), tuple(r["refill_sizes"]), tuple(r["refill_subgroup_orders"]))

print("== regenerating figures and B2 digest")
subprocess.run([sys.executable, "report_figures.py"], stdout=subprocess.DEVNULL, check=True)
subprocess.run([sys.executable, "dissect_b2.py"], stdout=subprocess.DEVNULL, check=True)
checks = []
def chk(name, value, expect=None):
    ok = (expect is None) or (value == expect)
    checks.append((name, value, expect, ok)); print("  %-70s %s%s" % (name, value, "" if expect is None else ("  [expected %s] %s" % (expect, "OK" if ok else "MISMATCH"))))

print("== verified schemes")
S47, G = FP.load_gamma(4)
chk("4x4 rank-47 verifies over F2", verify(S47, 4, 4, 4), True)
chk("47 invariant under its order-12 group", SG.is_invariant(S47, G, 4) and len(G) == 12, True)
chk("47 orbit sizes", sorted([len(o) for _, o, _ in SG.orbits(S47, G, 4)], reverse=True), [12, 6, 6, 6, 6, 3, 3, 3, 1, 1])
for f, n in [("../results/records/lille_555_rank93.json", 5), ("../results/records/lille_666_rank153.json", 6)]:
    S = [tuple(t) for t in json.load(open(f))["scheme_bitmasks"]]
    chk("%s verifies, rank %d, C3-invariant" % (f.split("/")[-1], len(S)), verify(S, n, n, n) and SG.is_invariant(S, SG.closure([SG.perm_cyc(n)], n), n), True)
new93 = glob.glob("../results/orbitn_5_same_rank93_*.json")
chk("new 5x5 rank-93 schemes, all verified and pairwise distinct", (len(new93), all(verify([tuple(t) for t in json.load(open(f))["scheme_bitmasks"]], 5, 5, 5) for f in new93),
     len({frozenset(map(tuple, json.load(open(f))["scheme_bitmasks"])) for f in new93})), (15, True, 7))   # 7 since the 5 Sep 21:43 overwrite of the D5-26 files (DAY4-LOG section 14.3); 6 at the Phase-0 check

print("== Theorem 1 (|D|<=2 reduce): distinct instances all UNSAT")
t1 = [r for r in jl("../results/orbit_lns_reduce_w*.jsonl") + jl("../results/orbit_lns_reduceB_w*.jsonl") + jl("../results/orbit_lns_reduceC_w*.jsonl") if len(r["deleted_orbits"]) <= 2]
by = {}
for r in t1: by.setdefault(key(r), []).append(r["results"][-1]["outcome"])
chk("distinct Theorem-1 instance keys", len(by))
chk("keys with any non-UNSAT final outcome", sum(1 for v in by.values() if any(o != "unsat" for o in v)), 0)

print("== Theorem 2 (|D|=3 reduce, <=4 refill): distinct instances all UNSAT after re-runs")
t2 = [r for r in jl("../results/orbit_lns_reduce3*.jsonl") if len(r["deleted_orbits"]) == 3]
by2 = {}
for r in t2: by2.setdefault(key(r), set()).add(r["results"][-1]["outcome"])
chk("distinct Theorem-2 instance keys", len(by2))
chk("keys never decided UNSAT", sum(1 for v in by2.values() if "unsat" not in v), 0)

print("== Theorem 3 (same-rank): exactness after the Phase-0 closure")
t3 = jl("../results/orbit_lns_same*.jsonl")
by3 = {}
for r in t3: by3.setdefault(key(r), set()).add(r["results"][-1]["outcome"])
capped = {k for k, v in by3.items() if "unsat" not in v}
chk("distinct same-rank instance keys", len(by3))
chk("keys not decided UNSAT in the Day-3 runs (capped or budget)", len(capped))
close = jl("../results/n1_close*.jsonl")
closed = collections.Counter(r["outcome"] for r in close)
chk("Phase-0 closure runs (all class combinations of the open shapes)", dict(closed))
chk("inequivalent 47s found in closure", sum(1 for r in close if r["outcome"] == "sat" and r.get("verified") and not r.get("identical_to_47")), 0)

print("== Theorems 4, 5 and Result 6 (5x5 / 6x6)")
for tag, pat, exp in [("5x5 rank-92 (T4)", "../results/C_555_reduce_w*.jsonl", 648), ("6x6 rank-152 (T5)", "../results/C_666_reduce_w*.jsonl", 1578)]:
    rs = jl(pat); by_ = {}
    for r in rs: by_.setdefault(key(r), set()).add(r["results"][-1]["outcome"])
    chk("%s distinct instances" % tag, len(by_), exp); chk("%s all UNSAT" % tag, all(v == {"unsat"} for v in by_.values()), True)

print("== census / filter")
b2 = json.load(open("../results/b2_dissection.json"))
chk("B2 outcomes", b2["outcomes"], {"budget": 93, "unsat": 7})
filt = {}
for l in open("../results/B2_filter.jsonl"):
    r = json.loads(l); filt[(r["group_index"], tuple(r["class_key"]))] = r["filter"]
fc = collections.Counter(filt.values())
chk("filter verdicts (distinct instances)", dict(fc))
chk("filter-eliminated total", fc.get("unsat", 0) + fc.get("count", 0) + fc.get("count7", 0), 892)
print("\n%d checks, %d mismatches" % (len(checks), sum(1 for c in checks if not c[3])))
