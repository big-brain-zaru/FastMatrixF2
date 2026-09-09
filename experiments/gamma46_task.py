"""pool_runner task: FULL from-scratch census of Gamma-invariant rank-46 schemes (Gamma = the 4x4 rank-47
symmetry group, order 12). Instances = every multiset of stabilizer classes with orbit sizes summing to 46
(<= --max-slots slots), fixed-point filter first (count bound + orbit-reduced projected SAT), survivors solved
with encode_opt (E1+E2) + CaDiCaL 1.9.5 at --conf conflicts. The Day-2 group-SAT sweep refuted 488 such
assignments with the old encoder; the from-scratch rank-47 instance of the same kind was solved on 6 Sep.
  python pool_runner.py --task gamma46_task --workers 8 --wall 3600 --log ../results/G46_census.jsonl -- --conf 20000000
"""
import argparse, json, time
from pysat.solvers import Cadical195
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import verify, target_tensor

n = 4; _c = {}

def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-slots", type=int, default=14)
    ap.add_argument("--conf", type=int, default=20_000_000)
    ap.add_argument("--filter-conf", type=int, default=100_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter-only", action="store_true", help="apply count bound + projected filter only; survivors logged as 'pass'")
    return ap.parse_args(argv)

def _setup():
    if "G" in _c: return
    S47, G = FP.load_gamma(4)
    _c["G"] = G; _c["T"] = set(target_tensor(n, n, n)); _c["classes"] = FP.subgroup_classes(G, FP.all_subgroups(G, n))
    _c["creps"] = EO.coord_orbit_reps(G, n); _c["invs"] = FP.involution_classes(G); _c["lb"] = FP.small_rank_table()

def instances(a):
    _setup(); G = _c["G"]
    cands = FP.slots_by_class(G, n, 46, a.max_slots, 10**9, _c["classes"])
    out = []
    for slots, key in cands:
        sizes = [len(G) // len(H) for H in slots]
        out.append({"key": "G46_" + "-".join(map(str, key)), "class_key": list(key), "sizes": sizes, "cost": -len(sizes)})   # fewest slots first
    if a.limit: out = out[:a.limit]
    return out

def run_instance(inst, a):
    _setup(); G = _c["G"]; T = _c["T"]; classes = _c["classes"]
    slots = [classes[ci][0] for ci in inst["class_key"]]
    rec = {"sizes": inst["sizes"]}
    t0 = time.time()
    # zero-cost count bounds for every involution class with fixed coordinates
    for g in _c["invs"]:
        if not FP.fixed_coords(g, n): continue
        b, _ = FP.count_bound(g, n, T, _c["lb"])
        if FP.fixed_count(G, slots, g) < b:
            rec.update(outcome="count", t=round(time.time() - t0, 2)); return rec
    v, dt, st = FP.run_filter(n, G, slots, T, conf=a.filter_conf)
    rec["filter"] = v; rec["filter_t"] = round(dt, 2)
    if v == "unsat":
        rec.update(outcome="filter", t=round(time.time() - t0, 2)); return rec
    if a.filter_only:
        rec.update(outcome="pass", t=round(time.time() - t0, 2)); return rec
    cl, reps, terms, nv, neq = EO.encode(n, G, slots, T, True, True, coord_reps=_c["creps"])
    s = Cadical195(bootstrap_with=cl); s.conf_budget(a.conf); ok = s.solve_limited(); dt = time.time() - t0
    rec.update(vars=nv, eqs=neq, conf=a.conf, t=round(dt, 1))
    if ok is None: rec["outcome"] = "budget"
    elif ok is False: rec["outcome"] = "unsat"
    else:
        sch = EO.decode(set(l for l in s.get_model() if l > 0), terms, n); okv = verify(sch, n, n, n)
        rec.update(outcome="sat", rank=len(set(sch)), verified=okv)
        if okv:
            fn = "../results/G46_rank%d_%s.json" % (len(set(sch)), inst["key"])
            json.dump({"format": [4, 4, 4], "rank": len(set(sch)), "field": "F2", "verified_brent_f2": True, "found_by": "gamma46_task %s sizes %s" % (inst["key"], inst["sizes"]), "scheme_bitmasks": [list(t) for t in sorted(set(sch))]}, open(fn, "w"), indent=1)
            rec["file"] = fn
    s.delete(); return rec
