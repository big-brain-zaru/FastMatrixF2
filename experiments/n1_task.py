"""pool_runner task: Theorem 3 closure (same logic as n1_close.py, one (shape, class combination) per
instance, skipping combinations already recorded in results/n1_close.jsonl)."""
import argparse, glob, itertools, json, time
from pysat.solvers import Cadical195
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import verify

_c = {}


def parse_args(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("--conf", type=int, default=5_000_000); return ap.parse_args(argv)


def _setup():
    if "G" in _c: return
    S47, G = FP.load_gamma(4); n = 4
    orbs = SG.orbits(S47, G, n)
    classes = FP.subgroup_classes(G, FP.all_subgroups(G, n))
    by_order = {}
    for H, _ in classes: by_order.setdefault(len(H), []).append(H)
    _c.update(S47=S47, G=G, orbs=orbs, classes=classes, by_order=by_order, creps=EO.coord_orbit_reps(G, n), S47set=set(S47))


def instances(a):
    _setup()
    shapes = set()
    for f in glob.glob("../results/orbit_lns_same*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["results"] and r["results"][-1]["outcome"] == "sat":
                shapes.add((tuple(r["deleted_orbits"]), tuple(r["refill_sizes"]), tuple(r["refill_subgroup_orders"])))
    shapes.add(((1, 3, 5), (12, 6, 6), (1, 2, 2)))
    done = set()
    for f in glob.glob("../results/n1_close*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("outcome") in ("unsat", "sat"):
                done.add((tuple(r["D"]), tuple(r["refill_sizes"]), tuple(r["sub_orders"]), tuple(r["class_ids"])))
    out = []
    for (D, sizes, orders) in sorted(shapes):
        per_order = {}
        for o in orders: per_order[o] = per_order.get(o, 0) + 1
        lists = [list(itertools.combinations_with_replacement(range(len(_c["by_order"][o])), k)) for o, k in sorted(per_order.items())]
        for choice in itertools.product(*lists):
            orders_sorted = [o for o, k in sorted(per_order.items()) for _ in range(k)]
            idx = [i for blk in choice for i in blk]
            class_ids = [_c["classes"].index((_c["by_order"][o][i], [c for c in _c["classes"] if c[0] == _c["by_order"][o][i]][0][1])) for o, i in zip(orders_sorted, idx)]
            k = (D, sizes, orders, tuple(class_ids))
            if k in done: continue
            out.append({"key": "D%s_R%s_O%s_C%s" % ("-".join(map(str, D)), "-".join(map(str, sizes)), "-".join(map(str, orders)), "-".join(map(str, class_ids))),
                        "D": list(D), "refill_sizes": list(sizes), "sub_orders": list(orders), "class_ids": class_ids, "cost": sum(sizes)})
    return out


def run_instance(inst, a):
    _setup(); n = 4; G = _c["G"]; orbs = _c["orbs"]; classes = _c["classes"]
    D = set(inst["D"])
    kept = [t for i, (rep, orb, H) in enumerate(orbs) if i not in D for t in orb]
    deleted = [t for i, (rep, orb, H) in enumerate(orbs) if i in D for t in orb]
    R = EO.residual(kept, n)
    slots = [classes[ci][0] for ci in inst["class_ids"]]
    t0 = time.time()
    cl, reps, terms, nv, neq = EO.encode(n, G, slots, R, True, True, coord_reps=_c["creps"])
    def bits(t):
        return [(t[0] >> i) & 1 for i in range(16)] + [(t[1] >> i) & 1 for i in range(16)] + [(t[2] >> i) & 1 for i in range(16)]
    dbits = [bits(t) for t in deleted]; nxt = nv; d_lits = []
    for v in reps:
        es = []
        for tb in dbits:
            nxt += 1; e = nxt; es.append(e)
            lits = [v[i] if tb[i] else -v[i] for i in range(48)]
            for L in lits: cl.append([-e, L])
            cl.append([e] + [-L for L in lits])
        nxt += 1; d = nxt; d_lits.append(d)
        cl.append([-d] + es)
        for e in es: cl.append([-e, d])
    cl.append([-d for d in d_lits])
    s = Cadical195(bootstrap_with=cl); s.conf_budget(a.conf); ok = s.solve_limited(); dt = time.time() - t0
    rec = {"vars": nv, "t": round(dt, 1), "conf": a.conf}
    if ok is None: rec["outcome"] = "budget"
    elif ok is False: rec["outcome"] = "unsat"
    else:
        m = set(l for l in s.get_model() if l > 0); sch = EO.decode(m, terms, n); full = kept + sch
        rec["outcome"] = "sat"; rec["verified"] = verify(full, 4, 4, 4); rec["identical_to_47"] = set(full) == _c["S47set"]; rec["rank"] = len(set(full))
        if rec["verified"] and not rec["identical_to_47"]:
            fn = "../results/n1_close_rank%d_%s.json" % (len(set(full)), inst["key"])
            json.dump({"format": [4, 4, 4], "rank": len(set(full)), "field": "F2", "verified_brent_f2": True, "found_by": "n1_task %s" % inst["key"], "scheme_bitmasks": [list(t) for t in set(full)]}, open(fn, "w"), indent=1)
            rec["file"] = fn
    s.delete()
    return rec
