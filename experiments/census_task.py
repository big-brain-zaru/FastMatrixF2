"""pool_runner task: census instances (from a B2-style worklist) with the optimised encoder and
CaDiCaL 1.9.5. Instance = (group_index, class_key). Optional filter to re-run only instances whose
previous outcome was 'budget'.

  python pool_runner.py --task census_task --workers 12 --wall 600 --log ../results/B3_w.jsonl -- \
      --worklist ../results/B2_worklist.json --census ../results/census_n4_r46_cyclic.json \
      --prev ../results/B2_w*.jsonl --only budget --conf 1000000
"""
import argparse, glob, json, time
from pysat.solvers import Cadical195, Cadical153
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import verify, target_tensor

_cache = {}


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prev", default=None, help="glob of previous outcome logs; with --only, select by previous outcome")
    ap.add_argument("--only", default=None, choices=[None, "budget", "unrun"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=int, default=1000000)
    ap.add_argument("--solver", default="cadical195")
    ap.add_argument("--tag", default="B3")
    return ap.parse_args(argv)


def instances(a):
    wl = json.load(open(a.worklist))
    work = wl["work"]
    prev = {}
    if a.prev:
        for f in glob.glob(a.prev):
            for l in open(f):
                r = json.loads(l); prev[(r["group_index"], tuple(r["class_key"]))] = r["outcome"]
    out = []
    for w in work:
        k = (w["group_index"], tuple(w["class_key"]))
        if a.only == "budget" and prev.get(k) != "budget": continue
        if a.only == "unrun" and k in prev: continue
        out.append({"key": "g%d:%s" % (w["group_index"], "-".join(map(str, w["class_key"]))), "group_index": w["group_index"],
                    "class_key": w["class_key"], "orbit_sizes": w["orbit_sizes"], "group_order": w["group_order"],
                    "cost": len(w["orbit_sizes"]) * 1.0 / max(w["group_order"], 1)})
    if a.limit: out = out[:a.limit]
    return out


def run_instance(inst, a):
    n = a.n
    if "groups" not in _cache:
        _cache["groups"] = json.load(open(a.census))["worklist"]; _cache["T"] = set(target_tensor(n, n, n))
    gi = inst["group_index"]
    G = [tuple(g) for g in _cache["groups"][gi]["elements"]]
    if gi not in _cache:
        _cache[gi] = (FP.subgroup_classes(G, FP.all_subgroups(G, n)), EO.coord_orbit_reps(G, n))
    classes, creps = _cache[gi]
    slots = [classes[ci][0] for ci in inst["class_key"]]
    t0 = time.time()
    cl, reps, terms, nv, neq = EO.encode(n, G, slots, _cache["T"], True, True, coord_reps=creps)
    t_enc = time.time() - t0
    S = {"cadical195": Cadical195, "cadical153": Cadical153}[a.solver]
    s = S(bootstrap_with=cl); s.conf_budget(a.conf); ok = s.solve_limited(); dt = time.time() - t0
    rec = {"vars": nv, "eqs": neq, "t_encode": round(t_enc, 2), "t": round(dt, 1), "conf": a.conf, "solver": a.solver, "enc": "E1+E2"}
    if ok is None: rec["outcome"] = "budget"
    elif ok is False: rec["outcome"] = "unsat"
    else:
        sch = EO.decode(set(l for l in s.get_model() if l > 0), terms, n)
        okv = verify(sch, n, n, n); rec["outcome"] = "sat"; rec["rank"] = len(sch); rec["verified"] = okv
        if okv:
            fn = "../results/%s_rank%d_%s.json" % (a.tag, len(sch), inst["key"].replace(":", "_"))
            json.dump({"format": [n, n, n], "rank": len(sch), "field": "F2", "verified_brent_f2": True,
                       "found_by": "census_task %s sizes %s" % (inst["key"], inst["orbit_sizes"]),
                       "group_elements": [list(g) for g in G], "scheme_bitmasks": [list(t) for t in sch]}, open(fn, "w"), indent=1)
            rec["file"] = fn
    s.delete()
    return rec
