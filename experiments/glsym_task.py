"""pool_runner task: census of ODD-ORDER cyclic sandwich symmetries in GL(4,2)^3 (non-permutation
groups: fixed-point-free order 3, Singer cycles of order 15, orders 5 and 7, and mixed triples).
Instances = (class-triple canonical under generator powers, orbit-size composition of 46).
Each instance is a from-scratch G-invariant rank-46 search with glsym.encode, CaDiCaL 1.9.5.
  python pool_runner.py --task glsym_task --workers 4 --wall 1800 --log ../results/GL_odd_w.jsonl -- --rank 46 --max-slots 8 --per-group 40 --conf 2000000
"""
import argparse, itertools, json, math, time, collections
from pysat.solvers import Cadical195
import glsym as GL
from flip_graph import verify, target_tensor

n = 4
_c = {}

def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=46)
    ap.add_argument("--max-slots", type=int, default=8)
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--conf", type=int, default=2_000_000)
    ap.add_argument("--min-order", type=int, default=3)
    ap.add_argument("--tag", default="GLodd")
    return ap.parse_args(argv)

def classes():
    d = json.load(open("../results/gl42_odd_classes.json"))
    return sorted(((v["order"], tuple(v["rep"]), k) for k, v in d.items()), key=lambda x: (x[0], x[2]))

def class_key_of(X):
    """classification key (order, kernel dims) of an odd-order matrix -- same as in the enumeration"""
    if "irr" not in _c:
        _c["irr"] = [(1, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 0, 0, 1), (1, 0, 0, 1, 1), (1, 1, 1, 1, 1)]
    def polyval(p, X):
        acc = (0,) * n; P = GL.ident(n)
        for c in p:
            if c: acc = tuple(a ^ b for a, b in zip(acc, P))
            P = GL.mmul(P, X, n)
        return acc
    def rank(M):
        rows = list(M); r = 0
        while rows:
            p = rows.pop()
            if not p: continue
            r += 1; hb = p.bit_length() - 1; rows = [x ^ p if (x >> hb) & 1 else x for x in rows if x]
        return r
    return (GL.mat_order(X, n),) + tuple(n - rank(polyval(p, X)) for p in _c["irr"])

def mpow(X, k):
    P = GL.ident(n)
    for _ in range(k): P = GL.mmul(P, X, n)
    return P

def instances(a):
    cls = classes()
    reps = [(o, X) for o, X, k in cls]
    triples = {}
    for (o1, X), (o2, Y), (o3, Z) in itertools.product(reps, repeat=3):
        order = math.lcm(o1, o2, o3)
        if order < a.min_order: continue
        # canonical form under generator change k coprime to order: min over k of the class-key triple
        keys = []
        for k in range(1, order):
            if math.gcd(k, order) != 1: continue
            keys.append((class_key_of(mpow(X, k)), class_key_of(mpow(Y, k)), class_key_of(mpow(Z, k))))
        canon = min(keys)
        if canon in triples: continue
        triples[canon] = (order, (X, Y, Z))
    out = []
    for canon, (order, g) in sorted(triples.items(), key=lambda kv: -kv[1][0]):
        divs = [d for d in range(1, order + 1) if order % d == 0]
        sizes = sorted(divs, reverse=True)
        comps = []
        def rec(t, maxs, k, cur):
            if len(comps) >= a.per_group: return
            if t == 0: comps.append(list(cur)); return
            if k == 0: return
            for s_ in sizes:
                if s_ <= maxs and s_ <= t:
                    rec(t - s_, s_, k - 1, cur + [s_])
        for parts in range(1, a.max_slots + 1):
            rec(a.rank, max(sizes), parts, [])
            if len(comps) >= a.per_group: break
        gid = "o%d_%s" % (order, "-".join("%d" % (sum(c[1:]) if False else 0) for c in [canon[0]]))  # placeholder id
        gid = "o%d_" % order + "_".join("k%s" % ("".join(map(str, ck[1:]))) for ck in canon)
        for ci, comp in enumerate(comps):
            out.append({"key": "%s_m%d" % (gid, ci), "gen": [list(X) for X in g], "order": order, "sizes": comp, "cost": -order * 10 + len(comp)})
    return out

def run_instance(inst, a):
    T = set(target_tensor(n, n, n))
    g = tuple(tuple(M) for M in inst["gen"])
    G = GL.group_closure([g], n)
    if not GL.check_group(G, n): return {"outcome": "error", "error": "group check failed"}
    order = len(G)
    # stabilizer subgroup for orbit size s = unique subgroup of order order/s in a cyclic group
    subs = {}
    for h in G:
        H = GL.group_closure([h], n); subs.setdefault(len(H), H)
    slots = [subs[order // s] for s in inst["sizes"]]
    res = GL.solve(n, G, slots, T, conf=a.conf)
    rec = {"order": order, "sizes": inst["sizes"], "vars": res["vars"], "clauses": res["clauses"], "t_encode": res["t_encode"], "t": res["t"], "outcome": res["outcome"], "conf": a.conf}
    if res["outcome"] == "sat":
        rec["rank"] = res["rank"]; rec["verified"] = res["verified"]
        if res["verified"]:
            fn = "../results/%s_rank%d_%s.json" % (a.tag, res["rank"], inst["key"])
            json.dump({"format": [n, n, n], "rank": res["rank"], "field": "F2", "verified_brent_f2": True, "found_by": "glsym_task %s" % inst["key"],
                       "group_generator_XYZ": inst["gen"], "scheme_bitmasks": [list(t) for t in sorted(set(res["scheme"]))]}, open(fn, "w"), indent=1)
            rec["file"] = fn
    return rec
