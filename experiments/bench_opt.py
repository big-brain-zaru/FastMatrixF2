"""Optimisation benchmark: old encoding (sym_break.encode / orbit_lns_n.build semantics) vs
encode_opt (orbit equations + gate sharing), and solver backends, on instances with known verdicts.
Every run logs to results/opt_bench.jsonl. Verdict agreement is asserted; any SAT decode is verified."""
import json, time, sys, itertools
from pysat.solvers import Cadical153, Cadical195, Cadical300, Glucose42
import symgroup as SG, fp_filter as FP, encode_opt as EO, sym_break as SB
from flip_graph import verify, target_tensor

LOG = open("../results/opt_bench.jsonl", "a")
def rec(**kw):
    kw["ts"] = time.strftime("%H:%M:%S"); LOG.write(json.dumps(kw) + "\n"); LOG.flush()
    print(json.dumps(kw), flush=True)

SOLVERS = {"cadical153": Cadical153, "cadical195": Cadical195, "glucose42": Glucose42}   # Cadical300 binding crashes the process (measured 20:22, 20:33); Kissat404 binding crashed on a 5-clause test

def run(clauses, terms, n, conf, solver="cadical153"):
    t0 = time.time(); s = SOLVERS[solver](bootstrap_with=clauses); t_load = time.time() - t0
    s.conf_budget(conf); ok = s.solve_limited(); dt = time.time() - t0
    out = {"verdict": "budget" if ok is None else ("sat" if ok else "unsat"), "t": round(dt, 1), "t_load": round(t_load, 2)}
    if ok:
        sch = EO.decode(set(l for l in s.get_model() if l > 0), terms, n)
        out["verified"] = verify(sch, n, n, n); out["rank"] = len(sch)
    s.delete(); return out

n = 4; T4 = set(target_tensor(4, 4, 4))
groups = json.load(open("../results/census_n4_r46_cyclic.json"))["worklist"]
wl = json.load(open("../results/B2_worklist.json"))
def census_instance(gi, sizes):
    G = [tuple(g) for g in groups[gi]["elements"]]
    classes = FP.subgroup_classes(G, FP.all_subgroups(G, n))
    key = [w for w in wl["work"] if w["group_index"] == gi and w["orbit_sizes"] == sizes][0]["class_key"]
    return G, [classes[ci][0] for ci in key]

which = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---- A. census instances with known verdicts (B2 log): g7 UNSAT 191 s, g33 UNSAT 139 s, g2 budget 664 s
if which in ("all", "census"):
    for gi, sizes, known in [(7, [12, 12, 12, 6, 3, 1], "unsat 191s"), (33, [4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 1, 1, 1, 1], "unsat 139s"), (2, [12, 12, 12, 6, 3, 1], "budget@1M 664s")]:
        G, slots = census_instance(gi, sizes)
        t0 = time.time(); cl_old, _, terms_old, nv_old = SB.encode(n, G, slots, T4, False, False); te_old = time.time() - t0
        t0 = time.time(); cl_new, _, terms_new, nv_new, neq = EO.encode(n, G, slots, T4, True, True); te_new = time.time() - t0
        r = run(cl_old, terms_old, n, 200000)
        rec(stage="census", inst="g%d %s" % (gi, sizes), known=known, enc="old", solver="cadical153", conf=200000, vars=nv_old, clauses=len(cl_old), t_encode=round(te_old, 1), **r)
        for sv in SOLVERS:
            r = run(cl_new, terms_new, n, 200000, sv)
            rec(stage="census", inst="g%d %s" % (gi, sizes), known=known, enc="opt", solver=sv, conf=200000, vars=nv_new, clauses=len(cl_new), eqs=neq, t_encode=round(te_new, 1), **r)
        for sv in ("cadical153", "cadical195"):
            r = run(cl_new, terms_new, n, 1000000, sv)
            rec(stage="census", inst="g%d %s" % (gi, sizes), known=known, enc="opt", solver=sv, conf=1000000, vars=nv_new, clauses=len(cl_new), eqs=neq, **r)

# ---- B. orbit-level LNS hard shape around the 47: delete orbits {2,3,5}, refill [12,6,3,2] with stabilizer orders [1,2,4,6]
if which in ("all", "lns"):
    S47, G = FP.load_gamma(4)
    orbs = SG.orbits(S47, G, n)
    kept = [t for i, (rep, orb, H) in enumerate(orbs) if i not in (2, 3, 5) for t in orb]
    R = EO.residual(kept, n)
    classes = FP.subgroup_classes(G, FP.all_subgroups(G, n))
    by_order = {}
    for H, _ in classes: by_order.setdefault(len(H), []).append(H)
    combos = list(itertools.product(by_order[2], by_order[6]))
    print("order-2 classes %d, order-6 classes %d -> %d instances of the hard shape" % (len(by_order[2]), len(by_order[6]), len(combos)))
    for k, (H2, H6) in enumerate(combos):
        # the recorded shape is refill sizes [12,6,3,2] = stabilizer orders [1,2,4,6]
        slots = [by_order[1][0], H2, by_order[4][0], H6]
        sizes = [12 // len(H) for H in slots]
        t0 = time.time(); cl_new, _, terms_new, nv_new, neq = EO.encode(n, G, slots, R, True, True); te_new = time.time() - t0
        r = run(cl_new, terms_new, n, 1000000)
        rec(stage="lns_hard", inst="D={2,3,5} refill %s combo %d" % (sizes, k), enc="opt", solver="cadical153", conf=1000000, vars=nv_new, clauses=len(cl_new), eqs=neq, t_encode=round(te_new, 1), **r)
        if k < 3:
            t0 = time.time(); cl_old, _, terms_old, nv_old, _ = EO.encode(n, G, slots, R, False, False); te_old = time.time() - t0
            r = run(cl_old, terms_old, n, 200000)
            rec(stage="lns_hard", inst="D={2,3,5} refill %s combo %d" % (sizes, k), enc="old", solver="cadical153", conf=200000, vars=nv_old, clauses=len(cl_old), t_encode=round(te_old, 1), **r)

# ---- C. 5x5 orbit LNS (C3): 30 reduce instances, old vs opt, encode and solve time split
if which in ("all", "c5"):
    n5 = 5; T5 = set(target_tensor(5, 5, 5))
    rec5 = [tuple(t) for t in json.load(open("../results/records/lille_555_rank93.json"))["scheme_bitmasks"]]
    G5 = SG.closure([SG.perm_cyc(5)], 5)
    orbs5 = SG.orbits(rec5, G5, 5)
    creps5 = EO.coord_orbit_reps(G5, 5)
    triv = frozenset([tuple(range(75))]); full = frozenset(G5)
    tot = {"old": [0.0, 0.0], "opt": [0.0, 0.0]}; agree = 0; cnt = 0
    for (i, j) in list(itertools.combinations(range(len(orbs5)), 2))[:30]:
        kept = [t for k, (rep, orb, H) in enumerate(orbs5) if k not in (i, j) for t in orb]
        deleted = len(orbs5[i][1]) + len(orbs5[j][1])
        # refill one term fewer with size-3 orbits and fixed points: compositions of deleted-1 into 3s and 1s (<=4 orbits)
        target = deleted - 1
        comps = [(a, b) for a in range(0, 5) for b in range(0, 5) if 3 * a + b == target and a + b <= 4]
        R = EO.residual(kept, 5)
        for (a, b) in comps:
            slots = [triv] * a + [full] * b
            t0 = time.time(); cl_old, _, terms_old, nv_old, _ = EO.encode(5, G5, slots, R, False, False); te_old = time.time() - t0
            r_old = run(cl_old, terms_old, 5, 2000000)
            t0 = time.time(); cl_new, _, terms_new, nv_new, neq = EO.encode(5, G5, slots, R, True, True, coord_reps=creps5); te_new = time.time() - t0
            r_new = run(cl_new, terms_new, 5, 2000000)
            cnt += 1; agree += (r_old["verdict"] == r_new["verdict"])
            tot["old"][0] += te_old; tot["old"][1] += r_old["t"]; tot["opt"][0] += te_new; tot["opt"][1] += r_new["t"]
            rec(stage="c5", inst="D={%d,%d} refill 3x%d+1x%d" % (i, j, a, b), enc_old={"vars": nv_old, "clauses": len(cl_old), "t_encode": round(te_old, 2), **r_old},
                enc_opt={"vars": nv_new, "clauses": len(cl_new), "eqs": neq, "t_encode": round(te_new, 2), **r_new})
    rec(stage="c5_summary", instances=cnt, verdicts_agree=agree, old_encode_s=round(tot["old"][0], 1), old_solve_s=round(tot["old"][1], 1), opt_encode_s=round(tot["opt"][0], 1), opt_solve_s=round(tot["opt"][1], 1))
print("done")
