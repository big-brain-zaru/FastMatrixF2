"""Phase 0 item 4: can theorem-grade refutations carry DRAT certificates through this binding?
python-sat emits proofs only for Glucose/Lingeling. Measure on a sample of Theorem 1 instances
(delete <=2 orbits of the 47, refill one term fewer): Glucose 4.2 with proof vs CaDiCaL 1.9.5 without.
Writes DIMACS + DRAT into results/certificates/ for external checking (drat-trim not installed here)."""
import glob, json, os, time, itertools
from pysat.solvers import Glucose42, Cadical195
from pysat.formula import CNF
import symgroup as SG, fp_filter as FP, encode_opt as EO

n = 4
S47, G = FP.load_gamma(4)
orbs = SG.orbits(S47, G, n)
classes = FP.subgroup_classes(G, FP.all_subgroups(G, n))
by_order = {}
for H, _ in classes: by_order.setdefault(len(H), []).append(H)
os.makedirs("../results/certificates", exist_ok=True)
# sample: first 12 distinct shapes from the Theorem-1 logs (reduce, |D|<=2)
shapes = []
for f in sorted(glob.glob("../results/orbit_lns_reduce_w*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if len(r["deleted_orbits"]) <= 2:
            k = (tuple(r["deleted_orbits"]), tuple(r["refill_sizes"]), tuple(r["refill_subgroup_orders"]))
            if k not in shapes: shapes.append(k)
    if len(shapes) >= 12: break
shapes = shapes[:12]
log = open("../results/cert_test.jsonl", "a")
for (D, sizes, orders) in shapes:
    kept = [t for i, (rep, orb, H) in enumerate(orbs) if i not in D for t in orb]
    R = EO.residual(kept, n)
    slots = [by_order[o][0] for o in orders]          # first class of each order (one representative instance per shape)
    cl, reps, terms, nv, neq = EO.encode(n, G, slots, R, True, True)
    rec = {"D": list(D), "refill": list(sizes), "orders": list(orders), "vars": nv, "clauses": len(cl)}
    t0 = time.time(); s = Cadical195(bootstrap_with=cl); s.conf_budget(5_000_000); ok = s.solve_limited(); s.delete()
    rec["cadical195"] = {"verdict": "budget" if ok is None else ("sat" if ok else "unsat"), "t": round(time.time() - t0, 1)}
    t0 = time.time(); g = Glucose42(bootstrap_with=cl, with_proof=True); g.conf_budget(5_000_000); okg = g.solve_limited()
    dt = time.time() - t0; proof = g.get_proof() if okg is False else None; g.delete()
    rec["glucose42"] = {"verdict": "budget" if okg is None else ("sat" if okg else "unsat"), "t": round(dt, 1), "proof_lines": len(proof) if proof else None}
    if proof:
        tag = "T1_D%s_R%s" % ("-".join(map(str, D)), "-".join(map(str, sizes)))
        CNF(from_clauses=cl).to_file("../results/certificates/%s.cnf" % tag)
        with open("../results/certificates/%s.drat" % tag, "w") as fh:
            fh.write("\n".join(proof) + "\n")
        rec["files"] = tag
    log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
print("done", flush=True)
