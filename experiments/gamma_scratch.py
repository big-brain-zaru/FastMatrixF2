"""Calibration: from-scratch search for a Gamma-invariant rank-47 scheme with the 47's own orbit-type multiset
(stabilizer classes), E1+E2 encoder + CaDiCaL 1.9.5, no hints. Day-2 result with the old encoder: unsolved at 50M."""
import json, time, sys
import symgroup as SG, fp_filter as FP, encode_opt as EO
from pysat.solvers import Cadical195
from flip_graph import verify, target_tensor
S47, G = FP.load_gamma(4); n = 4; T = set(target_tensor(n, n, n))
orbs = SG.orbits(S47, G, n); slots = [H for _, _, H in orbs]
conf = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
t0 = time.time(); cl, reps, terms, nv, neq = EO.encode(n, G, slots, T, True, True)
print("encoded %d vars %d clauses %d eqs in %.1fs; sizes %s" % (nv, len(cl), neq, time.time() - t0, sorted([12 // len(H) for H in slots], reverse=True)), flush=True)
s = Cadical195(bootstrap_with=cl); s.conf_budget(conf); t0 = time.time(); ok = s.solve_limited(); dt = time.time() - t0
rec = {"conf": conf, "t": round(dt, 1), "outcome": "budget" if ok is None else ("sat" if ok else "unsat")}
if ok:
    sch = EO.decode(set(l for l in s.get_model() if l > 0), terms, n); rec["verified"] = verify(sch, n, n, n); rec["rank"] = len(set(sch)); rec["is_the_47"] = set(sch) == set(S47)
    json.dump({"format": [4, 4, 4], "rank": len(set(sch)), "field": "F2", "verified_brent_f2": rec["verified"], "found_by": "gamma_scratch from scratch", "scheme_bitmasks": [list(t) for t in sorted(set(sch))]}, open("../results/gamma_scratch_rank%d.json" % len(set(sch)), "w"), indent=1)
print(json.dumps(rec), flush=True); open("../results/gamma_scratch.jsonl", "a").write(json.dumps(rec) + "\n")
