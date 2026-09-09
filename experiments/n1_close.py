"""Close Theorem 3 exactly. Day-3 N1 instances whose enumeration stopped at the 5-solution cap with
only relabelings of the 47 are NOT decided. Re-run every such shape (all stabilizer-class combinations
consistent with the recorded subgroup orders) with the optimised encoder, CaDiCaL 1.9.5 and a
SET-LEVEL blocking clause: the refill may not reproduce the deleted orbit set D. Because D is a union of
full Gamma-orbits, the refill equals D as a set iff every refill representative lies in D, so the block
is  OR_i (rep_i not in D). Any remaining SAT is a genuinely different term set (then host-verified and
saved); UNSAT is the exact statement that no other Gamma-invariant 47 exists for that shape."""
import glob, itertools, json, time
from pysat.solvers import Cadical195
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import verify

n = 4
S47, G = FP.load_gamma(4)
orbs = SG.orbits(S47, G, n)
classes = FP.subgroup_classes(G, FP.all_subgroups(G, n))
by_order = {}
for H, _ in classes: by_order.setdefault(len(H), []).append(H)

# shapes to close: capped same-rank instances + the one shape left undecided at 100M
shapes = set()
for f in glob.glob("../results/orbit_lns_same*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["results"] and r["results"][-1]["outcome"] == "sat":
            shapes.add((tuple(r["deleted_orbits"]), tuple(r["refill_sizes"]), tuple(r["refill_subgroup_orders"])))
shapes.add(((1, 3, 5), (12, 6, 6), (1, 2, 2)))
print("shapes to close:", len(shapes), flush=True)

def bits(t):
    a, b, c = t
    return [(a >> i) & 1 for i in range(16)] + [(b >> i) & 1 for i in range(16)] + [(c >> i) & 1 for i in range(16)]

log = open("../results/n1_close.jsonl", "a")
S47set = set(S47)
for (D, sizes, orders) in sorted(shapes):
    kept = [t for i, (rep, orb, H) in enumerate(orbs) if i not in D for t in orb]
    deleted = [t for i, (rep, orb, H) in enumerate(orbs) if i in D for t in orb]
    R = EO.residual(kept, n)
    # all class combinations: for each order, combinations with replacement over its classes
    per_order = {}
    for o in orders: per_order[o] = per_order.get(o, 0) + 1
    choice_lists = [list(itertools.combinations_with_replacement(by_order[o], k)) for o, k in sorted(per_order.items())]
    for choice in itertools.product(*choice_lists):
        slots = [H for blk in choice for H in blk]
        t0 = time.time()
        cl, reps, terms, nv, neq = EO.encode(n, G, slots, R, True, True)
        nxt = [nv]
        def new():
            nxt[0] += 1; return nxt[0]
        dbits = [bits(t) for t in deleted]
        d_lits = []
        for v in reps:
            es = []
            for tb in dbits:
                e = new(); es.append(e)
                lits = [v[i] if tb[i] else -v[i] for i in range(48)]
                for L in lits: cl.append([-e, L])          # e -> bits match
                cl.append([e] + [-L for L in lits])         # bits match -> e
            d = new(); d_lits.append(d)
            cl.append([-d] + es)                              # d -> some e
            for e in es: cl.append([-e, d])                   # e -> d
        cl.append([-d for d in d_lits])                       # not all reps in D
        s = Cadical195(bootstrap_with=cl); s.conf_budget(5_000_000); ok = s.solve_limited(); dt = time.time() - t0
        rec = {"D": list(D), "refill_sizes": list(sizes), "sub_orders": list(orders), "class_ids": [classes.index((H, [c for c in classes if c[0] == H][0][1])) for H in slots],
               "vars": nv, "t": round(dt, 1), "conf": 5_000_000}
        if ok is None: rec["outcome"] = "budget"
        elif ok is False: rec["outcome"] = "unsat"
        else:
            m = set(l for l in s.get_model() if l > 0); sch = EO.decode(m, terms, n); full = kept + sch
            rec["outcome"] = "sat"; rec["verified"] = verify(full, 4, 4, 4); rec["identical_to_47"] = set(full) == S47set; rec["rank"] = len(set(full))
            if rec["verified"] and not rec["identical_to_47"]:
                fn = "../results/n1_close_rank%d_D%s.json" % (len(set(full)), "-".join(map(str, D)))
                json.dump({"format": [4, 4, 4], "rank": len(set(full)), "field": "F2", "verified_brent_f2": True, "found_by": "n1_close %s" % rec, "scheme_bitmasks": [list(t) for t in set(full)]}, open(fn, "w"), indent=1)
                rec["file"] = fn
        s.delete()
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
print("done", flush=True)
