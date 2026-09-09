"""Top-down PERMUTATION symmetry census, "diagonal families": for each subgroup S <= S4 (up to conjugacy,
order >= 4) the diagonal sandwich group D_S = {(P,P,P) : P in S}, and its products with the slot symmetries
<cyc> and <cyc,tau>. Every instance = (group, multiset of stabilizer classes summing to 46); the fixed-point
filter (count bound + orbit-reduced projected SAT) runs first, survivors go to the E1+E2 encoder with
CaDiCaL 1.9.5. Exact throughout; every SAT decode host-verified. Log: results/PC2.jsonl"""
import itertools, json, time, collections
from pysat.solvers import Cadical195
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import verify, target_tensor

n = 4; T = set(target_tensor(n, n, n)); N = 16
I = [1 << r for r in range(n)]
perms = list(itertools.permutations(range(n)))
def pmat(p): return [1 << p[r] for r in range(n)]
def pmul(p, q): return tuple(p[q[i]] for i in range(n))
def closure_perm(gens):
    e = tuple(range(n)); S = {e}; fr = [e]
    while fr:
        g = fr.pop()
        for h in gens:
            k = pmul(h, g)
            if k not in S: S.add(k); fr.append(k)
    return frozenset(S)
# subgroups of S4 (all, by closure of pairs), up to conjugacy
subs = set()
for a in perms:
    for b in perms: subs.add(closure_perm([a, b]))
def conj(S, g):
    gi = tuple(sorted(range(n), key=lambda i: g[i]))
    return frozenset(pmul(pmul(g, s), gi) for s in S)
classes = []
for S in sorted(subs, key=len, reverse=True):
    if any(S in c for c in classes): continue
    classes.append({conj(S, g) for g in perms})
reps = [min(c, key=lambda s: sorted(s)) for c in classes]
print("S4 subgroup classes:", sorted(collections.Counter(len(s) for s in reps).items()), flush=True)

log = open("../results/PC2.jsonl", "a")
def run_group(name, gens):
    G = SG.closure(gens, n)
    chk = SG.check_generators(n, gens, ["g%d" % i for i in range(len(gens))])
    assert all(chk.values()), name
    invs = FP.involution_classes(G)
    cls = FP.subgroup_classes(G, FP.all_subgroups(G, n))
    cands = FP.slots_by_class(G, n, 46, 12, 150, cls)
    lb = FP.small_rank_table()
    bounds = [FP.count_bound(g, n, T, lb) for g in invs if FP.fixed_coords(g, n)]
    inv_fix = [g for g in invs if FP.fixed_coords(g, n)]
    stats = collections.Counter(); t0 = time.time()
    print("group %s: order %d, %d involution classes (%d with fixed coords), %d subgroup classes, %d multisets" % (name, len(G), len(invs), len(inv_fix), len(cls), len(cands)), flush=True)
    for slots, key in cands:
        sizes = [len(G) // len(H) for H in slots]
        rec = {"group": name, "order": len(G), "class_key": list(key), "sizes": sizes}
        # count bounds for every involution class with fixed coordinates
        killed = False
        for g, (b, _) in zip(inv_fix, bounds):
            if FP.fixed_count(G, slots, g) < b: killed = True; break
        if killed:
            rec["outcome"] = "count"; stats["count"] += 1
        else:
            v, dt, st = FP.run_filter(n, G, slots, T, conf=50000)
            rec["filter"] = v; rec["filter_t"] = round(dt, 2)
            if v == "unsat":
                rec["outcome"] = "filter"; stats["filter"] += 1
            else:
                t1 = time.time()
                cl, rp, terms, nv, neq = EO.encode(n, G, slots, T, True, True)
                s = Cadical195(bootstrap_with=cl); s.conf_budget(2_000_000); ok = s.solve_limited(); dt = time.time() - t1
                rec["vars"] = nv; rec["t"] = round(dt, 1)
                if ok is None: rec["outcome"] = "budget"
                elif ok is False: rec["outcome"] = "unsat"
                else:
                    sch = EO.decode(set(l for l in s.get_model() if l > 0), terms, n); okv = verify(sch, n, n, n)
                    rec["outcome"] = "sat"; rec["rank"] = len(set(sch)); rec["verified"] = okv
                    if okv:
                        fn = "../results/PC2_rank%d_%s_%s.json" % (len(set(sch)), name, "-".join(map(str, key)))
                        json.dump({"format": [4, 4, 4], "rank": len(set(sch)), "field": "F2", "verified_brent_f2": True, "found_by": "permcensus2 %s %s" % (name, sizes), "scheme_bitmasks": [list(t) for t in sorted(set(sch))]}, open(fn, "w"), indent=1)
                        rec["file"] = fn; print("  *** SAT rank %d verified -> %s" % (len(set(sch)), fn), flush=True)
                s.delete(); stats[rec["outcome"]] += 1
        log.write(json.dumps(rec) + "\n"); log.flush()
    print("  -> %s (%.0fs)" % (dict(stats), time.time() - t0), flush=True)

cyc = SG.perm_cyc(n); tau = SG.perm_tau(n)
for S in sorted(reps, key=len, reverse=True):
    if len(S) < 4: continue
    gens_S = [SG.perm_sandwich(pmat(p), pmat(p), pmat(p), n) for p in S if p != tuple(range(n))]
    nm = "diagS4sub%d_%s" % (len(S), "".join(str(x) for p in sorted(S)[1:2] for x in p))
    run_group(nm, gens_S)
    run_group(nm + "xcyc", gens_S + [cyc])
    run_group(nm + "xcyctau", gens_S + [cyc, tau])
print("PC2 done", flush=True)
