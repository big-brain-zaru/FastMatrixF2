"""Orbit-level LNS for n x n x n F2 schemes under an arbitrary symmetry group (any n).

Generalizes orbit_lns.py (hardcoded to 4x4 and the 47's order-12 group) in two ways:
  1. n-generic via symgroup.py (bit width 3n^2 instead of 48).
  2. RESIDUAL encoding: kept orbits are constants, so only the refill orbits are symbolic. The 4x4
     version encoded every term symbolically and pinned kept ones with unit clauses, which made
     each instance far larger than necessary.

Modes: reduce (refill one term fewer -> rank-1 improvement) and same (same rank, original blocked
as a set -> diversification).

Usage:
  python orbit_lns_n.py --scheme ../results/records/lille_555_rank93.json --n 5 --group cyc \
      --mode reduce --max-delete 2 --max-new 4 --stride 8 --offset 0
"""
import argparse, itertools, json, time, sys, os
from pysat.solvers import Cadical153, Cadical195
import encode_opt as EO
import symgroup as SG
from flip_graph import verify, target_tensor


def build_group(n, spec):
    if spec == "gamma":                      # the 4x4 rank-47 symmetry group S3 x <tau> (order 12), from the stabilizer file
        import fp_filter as FP
        assert n == 4
        return FP.load_gamma(4)[1]
    gens = {"cyc": [SG.perm_cyc(n)], "tau": [SG.perm_tau(n)],
            "s3": [SG.perm_cyc(n), SG.perm_tau(n)]}[spec]
    chk = SG.check_generators(n, gens, [spec] * len(gens))
    assert all(chk.values()), "generator check failed: %s" % chk
    return SG.closure(gens, n)


def subgroups_of(G, n):
    ident = tuple(range(SG.L_of(n)))
    subs = {frozenset([ident])}
    for a in G:
        for b in G:
            subs.add(frozenset(SG.closure([a, b], n)))
    return sorted(subs, key=len)


def conj_classes(subs, G, n):
    def conj(H, g):
        gi = tuple(SG.inverse_index(g))
        return frozenset(SG.compose(SG.compose(g, h), gi) for h in H)
    classes = []
    for H in subs:
        if any(conj(H, g) in cls for cls in classes for g in G):
            continue
        classes.append({conj(H, g) for g in G})
    return [min(cls, key=lambda x: sorted(x)) for cls in classes]


def residual(kept_terms, n):
    """Coordinates where target - sum(kept) is 1 over F2."""
    N = n * n
    acc = set(target_tensor(n, n, n))
    for (a, b, c) in kept_terms:
        for ia in range(N):
            if not (a >> ia) & 1:
                continue
            for ib in range(N):
                if not (b >> ib) & 1:
                    continue
                for ic in range(N):
                    if (c >> ic) & 1:
                        k = (ia, ib, ic)
                        if k in acc:
                            acc.remove(k)
                        else:
                            acc.add(k)
    return acc


def build(refill_subs, G, n, R):
    """Encode only the refill orbits. Returns (solver, rep_blocks, image_terms, nvars, nclauses)."""
    N = n * n
    L = 3 * N
    state = {"nv": 0}
    clauses = []

    def new():
        state["nv"] += 1
        return state["nv"]

    def AND(x, y):
        z = new()
        clauses.append([-z, x])
        clauses.append([-z, y])
        clauses.append([z, -x, -y])
        return z

    reps = []
    terms = []
    for H in refill_subs:
        v = [new() for _ in range(L)]
        reps.append(v)
        for h in H:
            for i in range(L):
                j = h[i]
                if j != i:
                    clauses.append([-v[i], v[j]])
                    clauses.append([v[i], -v[j]])
        clauses.append(v[0:N])
        clauses.append(v[N:2 * N])
        clauses.append(v[2 * N:3 * N])
        seen = set()
        cosets = []
        for g in G:
            key = frozenset(SG.compose(g, h) for h in H)
            if key in seen:
                continue
            seen.add(key)
            cosets.append(g)
        for g in cosets:
            inv = SG.inverse_index(g)
            terms.append([v[inv[j]] for j in range(L)])

    for ia in range(N):
        for ib in range(N):
            ab = [AND(tv[ia], tv[N + ib]) for tv in terms]
            for ic in range(N):
                lits = [AND(ab[t], terms[t][2 * N + ic]) for t in range(len(terms))]
                rhs = 1 if (ia, ib, ic) in R else 0
                cur = lits[0]
                for l in lits[1:]:
                    z = new()
                    clauses.append([-z, cur, l])
                    clauses.append([-z, -cur, -l])
                    clauses.append([z, -cur, l])
                    clauses.append([z, cur, -l])
                    cur = z
                clauses.append([cur] if rhs else [-cur])
    return Cadical153(bootstrap_with=clauses), reps, terms, state["nv"], len(clauses)


def decode(model, terms, n):
    N = n * n
    out = []
    for tv in terms:
        b = [1 if x in model else 0 for x in tv]
        out.append((sum(b[i] << i for i in range(N)),
                    sum(b[N + i] << i for i in range(N)),
                    sum(b[2 * N + i] << i for i in range(N))))
    return out


def compositions(total, sizes, max_parts):
    sizes = sorted(sizes, reverse=True)

    def rec(t, maxs, parts):
        if t == 0:
            yield []
            return
        if parts == 0:
            return
        for s in sizes:
            if s <= maxs and s <= t:
                for rest in rec(t - s, s, parts - 1):
                    yield [s] + rest
    return rec(total, max(sizes), max_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--group", default="cyc", choices=["cyc", "tau", "s3", "gamma"])
    ap.add_argument("--mode", choices=["reduce", "same"], default="reduce")
    ap.add_argument("--min-delete", type=int, default=1)
    ap.add_argument("--max-delete", type=int, default=2)
    ap.add_argument("--min-new", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--distinct", type=int, default=5)
    ap.add_argument("--conf", type=int, default=1000000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--enc", choices=["old", "opt"], default="opt", help="opt = orbit equations + gate sharing (encode_opt, validated 09-05)")
    ap.add_argument("--solver", choices=["cadical153", "cadical195"], default="cadical195")
    ap.add_argument("--tag", default="orbitn", help="prefix for saved scheme files and default log names (avoid overwriting earlier campaigns)")
    ap.add_argument("--no-block-set", action="store_true", help="same mode: do NOT add the set-level block of the input scheme (default: block, so every SAT is a genuinely different term set)")
    a = ap.parse_args()

    n = a.n
    G = build_group(n, a.group)
    subs = conj_classes(subgroups_of(G, n), G, n)
    by_size = {}
    for H in subs:
        by_size.setdefault(len(G) // len(H), []).append(H)
    S = [tuple(t) for t in json.load(open(a.scheme))["scheme_bitmasks"]]
    assert verify(S, n, n, n), "input scheme does not verify"
    assert SG.is_invariant(S, G, n), "input scheme is not invariant under the chosen group"
    orb = SG.orbits(S, G, n)
    sizes_orig = [len(o) for _, o, _ in orb]
    classes_txt = ", ".join("%d:%d" % (k, len(v)) for k, v in sorted(by_size.items()))
    print("n=%d group=%s |G|=%d rank=%d orbits=%d orbit_sizes=%s stabilizer_classes={%s}"
          % (n, a.group, len(G), len(S), len(orb), sorted(set(sizes_orig)), classes_txt), flush=True)

    instances = []
    for nd in range(a.min_delete, a.max_delete + 1):
        for D in itertools.combinations(range(len(orb)), nd):
            s = sum(sizes_orig[k] for k in D)
            target = s if a.mode == "same" else s - 1
            if target <= 0:
                continue
            for comp in compositions(target, list(by_size), a.max_new):
                if len(comp) < a.min_new:
                    continue
                blocks = [(sz, comp.count(sz)) for sz in sorted(set(comp), reverse=True)]
                per = [list(itertools.combinations_with_replacement(by_size[sz], cnt)) for sz, cnt in blocks]
                for choice in itertools.product(*per):
                    combo = tuple(H for blk in choice for H in blk)
                    instances.append((D, sorted(comp, reverse=True), combo))
    instances.sort(key=lambda x: (len(x[1]), -sum(sizes_orig[k] for k in x[0])))
    print("total instances %d" % len(instances), flush=True)
    if a.count_only:
        return
    mine = instances[a.offset::a.stride]
    print("this worker %d" % len(mine), flush=True)
    creps = EO.coord_orbit_reps(G, n) if a.enc == "opt" else None
    Solver = {"cadical153": Cadical153, "cadical195": Cadical195}[a.solver]
    L = 3 * n * n

    def bits(t):
        N = n * n
        return [(t[0] >> i) & 1 for i in range(N)] + [(t[1] >> i) & 1 for i in range(N)] + [(t[2] >> i) & 1 for i in range(N)]
    logpath = a.log or ("../results/%s_%d_%s_w%d.jsonl" % (a.tag, n, a.mode, a.offset))
    log = open(logpath, "a")
    Sset = set(S)

    for (D, comp, combo) in mine:
        kept = [t for k, (_, o, _) in enumerate(orb) if k not in D for t in o]
        R = residual(kept, n)
        t0 = time.time()
        if a.enc == "opt":
            clauses, reps, terms, nv, neq = EO.encode(n, G, list(combo), R, True, True, coord_reps=creps)
        else:
            clauses, reps, terms, nv, neq = EO.encode(n, G, list(combo), R, False, False)
        nxt = nv
        if a.mode == "same" and not a.no_block_set:
            # set-level block: the deleted set D is a union of full G-orbits, so the refill reproduces it
            # as a set iff every refill representative lies in D. Block: OR_i (rep_i not in D).
            deleted = [t for k, (_, o, _) in enumerate(orb) if k in D for t in o]
            dbits = [bits(t) for t in deleted]
            d_lits = []
            for v in reps:
                es = []
                for tb in dbits:
                    nxt += 1; e = nxt; es.append(e)
                    lits = [v[i] if tb[i] else -v[i] for i in range(L)]
                    for lit in lits: clauses.append([-e, lit])
                    clauses.append([e] + [-lit for lit in lits])
                nxt += 1; d = nxt; d_lits.append(d)
                clauses.append([-d] + es)
                for e in es: clauses.append([-e, d])
            clauses.append([-d for d in d_lits])
        nc = len(clauses)
        solver = Solver(bootstrap_with=clauses)
        rec = {"n": n, "group": a.group, "mode": a.mode, "deleted_orbits": list(D),
               "deleted_sizes": [sizes_orig[k] for k in D], "refill_sizes": comp,
               "refill_subgroup_orders": [len(H) for H in combo], "enc": a.enc, "solver": a.solver,
               "block_set": (a.mode == "same" and not a.no_block_set),
               "vars": nv, "clauses": nc, "eqs": neq, "results": []}
        found = 0
        for it in range(a.distinct if a.mode == "same" else 1):
            solver.conf_budget(a.conf)
            ok = solver.solve_limited()
            dt = time.time() - t0
            if ok is None:
                rec["results"].append({"outcome": "budget", "t": round(dt, 1)})
                break
            if ok is False:
                rec["results"].append({"outcome": "unsat", "t": round(dt, 1)})
                break
            model = set(l for l in solver.get_model() if l > 0)
            new_terms = decode(model, terms, n)
            sch = kept + new_terms
            okv = verify(sch, n, n, n)
            same = set(sch) == Sset
            rec["results"].append({"outcome": "sat", "t": round(dt, 1), "rank": len(sch),
                                   "verified": okv, "identical": same})
            if okv and not same:
                found += 1
                fn = "../results/%s_%d_%s_rank%d_D%s_%d.json" % (
                    a.tag, n, a.mode, len(sch), "-".join(str(x) for x in D), found)
                json.dump({"format": [n, n, n], "rank": len(sch), "field": "F2",
                           "verified_brent_f2": True,
                           "found_by": "orbit_lns_n %s n=%d group=%s D=%s refill=%s" % (
                               a.mode, n, a.group, list(D), comp),
                           "scheme_bitmasks": [list(t) for t in sch]}, open(fn, "w"), indent=1)
                rec["results"][-1]["file"] = fn
                print("  *** NEW rank %d VERIFIED -> %s" % (len(sch), fn), flush=True)
                if a.mode == "reduce":
                    break
            # block this assignment AND, when the decode is a new verified scheme, block its term SET
            # (all representative choices of it), so later solutions are genuinely different schemes
            # rather than relabelings of this one (correction of 09-05: Result 6 counted relabelings)
            solver.add_clause([-v if v in model else v for rv in reps for v in rv])
            if okv and not same and not a.no_block_set:
                nb = [bits(t) for t in new_terms]
                d_l = []
                for v in reps:
                    es = []
                    for tb in nb:
                        nxt += 1; e = nxt; es.append(e)
                        lits = [v[i] if tb[i] else -v[i] for i in range(L)]
                        for lit in lits: solver.add_clause([-e, lit])
                        solver.add_clause([e] + [-lit for lit in lits])
                    nxt += 1; d = nxt; d_l.append(d)
                    solver.add_clause([-d] + es)
                    for e in es: solver.add_clause([-e, d])
                solver.add_clause([-d for d in d_l])
        solver.delete()
        outcome = rec["results"][-1]["outcome"] if rec["results"] else "none"
        print("D=%s sizes=%s refill=%s subs=%s: %s (%.1fs, %d vars)" % (
            list(D), rec["deleted_sizes"], comp, rec["refill_subgroup_orders"],
            outcome, time.time() - t0, nv), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
