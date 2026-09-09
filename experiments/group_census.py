"""Approach B, step 1: census of candidate symmetry groups for n x n x n F2 schemes.

Ambient group Omega = < permutation sandwiches (X,Y,Z) in Sn^3, cyc, tau > acting on the 3n^2 bit
positions. We enumerate candidate subgroups, deduplicate by Omega-conjugacy, and then apply a
ZERO-COST arithmetic filter before any SAT is run:

    a G-invariant scheme of rank r must decompose into orbits whose sizes are indices [G:H] of
    subgroups H <= G, and those sizes must sum to exactly r.

If no multiset of admissible orbit sizes sums to r, that group cannot host a rank-r scheme at all
and is eliminated without solving. Surviving groups are emitted as a worklist ordered by |G|
descending (larger group = fewer unknowns = cheaper search).

Usage:
  python group_census.py --n 4 --rank 46 --max-gens 2 --out ../results/census_n4_r46.json
"""
import argparse, itertools, json, time
import symgroup as SG


def perm_matrices(n):
    """All n x n permutation matrices as row-bitmask lists."""
    out = []
    for p in itertools.permutations(range(n)):
        out.append([1 << p[r] for r in range(n)])
    return out


def ambient_generators(n):
    """Generators of Omega: sandwiches by transpositions/cycles on each axis, plus cyc and tau."""
    P = perm_matrices(n)
    ident = P[0] if all((P[0][r] >> r) & 1 for r in range(n)) else None
    I = [1 << r for r in range(n)]
    gens = [SG.perm_cyc(n), SG.perm_tau(n)]
    for M in P:
        if M == I:
            continue
        gens.append(SG.perm_sandwich(M, I, I, n))
        gens.append(SG.perm_sandwich(I, M, I, n))
        gens.append(SG.perm_sandwich(I, I, M, n))
    # dedupe
    seen = set(); out = []
    for g in gens:
        if g not in seen:
            seen.add(g); out.append(g)
    return out


def admissible_sizes(G, n):
    """Orbit sizes possible under G = indices of its subgroups."""
    ident = tuple(range(SG.L_of(n)))
    subs = {frozenset([ident])}
    for a in G:
        subs.add(frozenset(SG.closure([a], n)))
    for a in G:
        for b in G:
            subs.add(frozenset(SG.closure([a, b], n)))
    return sorted({len(G) // len(H) for H in subs})


def can_sum_to(sizes, r):
    """Can a multiset of `sizes` sum to exactly r? (unbounded coin problem)"""
    reach = [False] * (r + 1); reach[0] = True
    for v in range(1, r + 1):
        for s in sizes:
            if s <= v and reach[v - s]:
                reach[v] = True; break
    return reach[r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rank", type=int, default=46)
    ap.add_argument("--max-gens", type=int, default=2)
    ap.add_argument("--max-order", type=int, default=200, help="skip groups larger than this (search cost)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = a.n
    t0 = time.time()
    gens = ambient_generators(n)
    print("ambient generators: %d" % len(gens), flush=True)
    Omega = SG.closure(gens, n)
    print("|Omega| = %d  (%.1fs)" % (len(Omega), time.time() - t0), flush=True)

    # candidate subgroups: cyclic, then 2-generated
    cands = {}
    for g in Omega:
        H = frozenset(SG.closure([g], n))
        cands.setdefault(len(H), set()).add(H)
    print("cyclic subgroups by order: %s (%.1fs)" % (
        {k: len(v) for k, v in sorted(cands.items())}, time.time() - t0), flush=True)
    if a.max_gens >= 2:
        cyclic = [H for v in cands.values() for H in v]
        reps = sorted(cyclic, key=len)[-400:]  # pair up the largest cyclic pieces
        pairs = 0
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                gi = next(iter(reps[i] - {tuple(range(SG.L_of(n)))}), None)
                gj = next(iter(reps[j] - {tuple(range(SG.L_of(n)))}), None)
                if gi is None or gj is None: continue
                H = frozenset(SG.closure([gi, gj], n))
                if len(H) <= a.max_order:
                    cands.setdefault(len(H), set()).add(H); pairs += 1
        print("2-generated added (%d combos scanned) (%.1fs)" % (pairs, time.time() - t0), flush=True)

    # conjugacy dedup + arithmetic filter
    def conj(H, g):
        gi = tuple(SG.inverse_index(g))
        return frozenset(SG.compose(SG.compose(g, h), gi) for h in H)
    all_subs = [H for v in cands.values() for H in v if 1 < len(H) <= a.max_order]
    classes = []
    for H in sorted(all_subs, key=len, reverse=True):
        if any(H in cls for cls in classes): continue
        classes.append({conj(H, g) for g in Omega})
    print("conjugacy classes of candidate groups: %d (%.1fs)" % (len(classes), time.time() - t0), flush=True)

    worklist = []
    for cls in classes:
        H = min(cls, key=lambda x: sorted(x))
        Hs = sorted(H)
        sizes = admissible_sizes(Hs, n)
        ok = can_sum_to(sizes, a.rank)
        worklist.append({"order": len(Hs), "admissible_orbit_sizes": sizes,
                         "can_reach_rank": ok, "class_size": len(cls),
                         "elements": [list(g) for g in Hs]})   # representative subgroup, for census_run.py
    worklist.sort(key=lambda w: (-w["order"],))
    feas = [w for w in worklist if w["can_reach_rank"]]
    print("\ngroups: %d classes; %d can arithmetically reach rank %d, %d eliminated for free"
          % (len(worklist), len(feas), a.rank, len(worklist) - len(feas)))
    print("by order (feasible only):", {w["order"]: 1 for w in feas})
    if a.out:
        json.dump({"n": n, "rank": a.rank, "omega_order": len(Omega),
                   "classes": len(worklist), "feasible": len(feas), "worklist": worklist},
                  open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
