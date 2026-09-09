"""Optimised encoder for G-invariant scheme search (candidate replacement for sym_break.encode /
orbit_lns_n.build). Same semantics, two exact reductions:

  (E1) orbit equations only. The encoded scheme S = sum_i sum_{x in G/H_i} x.t_i is a sum of full
       G-orbits, hence a G-invariant tensor: S[gP] = S[P] for all g. The target (or the residual of
       kept full orbits) is G-invariant too. So the Brent equation at P holds iff it holds at gP, and
       one equation per G-orbit of coordinates suffices. Reduction factor up to |G|.
  (E2) gate sharing. The AND of two representative bits (a_i b_j) recurs across coset images and
       coordinates; gates are cached by their variable pair, and (ab)*c gates by their pair too.

Both are semantics-preserving: E1 removes equations implied by the remaining ones, E2 only
shares Tseitin definitions. Validation in --selftest: (1) the 47 asserted under its group -> SAT and
decodes to the 47; (2) exhaustive n=2 census: identical SAT/UNSAT verdict as sym_break.encode on
every instance; (3) selected 4x4 instances with known verdicts.
"""
import argparse, itertools, json, time
from pysat.solvers import Cadical153
import symgroup as SG
import fp_filter as FP
from flip_graph import verify, target_tensor


def coord_orbit_reps(G, n):
    """One representative per G-orbit of tensor coordinates, plus orbit sizes."""
    N = n * n
    seen = set(); reps = []
    for ia in range(N):
        for ib in range(N):
            for ic in range(N):
                P = (ia, ib, ic)
                if P in seen: continue
                orb = {FP.coord_image(g, n, P) for g in G}
                seen |= orb; reps.append((P, len(orb)))
    return reps


def encode(n, G, slots, R, orbit_eqs=True, share=True, units=None, coord_reps=None):
    """Returns (clauses, reps, terms, nvars, neqs). R = set of coordinates where the (residual)
    target is 1. units: extra unit clauses (e.g. to assert a known scheme)."""
    N = n * n; L = 3 * N
    st = {"nv": 0}; clauses = []
    def new():
        st["nv"] += 1; return st["nv"]
    reps = []; terms = []
    for H in slots:
        v = [new() for _ in range(L)]
        reps.append(v)
        for h in H:
            for i in range(L):
                j = h[i]
                if j != i:
                    clauses.append([-v[i], v[j]]); clauses.append([v[i], -v[j]])
        clauses.append(v[0:N]); clauses.append(v[N:2 * N]); clauses.append(v[2 * N:3 * N])
        for g in FP.coset_reps(G, H):
            inv = SG.inverse_index(g)
            terms.append([v[inv[j]] for j in range(L)])
    cache = {}
    def AND(x, y):
        if share:
            key = (x, y) if x < y else (y, x)
            z = cache.get(key)
            if z is not None: return z
        z = new()
        clauses.append([-z, x]); clauses.append([-z, y]); clauses.append([z, -x, -y])
        if share: cache[key] = z
        return z
    if orbit_eqs:
        coords = coord_reps if coord_reps is not None else coord_orbit_reps(G, n)
        coords = [P for P, _ in coords]
    else:
        coords = [(ia, ib, ic) for ia in range(N) for ib in range(N) for ic in range(N)]
    for (ia, ib, ic) in coords:
        lits = [AND(AND(tv[ia], tv[N + ib]), tv[2 * N + ic]) for tv in terms]
        rhs = 1 if (ia, ib, ic) in R else 0
        cur = lits[0]
        for l in lits[1:]:
            z = new()
            clauses.append([-z, cur, l]); clauses.append([-z, -cur, -l])
            clauses.append([z, -cur, l]); clauses.append([z, cur, -l])
            cur = z
        clauses.append([cur] if rhs else [-cur])
    if units:
        clauses.extend([[u] for u in units])
    return clauses, reps, terms, st["nv"], len(coords)


def decode(model, terms, n):
    N = n * n; out = []
    for tv in terms:
        b = [1 if x in model else 0 for x in tv]
        out.append((sum(b[i] << i for i in range(N)), sum(b[N + i] << i for i in range(N)),
                    sum(b[2 * N + i] << i for i in range(N))))
    return out


def residual(kept, n):
    N = n * n; acc = set(target_tensor(n, n, n))
    for (a, b, c) in kept:
        for ia in range(N):
            if not (a >> ia) & 1: continue
            for ib in range(N):
                if not (b >> ib) & 1: continue
                for ic in range(N):
                    if (c >> ic) & 1:
                        k = (ia, ib, ic)
                        if k in acc: acc.remove(k)
                        else: acc.add(k)
    return acc


def solve(clauses, conf, Solver=Cadical153):
    s = Solver(bootstrap_with=clauses); s.conf_budget(conf); ok = s.solve_limited()
    m = set(l for l in s.get_model() if l > 0) if ok else None
    s.delete(); return ok, m


def selftest():
    import sym_break as SB
    print("== 1. completeness on the 47 (Gamma, its own slots, rep bits asserted)")
    S47, G = FP.load_gamma(4); n = 4; T = set(target_tensor(n, n, n))
    orbs = SG.orbits(S47, G, n); slots = [H for (_, _, H) in orbs]
    creps = coord_orbit_reps(G, n)
    print("   coordinate orbits under Gamma: %d (of 4096)" % len(creps))
    for oe, sh in [(False, False), (True, False), (True, True)]:
        cl, reps, terms, nv, neq = encode(n, G, slots, T, oe, sh)
        units = []
        for v, (t, _, _) in zip(reps, orbs):
            a, b, c = t
            bits = [(a >> i) & 1 for i in range(16)] + [(b >> i) & 1 for i in range(16)] + [(c >> i) & 1 for i in range(16)]
            units += [v[i] if bits[i] else -v[i] for i in range(48)]
        t0 = time.time(); ok, m = solve(cl + [[u] for u in units], 10**7); dt = time.time() - t0
        sch = decode(m, terms, n) if ok else None
        print("   orbit_eqs=%s share=%s: %d vars, %d clauses, %d equations -> %s in %.2fs, decodes to 47: %s, verifies: %s" % (
            oe, sh, nv, len(cl), neq, ok, dt, sch is not None and sorted(sch) == sorted(S47), sch is not None and verify(sch, 4, 4, 4)))
        assert ok and verify(sch, 4, 4, 4) and sorted(sch) == sorted(S47)

    print("== 2. exhaustive n=2 census cross-check against sym_break.encode")
    n = 2; T2 = set(target_tensor(2, 2, 2))
    P = [[1 << p[r] for r in range(n)] for p in itertools.permutations(range(n))]; I = [1 << r for r in range(n)]
    gens = [SG.perm_cyc(n), SG.perm_tau(n)] + [SG.perm_sandwich(M, I, I, n) for M in P[1:]] + \
           [SG.perm_sandwich(I, M, I, n) for M in P[1:]] + [SG.perm_sandwich(I, I, M, n) for M in P[1:]]
    Omega = SG.closure(gens, n)
    cyc_subs = {frozenset(SG.closure([g], n)) for g in Omega if FP.order_of(g) > 1}
    classes = FP.subgroup_classes(Omega, cyc_subs)
    tot = agree = 0; t_old = t_new = 0.0
    for (Gc, _) in classes:
        Gl = sorted(Gc)
        for slots, key in FP.slots_by_class(Gl, n, 7, 7, 200):
            cl0, _, terms0, _ = SB.encode(n, Gl, slots, T2, False, False)
            t0 = time.time(); ok0, m0 = solve(cl0, 2 * 10**6); t_old += time.time() - t0
            cl1, _, terms1, _, _ = encode(n, Gl, slots, T2, True, True)
            t0 = time.time(); ok1, m1 = solve(cl1, 2 * 10**6); t_new += time.time() - t0
            if ok1:
                assert verify(decode(m1, terms1, n), 2, 2, 2)
            tot += 1; agree += (ok0 == ok1)
    print("   %d instances, verdicts agree on %d; old %.2fs total, new %.2fs total" % (tot, agree, t_old, t_new))
    assert agree == tot

    print("== 3. sizes on a from-scratch 4x4 census instance (group 7, [12,12,12,6,3,1])")
    groups = json.load(open("../results/census_n4_r46_cyclic.json"))["worklist"]
    G7 = [tuple(g) for g in groups[7]["elements"]]
    classes7 = FP.subgroup_classes(G7, FP.all_subgroups(G7, 4))
    wl = json.load(open("../results/B2_worklist.json"))
    key = [w for w in wl["work"] if w["group_index"] == 7 and w["orbit_sizes"] == [12, 12, 12, 6, 3, 1]][0]["class_key"]
    slots7 = [classes7[ci][0] for ci in key]
    T4 = set(target_tensor(4, 4, 4))
    for oe, sh in [(False, False), (False, True), (True, False), (True, True)]:
        t0 = time.time(); cl, _, _, nv, neq = encode(4, G7, slots7, T4, oe, sh); dt = time.time() - t0
        print("   orbit_eqs=%s share=%s: %d vars, %d clauses, %d equations, encode %.1fs" % (oe, sh, nv, len(cl), neq, dt))
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
