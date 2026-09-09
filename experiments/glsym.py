"""General-linear symmetric search for 4x4 (n x n) matrix multiplication schemes over F2.

The tensor's automorphisms include ALL sandwiches (a,b,c) -> (X a Y^-1, Y b Z^-1, Z c X^-1) with
X,Y,Z in GL(n,2), not only permutation matrices. Every earlier census used permutation matrices, whose
action permutes the bit positions. Here the action is linear: each group element g acts on the 3n^2-bit
term vector by a 3n^2 x 3n^2 matrix L_g over F2.

Encoding of a G-invariant scheme with slots (H_1..H_m):
  representative t_i lies in Fix(H_i) = ker(L_h - I : h in H_i); parametrise t_i = B_i w_i with B_i a
  basis of Fix(H_i) and w_i free bits; every coset image x.t_i = L_x B_i w_i is an XOR of the w_i bits
  (Tseitin XOR chains); Brent equations at every coordinate as before (AND3 gates, XOR chains).
Verification: every SAT decode is host-verified by flip_graph.verify; --selftest asserts the 47 under
the S3 part of its group (a permutation group, so it must reproduce the earlier completeness result)
and Strassen under a GL(2,2)^3 cyclic symmetry found by brute force.
"""
import argparse, itertools, json, time, collections
from pysat.solvers import Cadical195
from flip_graph import verify, target_tensor


# ---------------------------------------------------------------- F2 matrices as row bitmasks
def mmul(A, B, n):
    out = []
    for i in range(n):
        r = 0
        for k in range(n):
            if (A[i] >> k) & 1: r ^= B[k]
        out.append(r)
    return tuple(out)

def ident(n): return tuple(1 << i for i in range(n))

def transpose(A, n):
    return tuple(sum(((A[i] >> j) & 1) << i for i in range(n)) for j in range(n))

def minv(A, n):
    """inverse over F2 by Gauss-Jordan; None if singular"""
    rows = [(A[i] << n) | (1 << i) for i in range(n)]   # [A | I] with A in high bits
    for col in range(n):
        piv = next((r for r in range(col, n) if (rows[r] >> (n + col)) & 1), None)
        if piv is None: return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(n):
            if r != col and (rows[r] >> (n + col)) & 1: rows[r] ^= rows[col]
    return tuple(rows[i] & ((1 << n) - 1) for i in range(n))

def mat_order(A, n, limit=100000):
    I = ident(n); P = A; k = 1
    while P != I:
        P = mmul(P, A, n); k += 1
        if k > limit: return None
    return k

def bits_to_mat(m, n): return tuple((m >> (n * i)) & ((1 << n) - 1) for i in range(n))
def mat_to_bits(M, n): return sum(M[i] << (n * i) for i in range(n))


# ---------------------------------------------------------------- sandwich action
def sandwich_term(t, X, Y, Z, n, Yi=None, Zi=None, Xi=None):
    """(a,b,c) -> (X a Y^-1, Y b Z^-1, Z c X^-1)"""
    Yi = Yi or minv(Y, n); Zi = Zi or minv(Z, n); Xi = Xi or minv(X, n)
    a, b, c = (bits_to_mat(t[k], n) for k in range(3))
    return (mat_to_bits(mmul(mmul(X, a, n), Yi, n), n), mat_to_bits(mmul(mmul(Y, b, n), Zi, n), n), mat_to_bits(mmul(mmul(Z, c, n), Xi, n), n))

def lin_matrix(g, n):
    """3n^2 x 3n^2 F2 matrix (as list of column bitmasks) of the sandwich g=(X,Y,Z) on term bits."""
    N = n * n; L = 3 * N; X, Y, Z = g; Yi, Zi, Xi = minv(Y, n), minv(Z, n), minv(X, n)
    cols = []
    for p in range(L):
        t = (1 << p if p < N else 0, 1 << (p - N) if N <= p < 2 * N else 0, 1 << (p - 2 * N) if p >= 2 * N else 0)
        a, b, c = sandwich_term(t, X, Y, Z, n, Yi, Zi, Xi)
        cols.append(a | (b << N) | (c << 2 * N))
    return cols   # image of basis vector p is cols[p] (bitmask over L)

def apply_lin(cols, v):
    out = 0; p = 0
    while v:
        if v & 1: out ^= cols[p]
        v >>= 1; p += 1
    return out

def group_closure(gens, n):
    I = ident(n); G = {(I, I, I)}; frontier = [(I, I, I)]
    while frontier:
        g = frontier.pop()
        for h in gens:
            k = (mmul(h[0], g[0], n), mmul(h[1], g[1], n), mmul(h[2], g[2], n))
            if k not in G: G.add(k); frontier.append(k)
    return sorted(G)

def gmul(g, h, n): return (mmul(g[0], h[0], n), mmul(g[1], h[1], n), mmul(g[2], h[2], n))
def ginv(g, n): return (minv(g[0], n), minv(g[1], n), minv(g[2], n))

def check_group(G, n):
    """every element must map the trivial scheme to a valid scheme (guards conventions)"""
    from flip_graph import trivial_scheme
    triv = trivial_scheme(n, n, n)
    for g in G:
        img = [sandwich_term(t, *g, n) for t in triv]
        if len(set(img)) != len(img) or not verify(img, n, n, n): return False
    return True

def orbits(scheme, G, n):
    seen = set(); out = []
    for t in scheme:
        if t in seen: continue
        orb = {sandwich_term(t, *g, n) for g in G}; seen |= orb
        H = [g for g in G if sandwich_term(t, *g, n) == t]
        out.append((t, sorted(orb), H))
    return out

def subgroups_cyclic(G, n):
    subs = set()
    for g in G: subs.add(frozenset(group_closure([g], n)))
    return subs


# ---------------------------------------------------------------- linear algebra helpers
def nullspace_basis(rows, L):
    """basis of {v : row . v = 0 for all rows} where rows are bitmasks over L bits"""
    piv = {}   # pivot bit -> row
    for r in rows:
        x = r
        for pb, pr in piv.items():
            if (x >> pb) & 1: x ^= pr
        if x:
            pb = x.bit_length() - 1
            # reduce existing rows
            for k in list(piv):
                if (piv[k] >> pb) & 1: piv[k] ^= x
            piv[pb] = x
    free = [b for b in range(L) if b not in piv]
    basis = []
    for f in free:
        v = 1 << f
        for pb, pr in piv.items():
            if (pr >> f) & 1: v |= 1 << pb
        basis.append(v)
    return basis

def fixed_space(H, n):
    """basis of ker(L_h - I) over all h in H, as term-bit vectors"""
    L = 3 * n * n; rows = []
    for h in H:
        cols = lin_matrix(h, n)
        # (L_h - I) v = 0  <=>  for each output bit q: XOR over p with cols[p] bit q of v_p  ==  v_q
        for q in range(L):
            r = sum(((cols[p] >> q) & 1) << p for p in range(L)) ^ (1 << q)
            if r: rows.append(r)
    return nullspace_basis(rows, L)


# ---------------------------------------------------------------- encoder
def encode(n, G, slots, T, units=None):
    """slots: list of stabilizer subgroups (lists of group elements). Returns clauses, params, terms(bits as literal lists), nvars."""
    N = n * n; L = 3 * N
    st = {"nv": 0}; cl = []
    def new():
        st["nv"] += 1; return st["nv"]
    def XOR(lits):
        """Tseitin XOR chain; returns a literal equal to XOR of lits (constant 0 -> a fixed-false var)"""
        if not lits:
            z = new(); cl.append([-z]); return z
        cur = lits[0]
        for l in lits[1:]:
            z = new(); cl.extend([[-z, cur, l], [-z, -cur, -l], [z, -cur, l], [z, cur, -l]]); cur = z
        return cur
    and_cache = {}
    def AND3(x, y, z):
        key = tuple(sorted((x, y, z)))
        w = and_cache.get(key)
        if w is None:
            w = new(); cl.extend([[-w, x], [-w, y], [-w, z], [w, -x, -y, -z]]); and_cache[key] = w
        return w
    params = []; terms = []
    for H in slots:
        B = fixed_space(H, n); k = len(B)
        w = [new() for _ in range(k)]; params.append((B, w))
        # coset representatives of G/H
        Hset = set(H); seen = set(); cosets = []
        for g in G:
            key = frozenset(gmul(g, h, n) for h in H)
            if key in seen: continue
            seen.add(key); cosets.append(g)
        # representative bits (as XOR of params), then nonzero constraints
        rep_bits = []
        for q in range(L):
            rep_bits.append(XOR([w[j] for j in range(k) if (B[j] >> q) & 1]))
        cl.append(rep_bits[0:N]); cl.append(rep_bits[N:2 * N]); cl.append(rep_bits[2 * N:3 * N])
        for x in cosets:
            cols = lin_matrix(x, n)
            # image bit q = XOR over p of cols[p]_q * rep_bit_p  = XOR over params j of (sum_p cols[p]_q B_j_p) w_j
            img = []
            for q in range(L):
                js = [j for j in range(k) if bin(sum(((cols[p] >> q) & 1) for p in range(L) if (B[j] >> p) & 1)).count("1") % 2 == 1]
                img.append(XOR([w[j] for j in js]))
            terms.append(img)
    for ia in range(N):
        for ib in range(N):
            for ic in range(N):
                lits = [AND3(tv[ia], tv[N + ib], tv[2 * N + ic]) for tv in terms]
                z = XOR(lits)
                cl.append([z] if (ia, ib, ic) in T else [-z])
    if units: cl += [[u] for u in units]
    return cl, params, terms, st["nv"]

def decode(model, terms, n):
    N = n * n; out = []
    for tv in terms:
        b = [1 if x in model else 0 for x in tv]
        out.append((sum(b[i] << i for i in range(N)), sum(b[N + i] << i for i in range(N)), sum(b[2 * N + i] << i for i in range(N))))
    return out

def solve(n, G, slots, T, conf=2_000_000, units=None):
    t0 = time.time(); cl, params, terms, nv = encode(n, G, slots, T, units); te = time.time() - t0
    s = Cadical195(bootstrap_with=cl); s.conf_budget(conf); ok = s.solve_limited(); dt = time.time() - t0
    res = {"vars": nv, "clauses": len(cl), "t_encode": round(te, 1), "t": round(dt, 1)}
    if ok is None: res["outcome"] = "budget"
    elif ok is False: res["outcome"] = "unsat"
    else:
        sch = decode(set(l for l in s.get_model() if l > 0), terms, n)
        res["outcome"] = "sat"; res["scheme"] = sch; res["verified"] = verify(sch, n, n, n); res["rank"] = len(set(sch))
    s.delete(); return res


# ---------------------------------------------------------------- self test
def selftest():
    print("== 1. permutation case: the 47 under the S3 part of Gamma (order 6) -- must reproduce the 47")
    S47 = [tuple(t) for t in json.load(open("../results/alphatensor_444_rank47.json"))["scheme_bitmasks"]]
    stab = json.load(open("../results/h1_stabilizer_alphatensor_444_rank47.json"))["elements"]
    gens = [(tuple(X), tuple(Y), tuple(Z)) for X, Y, Z in stab]
    G = group_closure(gens, 4); print("   |G| =", len(G), "group check:", check_group(G, 4))
    orbs = orbits(S47, G, 4); print("   orbit sizes:", sorted(len(o) for _, o, _ in orbs))
    assert all(len(set(o) - set(S47)) == 0 for _, o, _ in orbs)
    T = set(target_tensor(4, 4, 4))
    slots = [H for _, _, H in orbs]
    # units asserting the 47: representative params w such that B w = rep bits -> easier: assert full scheme via terms? we assert rep bits through params by solving B w = t
    cl, params, terms, nv = encode(4, G, slots, T)
    units = []
    for (B, w), (t, _, _) in zip(params, orbs):
        target = t[0] | (t[1] << 16) | (t[2] << 32)
        # solve B w = target over F2 (B columns are basis vectors)
        sol = solve_in_span(B, target)
        assert sol is not None, "representative not in fixed space?!"
        units += [w[j] if (sol >> j) & 1 else -w[j] for j in range(len(w))]
    s = Cadical195(bootstrap_with=cl + [[u] for u in units]); t0 = time.time(); ok = s.solve_limited(); dt = time.time() - t0
    sch = decode(set(l for l in s.get_model() if l > 0), terms, 4) if ok else None; s.delete()
    print("   %d vars, %d clauses; asserted 47 -> %s in %.1fs; decodes to the 47: %s; verifies: %s" % (nv, len(cl), ok, dt, sch is not None and sorted(set(sch)) == sorted(S47), sch is not None and verify(sch, 4, 4, 4)))
    assert ok and sorted(set(sch)) == sorted(S47)
    print("== 2. GL(2,2)^3 symmetries of Strassen (brute force) and from-scratch search under one of them")
    Str = [tuple(t) for t in json.load(open("../results/gpu_222_rank7.json"))["scheme_bitmasks"]]
    GL2 = [M for M in itertools.product(range(1, 4), repeat=2) if minv(M, 2) is not None]
    stabS = [(X, Y, Z) for X in GL2 for Y in GL2 for Z in GL2 if set(sandwich_term(t, X, Y, Z, 2) for t in Str) == set(Str)]
    print("   Strassen scheme (rank %d) sandwich stabilizer in GL(2,2)^3: order %d" % (len(Str), len(stabS)))
    T2 = set(target_tensor(2, 2, 2))
    # pick a cyclic subgroup of order 3 with a non-permutation component if any
    cyc3 = [g for g in stabS if mat_order(g[0], 2) == 3 or mat_order(g[1], 2) == 3 or mat_order(g[2], 2) == 3]
    g = cyc3[0]; G3 = group_closure([g], 2); orbs = orbits(Str, G3, 2)
    print("   chosen order-3 element %s; Strassen orbit sizes %s" % (g, sorted(len(o) for _, o, _ in orbs)))
    res = solve(2, G3, [H for _, _, H in orbs], T2)
    print("   from-scratch search with Strassen's slot types: %s, rank %s, verified %s, %.1fs" % (res["outcome"], res.get("rank"), res.get("verified"), res["t"]))
    assert res["outcome"] == "sat" and res["verified"]
    print("selftest OK")

def solve_in_span(B, target):
    """find bits x with XOR_j x_j B_j = target; None if impossible"""
    piv = {}; comb = {}
    for j, v in enumerate(B):
        x = v; c = 1 << j
        for pb in sorted(piv, reverse=True):
            if (x >> pb) & 1: x ^= piv[pb]; c ^= comb[pb]
        if x:
            pb = x.bit_length() - 1; piv[pb] = x; comb[pb] = c
    x = target; c = 0
    for pb in sorted(piv, reverse=True):
        if (x >> pb) & 1: x ^= piv[pb]; c ^= comb[pb]
    return c if x == 0 else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
