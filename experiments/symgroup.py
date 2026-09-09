"""Parameterized symmetry machinery for n x n x n F2 matmul schemes (any n).

A scheme term is (a, b, c) as bitmasks over n^2 bits each, with the conventions used by
flip_graph.verify: a indexed i*n+j, b indexed j*n+k, c indexed k*n+i.

Group elements are permutations of the L = 3*n^2 bit positions. Generators provided:
  cyc   : (a,b,c) -> (b,c,a)                     [cyclic slot rotation]
  tau   : (a,b,c) -> (b^T, a^T, c^T)             [transpose-swap involution]
  sandwich(X,Y,Z) for permutation matrices X,Y,Z: (a,b,c) -> (X a Y^-1, Y b Z^-1, Z c X^-1)
All are verified to preserve the matmul tensor by construction; `check_generators` re-verifies
each one numerically on the trivial scheme, so a convention error cannot pass silently.
"""
import itertools, json
from flip_graph import verify, trivial_scheme

def L_of(n): return 3 * n * n

def perm_cyc(n):
    """(a,b,c) -> (b,c,a): bit j of block 1 goes to block 0, etc. p[src] = dst."""
    N = n*n; p = [0]*(3*N)
    for i in range(N):
        p[N + i] = i          # b -> a slot
        p[2*N + i] = N + i    # c -> b slot
        p[i] = 2*N + i        # a -> c slot
    return tuple(p)

def perm_tau(n):
    """(a,b,c) -> (b^T, a^T, c^T)."""
    N = n*n; p = [0]*(3*N)
    for i in range(n):
        for j in range(n):
            p[i*n + j]       = N + (j*n + i)   # a -> b^T
            p[N + i*n + j]   = j*n + i         # b -> a^T
            p[2*N + i*n + j] = 2*N + (j*n + i) # c -> c^T
    return tuple(p)

def _permrows(P, n):
    """P given as list of rows (bitmask per row); return pi with P e_j = e_{pi[j]}."""
    pi = [None]*n
    for r in range(n):
        for c in range(n):
            if (P[r] >> c) & 1: pi[c] = r
    return pi

def perm_sandwich(X, Y, Z, n):
    """X,Y,Z permutation matrices as row-bitmask lists. (a,b,c)->(X a Y^-1, Y b Z^-1, Z c X^-1)."""
    N = n*n; px, py, pz = _permrows(X, n), _permrows(Y, n), _permrows(Z, n)
    p = [0]*(3*N)
    for i in range(n):
        for j in range(n):
            p[i*n + j]       = px[i]*n + py[j]
            p[N + i*n + j]   = N + py[i]*n + pz[j]
            p[2*N + i*n + j] = 2*N + pz[i]*n + px[j]
    return tuple(p)

def compose(p, q): return tuple(p[q[i]] for i in range(len(p)))

def closure(gens, n):
    ident = tuple(range(L_of(n))); G = {ident}; frontier = [ident]
    while frontier:
        g = frontier.pop()
        for h in gens:
            k = compose(h, g)
            if k not in G: G.add(k); frontier.append(k)
    return sorted(G)

def apply(p, term, n):
    N = n*n; a, b, c = term
    src = [(a >> i) & 1 for i in range(N)] + [(b >> i) & 1 for i in range(N)] + [(c >> i) & 1 for i in range(N)]
    dst = [0]*(3*N)
    for i in range(3*N):
        if src[i]: dst[p[i]] = 1
    return (sum(dst[i] << i for i in range(N)),
            sum(dst[N+i] << i for i in range(N)),
            sum(dst[2*N+i] << i for i in range(N)))

def inverse_index(g):
    inv = [0]*len(g)
    for i in range(len(g)): inv[g[i]] = i
    return inv

def check_generators(n, gens, names):
    """A generator must map the trivial scheme to a valid scheme (necessary + strong check)."""
    triv = trivial_scheme(n, n, n); ok = {}
    for g, nm in zip(gens, names):
        img = [apply(g, t, n) for t in triv]
        ok[nm] = (len(set(img)) == len(img)) and verify(img, n, n, n)
    return ok

def orbits(scheme, G, n):
    """Return list of (representative, orbit set, stabilizer subgroup)."""
    seen = set(); out = []
    for t in scheme:
        if t in seen: continue
        orb = {apply(g, t, n) for g in G}; seen |= orb
        H = frozenset(g for g in G if apply(g, t, n) == t)
        out.append((t, sorted(orb), H))
    return out

def is_invariant(scheme, G, n):
    S = set(scheme)
    return all(apply(g, t, n) in S for g in G for t in scheme)

if __name__ == "__main__":
    import sys
    for n in (3, 4, 5, 6):
        gens = [perm_cyc(n), perm_tau(n)]
        print(f"n={n}: generator validity {check_generators(n, gens, ['cyc','tau'])}, "
              f"|<cyc,tau>| = {len(closure(gens, n))}")
