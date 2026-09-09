"""H1: exact sandwich-stabilizer of a 4x4 F2 scheme under (a,b,c) -> (X a Y^-1, Y b Z^-1, Z c X^-1).
Pruned search: rank-1 a-factors u v^T must map to rank-1 a-factors; with X fixed, Y^-T is pinned
by the images of 4 independent v's, so each X yields only a handful of Y candidates.
"""
import json, itertools, sys, time
n = 4
def load(p): return [tuple(t) for t in json.load(open(p))["scheme_bitmasks"]]
def rows(x): return [(x >> (r*n)) & 0xF for r in range(n)]
def bits(rw): return sum(r << (i*n) for i, r in enumerate(rw))
def mm(P, Q):
    return [ (lambda r: (lambda v: v)(0) ) and 0 for r in P ] if False else [ _row(r, Q) for r in P ]
def _row(r, Q):
    v = 0
    for j in range(n):
        if (r >> j) & 1: v ^= Q[j]
    return v
def inv(P):
    A = [ (P[i] | (1 << (n+i))) for i in range(n) ]
    for col in range(n):
        piv = next((i for i in range(col, n) if (A[i] >> col) & 1), None)
        if piv is None: return None
        A[col], A[piv] = A[piv], A[col]
        for i in range(n):
            if i != col and (A[i] >> col) & 1: A[i] ^= A[col]
    return [ (A[i] >> n) & 0xF for i in range(n) ]
def transpose(P):
    return [ sum(((P[r] >> c) & 1) << r for r in range(n)) for c in range(n) ]
def matvec(P, v):  # P (rows) times column vector v (4-bit)
    return sum((( bin(P[i] & v).count("1") & 1) << i) for i in range(n))
def rank1_uv(x):
    """if x = u v^T (u,v 4-bit), return (u,v) else None."""
    rw = rows(x); nz = [r for r in rw if r]
    if not nz or any(r != nz[0] for r in nz): return None
    v = nz[0]; u = sum((1 << i) for i, r in enumerate(rw) if r)
    return (u, v)
def sandwich(a, X, Yi):  # X a Y^-1
    return bits(mm(mm(X, rows(a)), Yi))
def gl4():
    out = []
    for rw in itertools.product(range(1, 16), repeat=n):
        if inv(list(rw)) is not None: out.append(list(rw))
    return out

def stabilizer(S, verbose=True):
    Sset = set(S); Aset = set(t[0] for t in S); Bset = set(t[1] for t in S); Cset = set(t[2] for t in S)
    r1 = [(a, rank1_uv(a)) for a in Aset]; r1 = [(a, uv) for a, uv in r1 if uv]
    us = {}; vs_by_u = {}
    for a, (u, v) in r1: vs_by_u.setdefault(u, set()).add(v)
    # pick 4 rank-1 factors with independent v's
    basis = []
    for a, (u, v) in r1:
        cand = [b for b in basis] + [(u, v)]
        M = [c[1] for c in cand]
        # independence check over F2
        rk = 0; rr = M[:]
        for bit in range(n):
            piv = next((i for i, x in enumerate(rr) if (x >> bit) & 1), None)
            if piv is None: continue
            p = rr.pop(piv); rk += 1; rr = [x ^ p if (x >> bit) & 1 else x for x in rr]
        if rk == len(cand): basis = cand
        if len(basis) == 4: break
    assert len(basis) == 4, "need 4 rank-1 a-factors with independent v's"
    t0 = time.time(); GL = gl4(); GLinv = {tuple(g): inv(g) for g in GL}
    if verbose: print("GL(4,2):", len(GL), "rank-1 a-factors:", len(r1), flush=True)
    found = []
    for X in GL:
        # for each basis (u_i, v_i): X u_i must equal some u' with rank-1 a-factors; then Y^-T v_i in vs_by_u[u']
        cand_lists = []
        ok = True
        for (u, v) in basis:
            up = matvec(X, u)
            if up not in vs_by_u: ok = False; break
            cand_lists.append(sorted(vs_by_u[up]))
        if not ok: continue
        for choice in itertools.product(*cand_lists):
            # Y^-T maps v_i -> choice_i ; build matrix M with M v_i = w_i
            V = [b[1] for b in basis]; W = list(choice)
            # solve M: M = W_mat * V_mat^-1 where columns are vectors
            Vm = transpose(V)  # rows->columns: matrix whose columns are v_i  (V list are columns)
            Vinv = inv(Vm)
            if Vinv is None: continue
            Wm = transpose(W)
            M = mm(Wm, Vinv)          # M = W V^-1 : M v_i = w_i
            if inv(M) is None: continue
            YiT = M                    # Y^-T = M  => Y^-1 = M^T => Y = (M^T)^-1
            Yi = transpose(YiT)
            Y = inv(Yi)
            if Y is None: continue
            if all(sandwich(a, X, Yi) in Aset for a in Aset):
                found.append((X, Y))
    if verbose: print("(X,Y) preserving a-factor set:", len(found), f"{time.time()-t0:.0f}s", flush=True)
    full = []
    for X, Y in found:
        Xi = GLinv[tuple(X)]; Yi = inv(Y)
        # Z pinned similarly via b-factors: Y b Z^-1 in Bset. Brute-force Z over GL (20160) with early exit.
        for Z in GL:
            Zi = GLinv[tuple(Z)]
            good = True
            for (a, b, c) in S:
                if (sandwich(a, X, Yi), sandwich(b, Y, Zi), sandwich(c, Z, Xi)) not in Sset: good = False; break
            if good: full.append((X, Y, Z))
    return full

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/alphatensor_444_rank47.json"
    S = load(path)
    full = stabilizer(S)
    print("STABILIZER ORDER (GL^3 sandwich part):", len(full))
    json.dump({"scheme": path, "order": len(full), "elements": full[:500]}, open("../results/h1_stabilizer_" + path.split("/")[-1], "w"))
