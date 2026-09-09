"""Phase 1 step 2: characterise the 5x5 rank-93 pool. Dedupe by term set; per distinct scheme:
verification, C3-invariance, tau-invariance, factor-rank signature (GL-invariant), weight histogram
(permutation-invariant), flip-isolation (shared factors per axis), Hamming distance to the record,
and the exact Z/4 Hensel lift test (linear system over F2). Output: results/P1_pool.json + summary."""
import glob, json, collections, itertools, sys
import numpy as np
import symgroup as SG
from flip_graph import verify, target_tensor

n = 5; N = n * n
def f2rank(m):
    rows = [(m >> (n * i)) & ((1 << n) - 1) for i in range(n)]; r = 0
    while rows:
        p = rows.pop()
        if not p: continue
        r += 1; hb = p.bit_length() - 1; rows = [x ^ p if (x >> hb) & 1 else x for x in rows if x]
    return r
def signature(S): return [dict(sorted(collections.Counter(f2rank(t[ax]) for t in S).items())) for ax in range(3)]
def weights(S): return dict(sorted(collections.Counter(bin(t[0]).count("1") + bin(t[1]).count("1") + bin(t[2]).count("1") for t in S).items()))
def shared(S): return [sum(v for v in collections.Counter(t[ax] for t in S).values() if v > 1) for ax in range(3)]

def z4_lift(S):
    """Exact first Hensel level. A scheme over F2 lifts to Z/4 term-by-term iff there are corrections
    a' = a + 2x, b' = b + 2y, c' = c + 2z (x,y,z in {0,1}) such that the Brent equations hold mod 4.
    Mod 4 the products expand to (sum over terms of abc) + 2*(sum of x b c + a y c + a b z) mod 4 with the
    first sum known: E = (T - sum abc)/2 mod 2 must equal the F2-linear expression in (x,y,z).
    Returns (liftable, rank, unknowns, equations)."""
    T = set(target_tensor(n, n, n))
    A = [[(t[0] >> i) & 1 for i in range(N)] for t in S]; B = [[(t[1] >> i) & 1 for i in range(N)] for t in S]; C = [[(t[2] >> i) & 1 for i in range(N)] for t in S]
    r = len(S); nvar = 3 * r * N
    rows = []; rhs = []
    for ia in range(N):
        for ib in range(N):
            for ic in range(N):
                s = sum(A[t][ia] * B[t][ib] * C[t][ic] for t in range(r))
                target = 1 if (ia, ib, ic) in T else 0
                assert (s - target) % 2 == 0
                e = ((target - s) // 2) % 2          # need 2*(linear part) == target - s  (mod 4)
                row = 0
                for t in range(r):
                    if B[t][ib] and C[t][ic]: row |= 1 << (t * N + ia)                       # x_t[ia]
                    if A[t][ia] and C[t][ic]: row |= 1 << (r * N + t * N + ib)               # y_t[ib]
                    if A[t][ia] and B[t][ib]: row |= 1 << (2 * r * N + t * N + ic)           # z_t[ic]
                if row or e:
                    rows.append(row); rhs.append(e)
    # Gaussian elimination over F2 on augmented rows (bitmask | rhs in top bit)
    aug = [(row << 1) | e for row, e in zip(rows, rhs)]
    rank = 0; consistent = True
    piv_rows = []
    for a_ in aug:
        x = a_
        for pr in piv_rows:
            if (x >> (pr.bit_length() - 1)) & 1: x ^= pr
        if x == 0: continue
        if x == 1:            # 0 = 1
            consistent = False; break
        piv_rows.append(x); rank += 1
        # keep pivot rows reduced against each other lazily (not needed for consistency)
    return consistent, rank, nvar, len(rows)

if __name__ == "__main__":
    rec = [tuple(t) for t in json.load(open("../results/records/lille_555_rank93.json"))["scheme_bitmasks"]]
    files = sorted(set(glob.glob("../results/orbitn_5_same_rank9[0-3]_*.json") + glob.glob("../results/P1_*_rank9[0-3]_*.json")))
    G3 = SG.closure([SG.perm_cyc(n)], n); tau = SG.perm_tau(n)
    sets = collections.OrderedDict(); sets[frozenset(rec)] = ["RECORD lille_555_rank93.json"]
    for f in files:
        S = frozenset(tuple(t) for t in json.load(open(f))["scheme_bitmasks"]); sets.setdefault(S, []).append(f.split("/")[-1])
    out = []
    for i, (S, fs) in enumerate(sets.items()):
        L = sorted(S)
        ok = verify(L, n, n, n); inv3 = SG.is_invariant(L, G3, n); invt = SG.is_invariant(L, [tau], n)
        lift, rk, nv, ne = z4_lift(L)
        row = {"id": i, "files": fs, "rank": len(L), "verified": ok, "C3": inv3, "tau": invt, "signature": signature(L), "weights": weights(L),
               "shared": shared(L), "common_with_record": len(S & frozenset(rec)), "z4_lift": lift, "z4_rank": rk, "z4_unknowns": nv, "z4_equations": ne}
        out.append(row); print(json.dumps({k: v for k, v in row.items() if k not in ("signature", "weights")}), flush=True)
    json.dump(out, open("../results/P1_pool.json", "w"), indent=1)
    sigs = collections.Counter(json.dumps(r["signature"]) for r in out)
    print("distinct schemes %d; distinct GL signatures %d; Z/4-liftable %d; tau-invariant %d; shared-factor range %s" % (
        len(out), len(sigs), sum(r["z4_lift"] for r in out), sum(r["tau"] for r in out), sorted({tuple(r["shared"]) for r in out})))
