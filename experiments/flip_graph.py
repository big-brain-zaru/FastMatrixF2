"""
Flip-graph random walk for fast matrix multiplication schemes over F2.
Kauers-Moosbauer style (arXiv:2212.01175), minimal prototype.

A scheme for <n,m,p> is a list of rank-1 terms (a,b,c):
  a in F2^(n*m), b in F2^(m*p), c in F2^(p*n)  -- stored as int bitmasks.
The scheme is valid iff sum_i a_i (x) b_i (x) c_i = T_<n,m,p> (Brent equations over F2).

Flip: two terms sharing factor on one axis, e.g. a_i == a_j:
  (a,b_i,c_i),(a,b_j,c_j) -> (a, b_i, c_i^c_j), (a, b_i^b_j, c_j)
Reduction: any zero factor -> drop term; two terms equal on two axes -> merge.
Plus transition (escape): split one term into two, rank +1.

Usage: python flip_graph.py [--fmt 3 3 3] [--seconds 90] [--target 23]
"""
import argparse, random, time, json, sys

def idx_a(i, j, m): return i * m + j          # a: n x m
def idx_b(j, k, p): return j * p + k          # b: m x p
def idx_c(k, i, n): return k * n + i          # c: p x n

def target_tensor(n, m, p):
    """Set of (a_bit, b_bit, c_bit) with coefficient 1 in T_<n,m,p>."""
    T = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T.add((idx_a(i, j, m), idx_b(j, k, p), idx_c(k, i, n)))
    return T

def trivial_scheme(n, m, p):
    return [(1 << a, 1 << b, 1 << c) for (a, b, c) in sorted(target_tensor(n, m, p))]

def verify(scheme, n, m, p):
    """Exact Brent-equations check over F2. Returns True iff scheme computes <n,m,p>."""
    la, lb, lc = n * m, m * p, p * n
    acc = {}
    for (a, b, c) in scheme:
        for ia in range(la):
            if not (a >> ia) & 1: continue
            for ib in range(lb):
                if not (b >> ib) & 1: continue
                for ic in range(lc):
                    if not (c >> ic) & 1: continue
                    key = (ia, ib, ic)
                    acc[key] = acc.get(key, 0) ^ 1
    ones = {k for k, v in acc.items() if v}
    return ones == target_tensor(n, m, p)

def reduce_scheme(scheme):
    """Apply all available reductions. Returns (scheme, n_removed)."""
    removed = 0
    # drop zero-factor terms
    scheme = [t for t in scheme if t[0] and t[1] and t[2]]
    changed = True
    while changed:
        changed = False
        seen = {}
        for i, (a, b, c) in enumerate(scheme):
            for axes, key in (((0, 1), (0, a, b)), ((0, 2), (1, a, c)), ((1, 2), (2, b, c))):
                if key in seen:
                    j = seen[key]
                    ta, tb, tc = scheme[j]
                    if key[0] == 0:   merged = (a, b, tc ^ c)
                    elif key[0] == 1: merged = (a, tb ^ b, c)
                    else:             merged = (ta ^ a, b, c)
                    scheme[j] = merged
                    del scheme[i]
                    removed += 1
                    if not (merged[0] and merged[1] and merged[2]):
                        scheme.remove(merged)
                        removed += 1
                    changed = True
                    break
            if changed: break
            seen[(0, a, b)] = i; seen[(1, a, c)] = i; seen[(2, b, c)] = i
    return scheme, removed

def flip_walk(n, m, p, seconds, target, seed, plus_after=50000, max_rank_slack=3):
    rng = random.Random(seed)
    scheme = trivial_scheme(n, m, p)
    best = len(scheme)
    best_scheme = list(scheme)
    start_rank = len(scheme)
    t0 = time.time()
    flips = 0
    since_reduction = 0
    while time.time() - t0 < seconds:
        for _ in range(2000):  # inner batch to avoid clock overhead
            flips += 1
            since_reduction += 1
            axis = rng.randrange(3)
            # group terms by the chosen axis factor
            groups = {}
            for i, t in enumerate(scheme):
                groups.setdefault(t[axis], []).append(i)
            cands = [g for g in groups.values() if len(g) >= 2]
            if not cands:
                since_reduction = plus_after + 1
            else:
                i, j = rng.sample(rng.choice(cands), 2)
                a1, b1, c1 = scheme[i]; a2, b2, c2 = scheme[j]
                if axis == 0:   scheme[i] = (a1, b1, c1 ^ c2); scheme[j] = (a1, b1 ^ b2, c2)
                elif axis == 1: scheme[i] = (a1 ^ a2, b1, c1); scheme[j] = (a2, b1, c1 ^ c2)
                else:           scheme[i] = (a1, b1 ^ b2, c1); scheme[j] = (a1 ^ a2, b2, c1)
                if rng.random() < 0.05 or not all(scheme[i]) or not all(scheme[j]):
                    scheme, r = reduce_scheme(scheme)
                    if r:
                        since_reduction = 0
                        if len(scheme) < best:
                            best = len(scheme); best_scheme = list(scheme)
                            if best <= target:
                                return best, best_scheme, flips, time.time() - t0
            # escape: plus transition or restart
            if since_reduction > plus_after:
                if len(scheme) >= best + max_rank_slack or len(scheme) >= start_rank:
                    scheme = trivial_scheme(n, m, p)  # restart
                else:
                    k = rng.randrange(len(scheme))
                    a, b, c = scheme[k]
                    la = n * m
                    mask = rng.randrange(1, 1 << la)
                    if mask != a:
                        scheme[k] = (mask, b, c)
                        scheme.append((mask ^ a, b, c))
                since_reduction = 0
    return best, best_scheme, flips, time.time() - t0

STRASSEN = [  # classic 7-mult scheme mapped to bitmask convention, over F2 (+ = XOR)
    # a: 2x2 row-major (A11,A12,A21,A22) -> bits 0..3 ; b likewise ; c: p x n layout (C11,C21,C12,C22)? see idx_c
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", nargs=3, type=int, default=[2, 2, 2])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--target", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n, m, p = args.fmt

    triv = trivial_scheme(n, m, p)
    assert verify(triv, n, m, p), "trivial scheme failed verification -- verifier broken"
    bad = list(triv); bad[0] = (triv[0][0] ^ 0b10, triv[0][1], triv[0][2])
    corrupted_ok = not verify(bad, n, m, p)
    print(f"verifier sanity: trivial<{n},{m},{p}> PASS, corrupted rejected: {corrupted_ok}")

    best, scheme, flips, dt = flip_walk(n, m, p, args.seconds, args.target, args.seed)
    valid = verify(scheme, n, m, p)
    print(f"<{n},{m},{p}>: naive={n*m*p}  best_found={best}  valid={valid}  "
          f"flips={flips:,} ({flips/dt:,.0f}/s)  time={dt:.1f}s")
    out = {
        "format": [n, m, p], "naive_rank": n * m * p, "best_rank_found": best,
        "verified": valid, "flips": flips, "flips_per_sec": flips / dt,
        "seconds": dt, "seed": args.seed,
        "scheme_bitmasks": [[t[0], t[1], t[2]] for t in scheme],
    }
    fn = f"../results/flip_{n}{m}{p}_seed{args.seed}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)
    if not valid:
        sys.exit(1)

if __name__ == "__main__":
    main()
