"""
SAT large-neighbourhood search (LNS) over F2 matrix-multiplication schemes.  (Track B hybrid)

Move that no flip graph can make: take a valid rank-r scheme, delete k terms, compute the
residual tensor R = T - sum(kept), and ask a CDCL SAT solver for a (k-1)-term decomposition
of R. Success => verified rank r-1 scheme. Brent equations are encoded with Tseitin ANDs and
XOR chains (CaDiCaL via python-sat). Optional tau-symmetry (a,b,c)->(bT,aT,cT) constraint
halves the unknowns when the input scheme is tau-symmetric and the deleted set is tau-closed.

Usage:
  python lns_sat.py --scheme ../results/gpu_444_rank49.json --fmt 4 4 4 --k 6 --tries 200 --timeout 60
"""
import argparse, json, random, time, sys, itertools
from pysat.solvers import Cadical153
from flip_graph import verify, target_tensor

def tr(x, n):
    y = 0
    for i in range(n):
        for j in range(n):
            if (x >> (i*n+j)) & 1: y |= 1 << (j*n+i)
    return y

def residual(scheme_kept, n, m, p):
    """Set of (ia,ib,ic) where T - sum(kept) is 1 over F2."""
    la, lb, lc = n*m, m*p, p*n
    acc = set(target_tensor(n, m, p))
    for (a, b, c) in scheme_kept:
        for ia in range(la):
            if not (a >> ia) & 1: continue
            for ib in range(lb):
                if not (b >> ib) & 1: continue
                for ic in range(lc):
                    if (c >> ic) & 1:
                        key = (ia, ib, ic)
                        if key in acc: acc.remove(key)
                        else: acc.add(key)
    return acc

class Enc:
    def __init__(self): self.nv = 0; self.clauses = []
    def new(self): self.nv += 1; return self.nv
    def AND(self, x, y):
        z = self.new(); self.clauses += [[-z, x], [-z, y], [z, -x, -y]]; return z
    def XOR_eq(self, lits, rhs):
        """XOR(lits) == rhs (0/1), via chain of aux vars."""
        if not lits:
            if rhs: self.clauses.append([])   # unsat
            return
        cur = lits[0]
        for l in lits[1:]:
            z = self.new()
            # z = cur xor l
            self.clauses += [[-z, cur, l], [-z, -cur, -l], [z, -cur, l], [z, cur, -l]]
            cur = z
        self.clauses.append([cur] if rhs else [-cur])

def solve_residual(R, r, n, m, p, timeout, symmetric=False, seed=0, conf_budget=300000, n_fixed=None):
    """Find r rank-1 terms over F2 summing to residual R. Returns list of terms or None."""
    la, lb, lc = n*m, m*p, p*n
    E = Enc()
    A = [[E.new() for _ in range(la)] for _ in range(r)]
    B = [[E.new() for _ in range(lb)] for _ in range(r)]
    C = [[E.new() for _ in range(lc)] for _ in range(r)]
    # symmetry breaking: lexicographic-ish ordering is expensive; use cheap nonzero constraints
    for t in range(r):
        E.clauses.append(A[t][:]); E.clauses.append(B[t][:]); E.clauses.append(C[t][:])
    if symmetric and n == m == p:
        # first 2*pairs terms are tau-partner pairs (2t, 2t+1); the last n_fixed terms are fixed points
        if n_fixed is None: n_fixed = r % 2
        assert (r - n_fixed) % 2 == 0 and 0 <= n_fixed <= r
        for t in range(0, r - n_fixed, 2):
            for i in range(n):
                for j in range(n):
                    # A[t+1][j*n+i] == B[t][i*n+j]  (a' = bT) ; B[t+1] = aT ; C[t+1] = cT
                    x, y = A[t+1][j*n+i], B[t][i*n+j]; E.clauses += [[-x, y], [x, -y]]
                    x, y = B[t+1][j*n+i], A[t][i*n+j]; E.clauses += [[-x, y], [x, -y]]
                    x, y = C[t+1][j*n+i], C[t][i*n+j]; E.clauses += [[-x, y], [x, -y]]
        for t in range(r - n_fixed, r):
            for i in range(n):
                for j in range(n):
                    x, y = B[t][j*n+i], A[t][i*n+j]; E.clauses += [[-x, y], [x, -y]]
                    x, y = C[t][j*n+i], C[t][i*n+j]; E.clauses += [[-x, y], [x, -y]]
    AB = [[[E.AND(A[t][ia], B[t][ib]) for ib in range(lb)] for ia in range(la)] for t in range(r)]
    for ia in range(la):
        for ib in range(lb):
            for ic in range(lc):
                lits = [E.AND(AB[t][ia][ib], C[t][ic]) for t in range(r)]
                E.XOR_eq(lits, 1 if (ia, ib, ic) in R else 0)
    s = Cadical153(bootstrap_with=E.clauses)
    s.conf_budget(int(conf_budget))          # CaDiCaL honours conflict budgets, not interrupts
    ok = s.solve_limited()
    if not ok:                               # False = UNSAT, None = budget exhausted
        s.delete(); return "budget" if ok is None else None
    model = set(l for l in s.get_model() if l > 0); s.delete()
    terms = []
    for t in range(r):
        a = sum(1 << i for i in range(la) if A[t][i] in model)
        b = sum(1 << i for i in range(lb) if B[t][i] in model)
        c = sum(1 << i for i in range(lc) if C[t][i] in model)
        terms.append((a, b, c))
    return terms

def overlap(t, u):
    return sum(bin(t[i] & u[i]).count("1") for i in range(3)) + 8 * sum(t[i] == u[i] for i in range(3))

def choose_removal(S, k, rng, n, symmetric, strategy="random"):
    """Pick k terms to delete. 'cluster' = a seed term plus its k-1 most overlapping neighbours."""
    idx = list(range(len(S)))
    if strategy == "cluster" and not symmetric:
        s0 = rng.randrange(len(S))
        order = sorted((i for i in idx if i != s0), key=lambda i: -overlap(S[s0], S[i]) + rng.random())
        return sorted([s0] + order[:k-1])
    if symmetric:
        part = {}
        for i, t in enumerate(S):
            part[i] = next(j for j, u in enumerate(S) if u == (tr(t[1], n), tr(t[0], n), tr(t[2], n)))
        chosen = set()
        order = idx[:]; rng.shuffle(order)
        for i in order:
            if len(chosen) >= k: break
            chosen.add(i); chosen.add(part[i])
        return sorted(chosen)
    return sorted(rng.sample(idx, k))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--fmt", nargs=3, type=int, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tries", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--strategy", default="random", choices=["random", "cluster"])
    ap.add_argument("--same-rank", action="store_true", help="calibration: re-solve with k terms (not k-1)")
    ap.add_argument("--conf", type=int, default=300000, help="CaDiCaL conflict budget per try")
    ap.add_argument("--max-fixed", type=int, default=5, help="symmetric: max fixed points in the refill")
    ap.add_argument("--delete-indices", default=None, help="comma-separated term indices to delete (overrides random choice)")
    ap.add_argument("--refill", type=int, default=None, help="number of refill terms (default k-1)")
    ap.add_argument("--pool", default=None, help="pool json (from a walker run); cycles through its schemes")
    ap.add_argument("--pool-stride", type=int, default=1)
    ap.add_argument("--pool-offset", type=int, default=0)
    a = ap.parse_args()
    n, m, p = a.fmt
    if a.pool:
        POOL = json.load(open(a.pool))["schemes"]
        POOL = [[tuple(t) for t in s_] for s_ in POOL[a.pool_offset::a.pool_stride]]
        print(f"pool: {len(POOL)} schemes (offset {a.pool_offset}, stride {a.pool_stride})", flush=True)
        S = POOL[0]
    else:
        POOL = None
        S = [tuple(t) for t in json.load(open(a.scheme))["scheme_bitmasks"]]
    assert verify(S, n, m, p), "input scheme invalid"
    rng = random.Random(a.seed)
    r0 = len(S)
    print(f"LNS-SAT on rank {r0} <{n},{m},{p}>: delete k={a.k}, solve k-1, timeout {a.timeout}s, symmetric={a.symmetric}", flush=True)
    t0 = time.time(); stats = {"sat": 0, "unsat": 0, "timeout": 0}
    for it in range(a.tries):
        if POOL is not None:
            S = POOL[it % len(POOL)]
            if not verify(S, n, m, p): continue
            r0 = len(S)
        rem = [int(v) for v in a.delete_indices.split(",")] if a.delete_indices else choose_removal(S, a.k, rng, n, a.symmetric, a.strategy)
        kept = [t for i, t in enumerate(S) if i not in rem]
        R = residual(kept, n, m, p)
        k = len(rem)
        t1 = time.time()
        target_r = a.refill if a.refill is not None else (k if a.same_rank else k - 1)
        if a.symmetric:
            # enumerate refill compositions: n_fixed with the parity of target_r, up to --max-fixed
            comps = [f for f in range(target_r % 2, min(target_r, a.max_fixed) + 1, 2)]
            sol = None
            for f in comps:
                sol = solve_residual(R, target_r, n, m, p, a.timeout, symmetric=True, seed=it, conf_budget=a.conf, n_fixed=f)
                if sol is not None and sol != "budget": break
        else:
            sol = solve_residual(R, target_r, n, m, p, a.timeout, symmetric=False, seed=it, conf_budget=a.conf)
        dt = time.time() - t1
        if sol is None or sol == "budget":
            kind = "timeout" if sol == "budget" else "unsat"
            stats[kind] += 1
            print(f"  try {it}: removed {k}, {kind} ({dt:.1f}s)", flush=True)
            continue
        new = kept + sol
        ok = verify(new, n, m, p)
        stats["sat"] += 1
        print(f"  try {it}: removed {k} -> re-solved with {target_r}: rank {len(new)} VERIFIED={ok} ({dt:.1f}s)", flush=True)
        if ok and a.same_rank:
            continue
        if ok:
            out = a.out or (a.scheme or a.pool).replace(f"rank{r0}", f"rank{len(new)}_lns").replace("_pool.json", f"_rank{len(new)}_lns.json")
            json.dump({"format": [n, m, p], "rank": len(new), "field": "F2", "verified_brent_f2": True,
                       "found_by": f"lns_sat from rank {r0}, k={k}", "scheme_bitmasks": [list(t) for t in new]},
                      open(out, "w"), indent=1)
            print(f"SAVED {out}", flush=True)
            sys.exit(0)
    print(json.dumps({"stats": stats, "seconds": time.time() - t0}), flush=True)
    sys.exit(2)

if __name__ == "__main__":
    main()
