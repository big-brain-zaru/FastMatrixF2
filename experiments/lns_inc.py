"""
Incremental LNS-SAT: build the refill encoding ONCE, then per try only change the residual via
assumptions. Learned clauses about "r rank-1 terms over F2" persist across thousands of tries.

Encoding: r unknown terms (optionally tau-paired), Tseitin ANDs, and for every tensor entry
  XOR(contributions) XOR rhs_e = 0   where rhs_e is a free variable set by ASSUMPTION each try.
So a try = choose deletion set -> compute residual -> assumptions [rhs_e literals] -> solve.

Usage: python lns_inc.py --scheme ../results/z2_444_rank55.json --fmt 4 --k 14 --symmetric --fixed 1 --tries 5000 --conf 400000
"""
import argparse, json, random, time, sys, itertools
from pysat.solvers import Cadical153
from flip_graph import verify, target_tensor
from lns_sat import residual, tr

def build(r, n, symmetric, n_fixed):
    la = lb = lc = n*n
    nv = 0; clauses = []
    def new():
        nonlocal nv; nv += 1; return nv
    def AND(x, y):
        z = new(); clauses.extend([[-z, x], [-z, y], [z, -x, -y]]); return z
    A = [[new() for _ in range(la)] for _ in range(r)]
    B = [[new() for _ in range(lb)] for _ in range(r)]
    C = [[new() for _ in range(lc)] for _ in range(r)]
    for t in range(r):
        clauses.append(A[t][:]); clauses.append(B[t][:]); clauses.append(C[t][:])
    if symmetric:
        for t in range(0, r - n_fixed, 2):
            for i in range(n):
                for j in range(n):
                    for x, y in ((A[t+1][j*n+i], B[t][i*n+j]), (B[t+1][j*n+i], A[t][i*n+j]), (C[t+1][j*n+i], C[t][i*n+j])):
                        clauses.extend([[-x, y], [x, -y]])
        for t in range(r - n_fixed, r):
            for i in range(n):
                for j in range(n):
                    for x, y in ((B[t][j*n+i], A[t][i*n+j]), (C[t][j*n+i], C[t][i*n+j])):
                        clauses.extend([[-x, y], [x, -y]])
    AB = [[[AND(A[t][ia], B[t][ib]) for ib in range(lb)] for ia in range(la)] for t in range(r)]
    RHS = {}
    for ia in range(la):
        for ib in range(lb):
            for ic in range(lc):
                lits = [AND(AB[t][ia][ib], C[t][ic]) for t in range(r)]
                rv = new(); RHS[(ia, ib, ic)] = rv
                lits.append(rv)                      # XOR(contribs) XOR rhs == 0
                cur = lits[0]
                for l in lits[1:]:
                    z = new(); clauses.extend([[-z, cur, l], [-z, -cur, -l], [z, -cur, l], [z, cur, -l]]); cur = z
                clauses.append([-cur])
    return clauses, A, B, C, RHS, nv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default=None); ap.add_argument("--pool", default=None)
    ap.add_argument("--fmt", type=int, default=4)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--refill", type=int, default=None)
    ap.add_argument("--symmetric", action="store_true"); ap.add_argument("--fixed", type=int, default=None)
    ap.add_argument("--tries", type=int, default=2000); ap.add_argument("--conf", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = a.fmt; rng = random.Random(a.seed)
    if a.pool:
        POOL = [[tuple(t) for t in s] for s in json.load(open(a.pool))["schemes"]]
    else:
        POOL = [[tuple(t) for t in json.load(open(a.scheme))["scheme_bitmasks"]]]
    r = a.refill if a.refill is not None else a.k - 1
    n_fixed = a.fixed if a.fixed is not None else (r % 2)
    t0 = time.time()
    clauses, A, B, C, RHS, nv = build(r, n, a.symmetric, n_fixed)
    s = Cadical153(bootstrap_with=clauses)
    print(f"built once: r={r} symmetric={a.symmetric} fixed={n_fixed} vars={nv} clauses={len(clauses)} ({time.time()-t0:.1f}s)", flush=True)
    stats = {"sat": 0, "unsat": 0, "budget": 0}; T = set(target_tensor(n, n, n))
    for it in range(a.tries):
        S = POOL[it % len(POOL)]
        # deletion set (tau-closed if symmetric)
        idx = list(range(len(S)))
        if a.symmetric:
            part = {i: next(j for j, u in enumerate(S) if u == (tr(t[1], n), tr(t[0], n), tr(t[2], n))) for i, t in enumerate(S)}
            chosen = set(); order = idx[:]; rng.shuffle(order)
            for i in order:
                if len(chosen) >= a.k: break
                chosen.add(i); chosen.add(part[i])
            rem = sorted(chosen)
        else:
            rem = sorted(rng.sample(idx, a.k))
        kept = [t for i, t in enumerate(S) if i not in rem]
        R = residual(kept, n, n, n)
        assumptions = [RHS[e] if e in R else -RHS[e] for e in RHS]
        s.conf_budget(a.conf)
        t1 = time.time(); ok = s.solve_limited(assumptions=assumptions); dt = time.time() - t1
        if ok:
            model = set(l for l in s.get_model() if l > 0)
            terms = [(sum(1 << i for i in range(n*n) if A[t][i] in model), sum(1 << i for i in range(n*n) if B[t][i] in model),
                      sum(1 << i for i in range(n*n) if C[t][i] in model)) for t in range(r)]
            new_s = kept + terms; okv = verify(new_s, n, n, n); stats["sat"] += 1
            print(f"  try {it}: removed {len(rem)} -> refilled {r}: rank {len(new_s)} VERIFIED={okv} ({dt:.1f}s)", flush=True)
            if okv:
                fn = a.out or f"../results/lns_inc_rank{len(new_s)}_s{a.seed}.json"
                json.dump({"format":[n,n,n],"rank":len(new_s),"field":"F2","verified_brent_f2":True,"found_by":"lns_inc",
                           "scheme_bitmasks":[list(t) for t in new_s]}, open(fn, "w"), indent=1)
                print("SAVED", fn, flush=True); sys.exit(0)
        else:
            kind = "unsat" if ok is False else "budget"; stats[kind] += 1
            if it % 20 == 0: print(f"  try {it}: {kind} ({dt:.1f}s)  cum={stats}", flush=True)
    print(json.dumps(stats), flush=True); sys.exit(2)

if __name__ == "__main__":
    main()
