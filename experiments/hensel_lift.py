"""Exact Hensel-lift test: does an F2 scheme lift to Z/2^k? Each step is a linear system over F2."""
import json, sys, numpy as np
from flip_graph import target_tensor
n=4; L=16
def load(p): return [tuple(t) for t in json.load(open(p))["scheme_bitmasks"]]
def vec(x): return np.array([(x>>i)&1 for i in range(L)], dtype=np.int64)
def f2_solve(M, rhs):
    """Gaussian elimination over F2 on augmented [M|rhs]; returns (consistent, solution or None, rank)."""
    A = np.concatenate([M % 2, (rhs % 2).reshape(-1,1)], axis=1).astype(np.uint8)
    rows, cols = A.shape; piv_cols=[]; r=0
    for c in range(cols-1):
        p = np.nonzero(A[r:, c])[0]
        if p.size == 0: continue
        p = p[0] + r
        if p != r: A[[r,p]] = A[[p,r]]
        mask = A[:, c].astype(bool); mask[r] = False
        A[mask] ^= A[r]
        piv_cols.append(c); r += 1
        if r == rows: break
    # consistency: any zero row with rhs 1
    zero_rows = ~A[:, :-1].any(axis=1)
    if (A[zero_rows, -1] == 1).any(): return False, None, r
    x = np.zeros(cols-1, dtype=np.uint8)
    for i, c in enumerate(piv_cols): x[c] = A[i, -1]
    return True, x, r
def lift(S, steps=3):
    T = np.zeros((L,L,L), dtype=np.int64)
    for (i,j,k) in target_tensor(n,n,n): T[i,j,k]=1
    A = [vec(a) for a,b,c in S]; B=[vec(b) for a,b,c in S]; C=[vec(c) for a,b,c in S]   # integer 0/1 lifts
    r = len(S)
    for step in range(1, steps+1):
        mod = 2**(step+1)
        Ssum = sum(np.einsum('i,j,k->ijk', A[t], B[t], C[t]) for t in range(r))
        diff = (Ssum - T)
        assert (diff % (2**step) == 0).all(), "previous level broken"
        E = ((diff // (2**step)) % 2).reshape(-1)          # carry to cancel at this level
        # unknowns: alpha_t (16), beta_t (16), gamma_t (16) per term; equation: sum_t alpha_t[i] b[j] c[k] + a[i] beta[j] c[k] + a[i] b[j] gamma[k] = E (mod 2)
        M = np.zeros((L**3, 3*L*r), dtype=np.uint8)
        idx = np.arange(L**3).reshape(L,L,L)
        for t in range(r):
            a,b,c = A[t]%2, B[t]%2, C[t]%2
            bc = np.einsum('j,k->jk', b, c); ac = np.einsum('i,k->ik', a, c); ab = np.einsum('i,j->ij', a, b)
            for i in range(L):   # alpha_t[i] multiplies b_j c_k over all (j,k)
                M[idx[i,:,:].reshape(-1), 3*L*t + i] = bc.reshape(-1)
            for j in range(L):
                M[idx[:,j,:].reshape(-1), 3*L*t + L + j] = ac.reshape(-1)
            for k in range(L):
                M[idx[:,:,k].reshape(-1), 3*L*t + 2*L + k] = ab.reshape(-1)
        ok, x, rank = f2_solve(M, E)
        print(f"  level Z/{mod}: system {M.shape}, rank {rank}, consistent={ok}", flush=True)
        if not ok: return step
        for t in range(r):
            A[t] = A[t] + (2**step) * x[3*L*t:3*L*t+L].astype(np.int64)
            B[t] = B[t] + (2**step) * x[3*L*t+L:3*L*t+2*L].astype(np.int64)
            C[t] = C[t] + (2**step) * x[3*L*t+2*L:3*L*t+3*L].astype(np.int64)
    return steps+1
if __name__ == "__main__":
    p = sys.argv[1]; S = load(p); print(p, "rank", len(S))
    lvl = lift(S, steps=int(sys.argv[2]) if len(sys.argv)>2 else 3)
    print("RESULT:", "lifts through Z/%d (all tested levels)" % (2**lvl) if lvl>3 else f"NO lift to Z/{2**(lvl+1)} -> no lift to Z with this F2 reduction")
