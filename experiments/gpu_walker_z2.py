"""
Z2-symmetric (transposition-invariant) fused CUDA flip-graph walker.  (A3, new variant)

Discovery driving this file: AlphaTensor's 4x4 rank-47 scheme over F2 is exactly
invariant under the involution  tau(a,b,c) = (b^T, a^T, c^T)  (verified in-session),
while the published symmetric flip-graph work (Moosbauer-Poole) searched C3 and C3xZ2
and found nothing below 49. So we walk the tau-symmetric subspace: every scheme is a set
of tau-orbits -- pairs {t, tau t} and fixed points (a, a^T, c) with c symmetric.

Symmetric moves (all keep the scheme tau-invariant and valid):
  * pair flip: flip (x,y) on a shared axis, then overwrite partners  x~ := tau(x'),
    y~ := tau(y')  (valid because tau is a tensor automorphism), when {x,y} and
    {x~,y~} are disjoint;
  * intra-orbit flip: x and tau x sharing c -> standard c-axis flip (result symmetric);
  * fixed-point flip (derived in notes): f=(a,a^T,c) sharing a with y=(a,b,d):
        f.c ^= y.c ^ y~.c ;  y.b ^= f.b ;  y~.a ^= f.a      (mirror for shared b);
  * symmetric plus transition: split regular x with another term's factor, then
    x~ := tau(x1), append tau(x2)  (rank +2);
  * reductions: reduce_all (symmetry-preserving), partners rebuilt afterwards.
Everything host-verified before it counts.

Usage: python gpu_walker_z2.py --fmt 4 --target 47 --seconds 1200
"""
import argparse, json, time, sys
import numpy as np
import cupy as cp
from flip_graph import verify, trivial_scheme

KERNEL = r'''
extern "C" {
__device__ __forceinline__ unsigned long long xs64(unsigned long long &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
__device__ __forceinline__ unsigned int rnd(unsigned long long &s, unsigned int n) {
    return (unsigned int)(xs64(s) % n); }
__device__ __forceinline__ unsigned long long tr(unsigned long long x) {   // NxN transpose
    unsigned long long y = 0;
    for (int i = 0; i < NN; i++) for (int j = 0; j < NN; j++)
        if ((x >> (i*NN+j)) & 1ULL) y |= 1ULL << (j*NN+i);
    return y;
}
__device__ __forceinline__ int f2rank(unsigned long long x) {
    unsigned int rows[8]; int rk = 0;
    for (int r = 0; r < NN; r++) rows[r] = (unsigned int)((x >> (r*NN)) & ((1u << NN) - 1u));
    for (int bit = 0; bit < NN; bit++) {
        int piv = -1;
        for (int r = 0; r < NN; r++) if (rows[r] & (1u << bit)) { piv = r; break; }
        if (piv < 0) continue;
        unsigned int p = rows[piv]; rows[piv] = 0; rk++;
        for (int r = 0; r < NN; r++) if (rows[r] & (1u << bit)) rows[r] ^= p;
    }
    return rk;
}
__device__ __forceinline__ bool rank_ok(unsigned long long x) { return MAXRANK == 0 || f2rank(x) <= MAXRANK; }
__device__ int reduce_all(unsigned long long *A, unsigned long long *B, unsigned long long *C, int rank) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int t = 0; t < rank; t++)
            if (!A[t] || !B[t] || !C[t]) { rank--; A[t]=A[rank]; B[t]=B[rank]; C[t]=C[rank]; t--; changed = true; }
        for (int u = 0; u < rank && !changed; u++)
            for (int v = u + 1; v < rank; v++) {
                int sa = (A[u]==A[v]), sb = (B[u]==B[v]), sc = (C[u]==C[v]);
                if (sa + sb + sc >= 2) {
                    if (sa && sb) C[u] ^= C[v]; else if (sa && sc) B[u] ^= B[v]; else A[u] ^= A[v];
                    rank--; A[v]=A[rank]; B[v]=B[rank]; C[v]=C[rank]; changed = true; break;
                }
            }
    }
    return rank;
}
// rebuild partner table; returns false if the scheme is not tau-symmetric
__device__ bool partners(const unsigned long long *A, const unsigned long long *B,
                         const unsigned long long *C, int rank, short *P) {
    bool ok = true;
    for (int t = 0; t < rank; t++) P[t] = -1;
    for (int t = 0; t < rank; t++) {
        if (P[t] != -1) continue;                 // already matched as someone's partner
        unsigned long long ta = tr(B[t]), tb = tr(A[t]), tc = tr(C[t]);
        int p = -1;
        for (int u = t; u < rank; u++)            // one-to-one matching (handles duplicates)
            if (P[u] == -1 && A[u]==ta && B[u]==tb && C[u]==tc) { p = u; break; }
        if (p < 0) { ok = false; P[t] = (short)t; }
        else { P[t] = (short)p; P[p] = (short)t; }
    }
    return ok;
}


// symmetry-aware reduction: merge mirrored pairs together. Requires valid matching P.
// Returns new rank; sets *broke if an unsupported case was left unreduced.
__device__ int reduce_sym(unsigned long long *A, unsigned long long *B, unsigned long long *C,
                          int rank, short *P, bool *broke) {
    bool changed = true;
    while (changed) {
        changed = false;
        // zero-factor drop (drop partner too if it is also zero -> it will be, by symmetry)
        for (int t = 0; t < rank; t++)
            if (!A[t] || !B[t] || !C[t]) {
                rank--; A[t]=A[rank]; B[t]=B[rank]; C[t]=C[rank]; t--; changed = true;
            }
        if (changed) { partners(A, B, C, rank, P); continue; }
        for (int u = 0; u < rank && !changed; u++)
            for (int v = u + 1; v < rank; v++) {
                int sa = (A[u]==A[v]), sb = (B[u]==B[v]), sc = (C[u]==C[v]);
                if (sa + sb + sc < 2) continue;
                int pu = P[u], pv = P[v];
                bool uf = (pu == u), vf = (pv == v);
                if (pv == u || (uf && vf)) {                 // partners, or two fixed points
                    if (sa && sb) C[u] ^= C[v]; else if (sa && sc) B[u] ^= B[v]; else A[u] ^= A[v];
                    rank--; A[v]=A[rank]; B[v]=B[rank]; C[v]=C[rank]; changed = true; break;
                } else if (uf || vf) {                        // fixed f + regular r (+ its partner)
                    int f = uf ? u : v, r = uf ? v : u, pr = P[r];
                    if (sa && sb) {                           // f=(a,aT,c), r=(a,aT,d), pr=(a,aT,dT)
                        C[f] ^= C[r] ^ C[pr];
                        int hi = r > pr ? r : pr, lo = r > pr ? pr : r;
                        rank--; A[hi]=A[rank]; B[hi]=B[rank]; C[hi]=C[rank];
                        rank--; A[lo]=A[rank]; B[lo]=B[rank]; C[lo]=C[rank];
                        changed = true; break;
                    }
                    // (a&c) or (b&c) shared with a fixed point: handled by generalized R3 below
                    continue;
                } else {                                      // two regular, distinct orbits
                    if (sa && sb) { C[u] ^= C[v]; C[pu] ^= C[pv]; }
                    else if (sa && sc) { B[u] ^= B[v]; A[pu] ^= A[pv]; }
                    else { A[u] ^= A[v]; B[pu] ^= B[pv]; }
                    int hi = v > pv ? v : pv, lo = v > pv ? pv : v;
                    rank--; A[hi]=A[rank]; B[hi]=B[rank]; C[hi]=C[rank];
                    rank--; A[lo]=A[rank]; B[lo]=B[rank]; C[lo]=C[rank];
                    changed = true; break;
                }
            }
        if (!changed) {   // R3: fixed f + pair (r,pr) all sharing c, A[f] = A[r]^A[pr] -> (a,aT,c),(e,eT,c)
            for (int f = 0; f < rank && !changed; f++) {
                if (P[f] != f) continue;
                for (int r = 0; r < rank; r++) {
                    int pr = P[r];
                    if (pr == r || pr < r) continue;
                    if (C[r] != C[f] || C[pr] != C[f]) continue;
                    unsigned long long a = A[r], e = A[pr], g = A[f], x, y;
                    if (g == (a ^ e))      { x = a;     y = e; }
                    else if (g == a)       { x = a ^ e; y = e; }
                    else if (g == e)       { x = a ^ e; y = a; }
                    else continue;
                    // g gT + a b + bT aT = g gT + (a+e)(a+e)T + a aT + e eT  (e = bT); g cancels one
                    A[f] = x; B[f] = tr(x);            // C[f] unchanged (= c)
                    A[r] = y; B[r] = tr(y); C[r] = C[f];
                    rank--; A[pr]=A[rank]; B[pr]=B[rank]; C[pr]=C[rank];
                    changed = true; break;
                }
            }
        }
        if (!changed) {   // R4': fixed (x,xT,c),(y,yT,c),(x^y,(x^y)T,c) -> pair (x,yT,c),(y,xT,c)
            for (int f1 = 0; f1 < rank && !changed; f1++) {
                if (P[f1] != f1) continue;
                for (int f2 = f1 + 1; f2 < rank && !changed; f2++) {
                    if (P[f2] != f2 || C[f2] != C[f1]) continue;
                    unsigned long long z = A[f1] ^ A[f2];
                    for (int f3 = f2 + 1; f3 < rank; f3++) {
                        if (P[f3] != f3 || C[f3] != C[f1] || A[f3] != z) continue;
                        unsigned long long x = A[f1], y = A[f2], c = C[f1];
                        A[f1] = x; B[f1] = tr(y); C[f1] = c;
                        A[f2] = y; B[f2] = tr(x); C[f2] = c;
                        rank--; A[f3]=A[rank]; B[f3]=B[rank]; C[f3]=C[rank];
                        changed = true; break;
                    }
                }
            }
        }
        if (changed) partners(A, B, C, rank, P);
    }
    return rank;
}

__global__ void walk(
    unsigned long long *gA, unsigned long long *gB, unsigned long long *gC,
    int *gRank, int *gSince, int *gBestW, unsigned long long *gSeed,
    const unsigned long long *elite, const int *eliteRank, const int *walkElite,
    const int *policy, int nWalks, int maxt, int steps,
    int dumpThresh, int *globalBest, int dumpMod,
    unsigned long long *outBuf, int *outRank, int *outSlots, int outCap, int *symBroken)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= nWalks) return;
    const int plusAfter = policy[blockIdx.x*4+0], climb = policy[blockIdx.x*4+1];
    const int greedy = policy[blockIdx.x*4+2], redMask = policy[blockIdx.x*4+3];

    unsigned long long A[MAXT], B[MAXT], C[MAXT]; short P[MAXT];
    int rank = gRank[w];
    for (int t = 0; t < rank; t++) { A[t]=gA[(size_t)w*maxt+t]; B[t]=gB[(size_t)w*maxt+t]; C[t]=gC[(size_t)w*maxt+t]; }
    unsigned long long seed = gSeed[w];
    int since = gSince[w], bestw = gBestW[w];
    const int e = walkElite[w];
    bool sym = partners(A, B, C, rank, P);

    for (int step = 0; step < steps; step++) {
        since++;
        unsigned int x = rnd(seed, rank), y = rnd(seed, rank);
        if (x == y) continue;
        int sh[3]; int ns = 0;
        if (A[x]==A[y]) sh[ns++]=0;
        if (B[x]==B[y]) sh[ns++]=1;
        if (C[x]==C[y]) sh[ns++]=2;
        if (ns == 0 || ns == 3) goto escape;
        {
        int axis = sh[rnd(seed, ns)];
        int xb = P[x], yb = P[y];
        bool xf = (xb == (int)x), yf = (yb == (int)y);
        bool did = false;
        if (!sym) {                                   // fallback: plain asymmetric flip
            if (xs64(seed)&1) { unsigned t=x; x=y; y=t; }
            if (axis==0 && rank_ok(C[x]^C[y]) && rank_ok(B[y]^B[x])) { C[x]^=C[y]; B[y]^=B[x]; did = true; }
            else if (axis==1 && rank_ok(A[x]^A[y]) && rank_ok(C[y]^C[x])) { A[x]^=A[y]; C[y]^=C[x]; did = true; }
            else if (axis==2 && rank_ok(B[x]^B[y]) && rank_ok(A[y]^A[x])) { B[x]^=B[y]; A[y]^=A[x]; did = true; }
        } else if (yb == (int)x) {                    // intra-orbit: only c-axis flip is symmetric
            if (axis == 2 && rank_ok(B[x]^B[y]) && rank_ok(A[y]^A[x])) { B[x]^=B[y]; A[y]^=A[x]; did = true; }
        } else if (xf || yf) {                        // fixed point + regular pair
            unsigned f = xf ? x : y, r = xf ? y : x; int rb = P[r];
            if (yf && xf) { /* two fixed points sharing one axis: no symmetric flip */ }
            else if (axis == 0) {                     // shared a
                unsigned long long nb = B[r]^B[f], na = A[rb]^A[f];
                int before = __popcll(B[r]) + __popcll(A[rb]), after = __popcll(nb) + __popcll(na);
                if ((after <= before || rnd(seed,1000) >= (unsigned)greedy) && rank_ok(nb) && rank_ok(na) && rank_ok(C[f]^C[r]^C[rb])) {
                    C[f] ^= C[r] ^ C[rb]; B[r] = nb; A[rb] = na; did = true; }
            } else if (axis == 1) {                   // shared b
                unsigned long long na = A[r]^A[f], nb = B[rb]^B[f];
                int before = __popcll(A[r]) + __popcll(B[rb]), after = __popcll(na) + __popcll(nb);
                if ((after <= before || rnd(seed,1000) >= (unsigned)greedy) && rank_ok(na) && rank_ok(nb) && rank_ok(C[f]^C[r]^C[rb])) {
                    C[f] ^= C[r] ^ C[rb]; A[r] = na; B[rb] = nb; did = true; }
            }
        } else if (xb != (int)y && yb != (int)x && xb != (int)yb) {   // disjoint pairs
            if (xs64(seed)&1) { unsigned t=x; x=y; y=t; int tb=xb; xb=yb; yb=tb; }
            unsigned long long ni, nj; int before, after;
            if (axis==0) { ni=C[x]^C[y]; nj=B[y]^B[x]; before=__popcll(C[x])+__popcll(B[y]); }
            else if (axis==1) { ni=A[x]^A[y]; nj=C[y]^C[x]; before=__popcll(A[x])+__popcll(C[y]); }
            else { ni=B[x]^B[y]; nj=A[y]^A[x]; before=__popcll(B[x])+__popcll(A[y]); }
            after = __popcll(ni)+__popcll(nj);
            if ((after <= before || rnd(seed,1000) >= (unsigned)greedy) && rank_ok(ni) && rank_ok(nj)) {
                if (axis==0) { C[x]=ni; B[y]=nj; } else if (axis==1) { A[x]=ni; C[y]=nj; } else { B[x]=ni; A[y]=nj; }
                A[xb]=tr(B[x]); B[xb]=tr(A[x]); C[xb]=tr(C[x]);
                A[yb]=tr(B[y]); B[yb]=tr(A[y]); C[yb]=tr(C[y]);
                did = true;
            }
        }
        if (did) {
            bool zero = false;
            for (int t = 0; t < rank && !zero; t++) zero = (!A[t]||!B[t]||!C[t]);
            if (zero || (xs64(seed) & (unsigned)redMask) == 0) {
                int nr; bool broke = false;
                if (sym) { nr = reduce_sym(A, B, C, rank, P, &broke); if (broke) nr = reduce_all(A, B, C, nr); }
                else nr = reduce_all(A, B, C, rank);
                if (nr < rank) {
                    rank = nr; since = 0;
                    sym = partners(A, B, C, rank, P);
                    if (!sym) atomicAdd(symBroken, 1);
                    if (rank < bestw) bestw = rank;
                    int old = atomicMin(globalBest, rank);
                    bool record = rank < old, sample = (rank <= old) && (rnd(seed, dumpMod) == 0);
                    if ((record || sample) && rank <= dumpThresh) {
                        int slot = atomicAdd(outSlots, 1);
                        if (slot < outCap) {
                            outRank[slot] = rank;
                            for (int t = 0; t < rank; t++) {
                                outBuf[((size_t)slot*maxt+t)*3+0]=A[t]; outBuf[((size_t)slot*maxt+t)*3+1]=B[t]; outBuf[((size_t)slot*maxt+t)*3+2]=C[t]; }
                        }
                    }
                }
            }
        }
        }
        escape:
        if (since > plusAfter) {
            since = 0;
            if (!sym || rank > bestw + climb || rank >= maxt - 2) {   // lost symmetry -> restart
                rank = eliteRank[e];
                for (int t = 0; t < rank; t++) { A[t]=elite[((size_t)e*maxt+t)*3+0]; B[t]=elite[((size_t)e*maxt+t)*3+1]; C[t]=elite[((size_t)e*maxt+t)*3+2]; }
                sym = partners(A, B, C, rank, P);
            } else {
                int k = rnd(seed, rank), jj = rnd(seed, rank), ax = rnd(seed, 3);
                int kb = P[k];
                if (sym && kb != k && C[k] == tr(C[k]) && rank < maxt - 3 && (xs64(seed) & 1)) {
                    // P4': pair (a,b,c),(bT,aT,c), c symmetric -> (a+bT)(a+bT)T c + a aT c + bT b c
                    unsigned long long a = A[k], e = tr(B[k]), c = C[k], g = a ^ e;
                    if (g) {
                        A[k]=g; B[k]=tr(g); C[k]=c;
                        A[kb]=a; B[kb]=tr(a); C[kb]=c;
                        A[rank]=e; B[rank]=tr(e); C[rank]=c;
                        P[k]=(short)k; P[kb]=(short)kb; P[rank]=(short)rank;
                        rank += 1;
                    }
                    goto escape_done;
                }
                if (sym && kb == k && rank < maxt - 4) {          // P3: split fixed (g,gT,c) with a = A[jj]
                    unsigned long long g = A[k], a = A[jj], c = C[k];
                    if (a && a != g) {
                        unsigned long long e = g ^ a;
                        A[k]=a; B[k]=tr(a); C[k]=c;                          // (a,aT,c) fixed
                        A[rank]=e; B[rank]=tr(e); C[rank]=c;                 // (e,eT,c) fixed
                        A[rank+1]=a; B[rank+1]=tr(e); C[rank+1]=c;           // (a,eT,c)
                        A[rank+2]=e; B[rank+2]=tr(a); C[rank+2]=c;           // (e,aT,c) = tau of previous
                        P[rank]=(short)rank; P[rank+1]=(short)(rank+2); P[rank+2]=(short)(rank+1);
                        rank += 3;
                    }
                    goto escape_done;
                }
                unsigned long long fk = ax==0?A[k]:(ax==1?B[k]:C[k]);
                unsigned long long fj = ax==0?A[jj]:(ax==1?B[jj]:C[jj]);
                if (fj != fk && fj && kb != k && sym) {
                    // split k -> k, new ; then partner kb := tau(k), append tau(new)
                    A[rank]=A[k]; B[rank]=B[k]; C[rank]=C[k];
                    if (ax==0) { A[k]=fj; A[rank]=fk^fj; } else if (ax==1) { B[k]=fj; B[rank]=fk^fj; } else { C[k]=fj; C[rank]=fk^fj; }
                    A[kb]=tr(B[k]); B[kb]=tr(A[k]); C[kb]=tr(C[k]);
                    A[rank+1]=tr(B[rank]); B[rank+1]=tr(A[rank]); C[rank+1]=tr(C[rank]);
                    P[rank]=(short)(rank+1); P[rank+1]=(short)rank;
                    rank += 2;
                } else if (fj != fk && fj && kb == k && !sym) {   // asymmetric fallback split
                    A[rank]=A[k]; B[rank]=B[k]; C[rank]=C[k];
                    if (ax==0) { A[k]=fj; A[rank]=fk^fj; } else if (ax==1) { B[k]=fj; B[rank]=fk^fj; } else { C[k]=fj; C[rank]=fk^fj; }
                    rank++;
                }
            }
            escape_done: ;
        }
    }
    for (int t = 0; t < rank; t++) { gA[(size_t)w*maxt+t]=A[t]; gB[(size_t)w*maxt+t]=B[t]; gC[(size_t)w*maxt+t]=C[t]; }
    gRank[w]=rank; gSince[w]=since; gBestW[w]=bestw; gSeed[w]=seed;
}
}'''

def tr_host(x, n):
    y = 0
    for i in range(n):
        for j in range(n):
            if (x >> (i*n+j)) & 1: y |= 1 << (j*n+i)
    return y

def is_tau_symmetric(sch, n):
    s = set(sch)
    return all((tr_host(b, n), tr_host(a, n), tr_host(c, n)) in s for (a, b, c) in sch)

def run(n, target, seconds, walks, steps, seed, out_prefix, n_policies=8, log_every=20, seed_scheme=None, max_rank=0):
    rs = np.random.RandomState(seed)
    if seed_scheme:
        base = [tuple(t) for t in json.load(open(seed_scheme))["scheme_bitmasks"]]
        assert verify(base, n, n, n), "seed scheme invalid"
        print(f"seeding from {seed_scheme} rank {len(base)}", flush=True)
    else:
        base = trivial_scheme(n, n, n)
    assert is_tau_symmetric(base, n), "seed scheme is not tau-symmetric"
    maxt = n**3 + 12
    mod = cp.RawModule(code=KERNEL.replace("MAXT", str(maxt)).replace("MAXRANK", str(max_rank)).replace("NN", str(n)),
                       options=("-std=c++14",))
    kern = mod.get_function("walk")
    pool = {tuple(sorted(base)): len(base)}
    ECAP = 256
    eliteBuf = cp.zeros((ECAP, maxt, 3), dtype=cp.uint64); eliteRank = cp.zeros(ECAP, dtype=cp.int32)
    def push_elites():
        r0 = min(pool.values())
        keep = [s for s, r in pool.items() if r <= r0 + 2][:ECAP]
        buf = np.zeros((ECAP, maxt, 3), dtype=np.uint64); rk = np.zeros(ECAP, dtype=np.int32)
        for e in range(ECAP):
            s = keep[e % len(keep)]; rk[e] = len(s)
            for t, (a, b, c) in enumerate(s): buf[e, t] = (a, b, c)
        eliteBuf.set(buf); eliteRank.set(rk); return len(keep)
    n_elite = push_elites()
    block = 128; grid = (walks + block - 1) // block
    walkElite = cp.asarray(rs.randint(0, n_elite, walks).astype(np.int32))
    import os
    def rand_policy():
        if os.environ.get("FORCE_POLICY"):
            return [int(v) for v in os.environ["FORCE_POLICY"].split(",")]
        return [int(rs.choice([300, 1000, 3000, 10000])), int(rs.choice([1, 2, 3, 4])),
                int(rs.choice([0, 200, 500, 800])), int(rs.choice([3, 7, 15, 31]))]
    policies = [rand_policy() for _ in range(n_policies)]; pol_scores = np.zeros(n_policies)
    block_pol = rs.randint(0, n_policies, grid)
    def policy_array():
        return cp.asarray(np.array([policies[block_pol[g]] for g in range(grid)], dtype=np.int32).ravel())
    gPol = policy_array()
    r0 = len(base)
    gA = cp.zeros((walks, maxt), dtype=cp.uint64); gB = cp.zeros_like(gA); gC = cp.zeros_like(gA)
    for t, (a, b, c) in enumerate(base): gA[:, t] = a; gB[:, t] = b; gC[:, t] = c
    gRank = cp.full(walks, r0, dtype=cp.int32); gSince = cp.zeros(walks, dtype=cp.int32)
    gBestW = cp.full(walks, r0, dtype=cp.int32)
    gSeed = cp.asarray(rs.randint(1, 2**63, size=walks, dtype=np.uint64))
    OUTCAP = 128
    outBuf = cp.zeros((OUTCAP, maxt, 3), dtype=cp.uint64); outRank = cp.zeros(OUTCAP, dtype=cp.int32)
    outSlots = cp.zeros(1, dtype=cp.int32); globalBest = cp.full(1, r0, dtype=cp.int32)
    symBroken = cp.zeros(1, dtype=cp.int32)

    t0 = time.time(); launches = 0; total = 0; best = r0; verified = {}; prog = []
    prev_bestw = gBestW.get(); n_sym = n_asym = 0
    while time.time() - t0 < seconds:
        kern((grid,), (block,), (gA, gB, gC, gRank, gSince, gBestW, gSeed, eliteBuf, eliteRank, walkElite,
             gPol, np.int32(walks), np.int32(maxt), np.int32(steps), np.int32(best), globalBest,
             np.int32(2048), outBuf, outRank, outSlots, np.int32(OUTCAP), symBroken))
        cp.cuda.Stream.null.synchronize(); launches += 1; total += walks*steps
        slots = min(int(outSlots.get()[0]), OUTCAP)
        if slots:
            buf = outBuf.get()[:slots]; rks = outRank.get()[:slots]; outSlots.fill(0)
            for s in range(slots):
                r = int(rks[s]); sch = [(int(buf[s,t,0]), int(buf[s,t,1]), int(buf[s,t,2])) for t in range(r)]
                key = tuple(sorted(sch))
                if key in pool: continue
                if verify(sch, n, n, n):
                    symm = is_tau_symmetric(sch, n)
                    if symm: n_sym += 1
                    else: n_asym += 1
                    pool[key] = r
                    if r not in verified:
                        verified[r] = sch
                        json.dump({"format":[n,n,n],"rank":r,"field":"F2","verified_brent_f2":True,
                                   "tau_symmetric":symm,"found_by":"gpu_walker_z2 self-search",
                                   "scheme_bitmasks":[list(t) for t in sch]},
                                  open(f"{out_prefix}_rank{r}.json","w"), indent=1)
                        print(f"  rank {r}: VERIFIED  tau-symmetric={symm}", flush=True)
                else:
                    print(f"  rank {r}: REJECTED-BY-HOST", flush=True)
                    json.dump({"scheme_bitmasks":[list(t) for t in sch]}, open(f"{out_prefix}_REJECTED_rank{r}.json","w"))
        gb = int(globalBest.get()[0])
        if gb < best:
            best = gb; el = time.time()-t0
            print(f"[{el:7.1f}s] best rank {gb}  pool={len(pool)}  {total/el/1e6:.0f}M steps/s", flush=True)
            prog.append({"t": el, "best": gb})
        cur = gBestW.get(); gains = np.maximum(prev_bestw - cur, 0).reshape(grid, block).sum(axis=1)
        for g in range(grid): pol_scores[block_pol[g]] += gains[g]
        prev_bestw = cur
        if launches % log_every == 0:
            order = np.argsort(-pol_scores)
            for wi, bi in zip(order[-2:], order[:2]):
                child = list(policies[bi]); idx = rs.randint(4); child[idx] = rand_policy()[idx]
                policies[wi] = child; pol_scores[wi] = pol_scores[bi]*0.5
            block_pol = rs.randint(0, n_policies, grid); gPol = policy_array()
            n_elite = push_elites(); walkElite = cp.asarray(rs.randint(0, n_elite, walks).astype(np.int32))
            el = time.time()-t0
            print(f"[{el:7.1f}s] launch {launches}: best={best} pool={len(pool)} sym/asym pool adds={n_sym}/{n_asym} "
                  f"symBreaks={int(symBroken.get()[0])} top={policies[order[0]]} {total/el/1e6:.0f}M steps/s", flush=True)
        if target in verified:
            print(f"TARGET {target} HIT and verified.", flush=True); break
    dt = time.time()-t0
    r0min = min(pool.values())
    json.dump({"format":[n,n,n],"schemes":[[list(t) for t in s] for s, r in pool.items() if r <= r0min + 1][:20000]},
              open(f"{out_prefix}_pool.json","w"))
    res = {"format":[n,n,n],"target":target,"best_rank":best,"verified_ranks":sorted(verified),
           "pool":len(pool),"launches":launches,"total_steps":total,"steps_per_sec":total/dt,"seconds":dt,
           "sym_breaks":int(symBroken.get()[0]),"progress":prog,"final_policies":policies}
    json.dump(res, open(f"{out_prefix}_runz2.json","w"), indent=1)
    print(json.dumps({k:v for k,v in res.items() if k not in ("progress","final_policies")}, indent=1))
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", type=int, required=True, help="n for square n x n x n")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--walks", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-scheme", type=str, default=None)
    ap.add_argument("--max-rank", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    res = run(a.fmt, a.target, a.seconds, a.walks, a.steps, a.seed, f"../results/z2_{a.fmt}{a.fmt}{a.fmt}{a.tag}",
              seed_scheme=a.seed_scheme, max_rank=a.max_rank)
    sys.exit(0 if a.target in res["verified_ranks"] else 2)
