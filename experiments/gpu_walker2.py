"""
Generation-2 fused CUDA flip-graph walker: population dynamics + learned policy.

Upgrades over gpu_walker.py (each tagged with the plan hypothesis it serves):
  U1  Targeted plus transitions: split term k using another term's factor, so the
      pieces immediately share a factor with an existing term (Kauers-Moosbauer /
      Arai et al. operator). The gen-1 random-mask split almost never did.
  A1  Population: elite pool of verified schemes on the host; every walk restarts
      from a randomly assigned elite (migration). Plus a NEW operator, cross-scheme
      subtensor exchange: if k terms of scheme A sum to the same tensor as k' < k
      terms of scheme B (or of A itself), swap them -> rank drops by k-k'. Flips and
      reductions are the k=2,k'=1 special case; we search k=3,k'=2.
  A2  Learned policy (evolution strategy): per-block policy parameters
      (plus_after, climb, greedy-sparsity probability) are scored each launch by
      the rank improvements their walks produce; losers are replaced by mutated
      winners. The LLM-free AlphaEvolve-lite loop.
  A3  Symmetry images: the S3 action (cyclic (a,b,c)->(b,c,a) and transposition)
      maps valid schemes to valid schemes; images of elites diversify the pool for
      the exchange operator. Full quotient (symmetry-restricted) walks are
      deliberately NOT used for 4x4: Moosbauer-Poole proved they bottom out at 49.

Verifier-first: nothing enters the pool or results without the exact host Brent
certificate. Usage:
  python gpu_walker2.py --fmt 4 4 4 --target 47 --seconds 1800 \
      --seed-scheme ../results/gpu_444_rank49.json
"""
import argparse, json, time, sys, itertools, os
import numpy as np
import cupy as cp
from flip_graph import verify, trivial_scheme

KERNEL = r'''
extern "C" {
__device__ __forceinline__ unsigned long long xs64(unsigned long long &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
__device__ __forceinline__ unsigned int rnd(unsigned long long &s, unsigned int n) {
    return (unsigned int)(xs64(s) % n); }

__device__ __forceinline__ int f2rank(unsigned long long x) {   // rank of NN x NN F2 matrix
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
__device__ int reduce_all(unsigned long long *A, unsigned long long *B,
                          unsigned long long *C, int rank) {
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

// policy params per block: [plusAfter, climb, greedyPermille, reduceMask]
__global__ void walk(
    unsigned long long *gA, unsigned long long *gB, unsigned long long *gC,
    int *gRank, int *gSince, int *gBestW, unsigned long long *gSeed,
    const unsigned long long *elite, const int *eliteRank, const int *walkElite,
    const int *policy, int nWalks, int maxt, int steps,
    int dumpThresh, int *globalBest, int dumpMod,
    unsigned long long *outBuf, int *outRank, int *outSlots, int outCap)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= nWalks) return;
    const int plusAfter = policy[blockIdx.x*5+0];
    const int climb     = policy[blockIdx.x*5+1];
    const int greedy    = policy[blockIdx.x*5+2];
    const int redMask   = policy[blockIdx.x*5+3];
    const int rank3P    = policy[blockIdx.x*5+4];   // reject rank-3-destroying flips with this permille
    const int maxRank   = MAXRANK;   // compile-time: 0 = off

    unsigned long long A[MAXT], B[MAXT], C[MAXT];
    int rank = gRank[w];
    for (int t = 0; t < rank; t++) { A[t]=gA[(size_t)w*maxt+t]; B[t]=gB[(size_t)w*maxt+t]; C[t]=gC[(size_t)w*maxt+t]; }
    unsigned long long seed = gSeed[w];
    int since = gSince[w], bestw = gBestW[w];
    const int e = walkElite[w];

    for (int step = 0; step < steps; step++) {
        since++;
        int i=-1, j=-1, axis=-1;
        for (int t = 0; t < 24 && axis < 0; t++) {
            unsigned int x = rnd(seed, rank), y = rnd(seed, rank);
            if (x == y) continue;
            int sh[3]; int ns = 0;
            if (A[x]==A[y]) sh[ns++]=0;
            if (B[x]==B[y]) sh[ns++]=1;
            if (C[x]==C[y]) sh[ns++]=2;
            if (ns > 0 && ns < 3) { axis = sh[rnd(seed, ns)]; i = x; j = y; }
        }
        if (axis >= 0) {
            if (xs64(seed) & 1) { int tmp=i; i=j; j=tmp; }
            // A2 greedy-sparsity gate: reject weight-increasing flips with prob greedy/1000
            unsigned long long ni, nj; int before, after;
            if (axis == 0)      { ni = C[i]^C[j]; nj = B[j]^B[i]; before = __popcll(C[i])+__popcll(B[j]); }
            else if (axis == 1) { ni = A[i]^A[j]; nj = C[j]^C[i]; before = __popcll(A[i])+__popcll(C[j]); }
            else                { ni = B[i]^B[j]; nj = A[j]^A[i]; before = __popcll(B[i])+__popcll(A[j]); }
            after = __popcll(ni) + __popcll(nj);
            bool accept = (after <= before) || (rnd(seed, 1000) >= (unsigned)greedy);
            if (accept && maxRank > 0 && NN <= 8) {
                int ri = f2rank(ni), rj = f2rank(nj);
                if (ri > maxRank || rj > maxRank) accept = false;
                else if (rank3P > 0) {
                    unsigned long long oi, oj;
                    if (axis == 0) { oi = C[i]; oj = B[j]; } else if (axis == 1) { oi = A[i]; oj = C[j]; } else { oi = B[i]; oj = A[j]; }
                    int r3b = (f2rank(oi) == 3) + (f2rank(oj) == 3), r3a = (ri == 3) + (rj == 3);
                    if (r3a < r3b && rnd(seed, 1000) < (unsigned)rank3P) accept = false;
                }
            }
            if (accept) {
                if (axis == 0)      { C[i] = ni; B[j] = nj; }
                else if (axis == 1) { A[i] = ni; C[j] = nj; }
                else                { B[i] = ni; A[j] = nj; }
                bool zero = (!A[i]||!B[i]||!C[i]||!A[j]||!B[j]||!C[j]);
                if (zero || (xs64(seed) & (unsigned)redMask) == 0) {
                    int nr = reduce_all(A, B, C, rank);
                    if (nr < rank) {
                        rank = nr; since = 0;
                        if (rank < bestw) bestw = rank;
                        int old = atomicMin(globalBest, rank);
                        bool record = rank < old;
                        bool sample = (rank <= old) && (rnd(seed, dumpMod) == 0);
                        if ((record || sample) && rank <= dumpThresh) {
                            int slot = atomicAdd(outSlots, 1);
                            if (slot < outCap) {
                                outRank[slot] = rank;
                                for (int t = 0; t < rank; t++) {
                                    outBuf[((size_t)slot*maxt+t)*3+0]=A[t];
                                    outBuf[((size_t)slot*maxt+t)*3+1]=B[t];
                                    outBuf[((size_t)slot*maxt+t)*3+2]=C[t]; }
                            }
                        }
                    }
                }
            }
        }
        if (since > plusAfter) {
            since = 0;
            if (rank > bestw + climb || rank >= maxt - 1) {
                // A1 migration: restart from assigned elite
                rank = eliteRank[e];
                for (int t = 0; t < rank; t++) {
                    A[t]=elite[((size_t)e*maxt+t)*3+0]; B[t]=elite[((size_t)e*maxt+t)*3+1]; C[t]=elite[((size_t)e*maxt+t)*3+2]; }
            } else {
                // U1 targeted plus transition: split term k along axis ax using term j's factor
                int k = rnd(seed, rank), jj = rnd(seed, rank), ax = rnd(seed, 3);
                unsigned long long fk = ax==0?A[k]:(ax==1?B[k]:C[k]);
                unsigned long long fj = ax==0?A[jj]:(ax==1?B[jj]:C[jj]);
                if (fj != fk && fj) {
                    A[rank]=A[k]; B[rank]=B[k]; C[rank]=C[k];
                    if (ax==0) { A[k]=fj; A[rank]=fk^fj; }
                    else if (ax==1) { B[k]=fj; B[rank]=fk^fj; }
                    else { C[k]=fj; C[rank]=fk^fj; }
                    rank++;
                }
            }
        }
    }
    for (int t = 0; t < rank; t++) { gA[(size_t)w*maxt+t]=A[t]; gB[(size_t)w*maxt+t]=B[t]; gC[(size_t)w*maxt+t]=C[t]; }
    gRank[w]=rank; gSince[w]=since; gBestW[w]=bestw; gSeed[w]=seed;
}
}'''

# ---------------- host-side population machinery (A1, A3) ----------------
def canon(sch):
    return tuple(sorted(sch))

def term_tensor_bits(a, b, c, la, lb, lc):
    """Rank-1 tensor of (a,b,c) as a big int over la*lb*lc F2 entries."""
    x = 0
    for ia in range(la):
        if not (a >> ia) & 1: continue
        for ib in range(lb):
            if not (b >> ib) & 1: continue
            base = (ia * lb + ib) * lc
            x |= (c << base)   # c is an lc-bit mask -> shifts straight in
    return x

def sym_images(sch, n):
    """A3: S3 images of a square scheme (valid by tensor symmetry). Cyclic (a,b,c)->(b,c,a)
    and transpose-swap (a,b,c)->(bT,aT,cT). Returns the 6 images incl. identity."""
    def tr(x):
        y = 0
        for i in range(n):
            for j in range(n):
                if (x >> (i*n+j)) & 1: y |= 1 << (j*n+i)
        return y
    out = []
    s = list(sch)
    for _ in range(3):
        out.append(list(s))
        out.append([(tr(b), tr(a), tr(c)) for (a, b, c) in s])
        s = [(b, c, a) for (a, b, c) in s]
    return out

def subtensor_exchange(pool, n, m, p, max_pairs=6, k=3):
    """A1: cross-scheme k -> (k-1) exchange. Returns list of new (rank-reduced) schemes."""
    la, lb, lc = n*m, m*p, p*n
    sums2 = {}   # tensor -> (scheme_idx, (i,j))
    found = []
    for si, sch in enumerate(pool[:max_pairs]):
        tt = [term_tensor_bits(a, b, c, la, lb, lc) for (a, b, c) in sch]
        for i, j in itertools.combinations(range(len(sch)), 2):
            sums2.setdefault(tt[i] ^ tt[j], (si, (i, j)))
    for si, sch in enumerate(pool[:max_pairs]):
        tt = [term_tensor_bits(a, b, c, la, lb, lc) for (a, b, c) in sch]
        for i, j, l in itertools.combinations(range(len(sch)), 3):
            hit = sums2.get(tt[i] ^ tt[j] ^ tt[l])
            if hit is None: continue
            sj, (u, v) = hit
            if sj == si and len({i, j, l} & {u, v}) > 0: continue
            child = [t for idx, t in enumerate(sch) if idx not in (i, j, l)]
            child += [pool[sj][u], pool[sj][v]]
            if verify(child, n, m, p):
                found.append(child)
    return found

def run(n, m, p, target, seconds, walks, steps, seed, seed_scheme, out_prefix,
        n_policies=8, log_every=20, pool_path=None, max_rank=0):
    rs = np.random.RandomState(seed)
    la_bits = n*m
    if seed_scheme:
        d = json.load(open(seed_scheme)); base = [tuple(t) for t in d["scheme_bitmasks"]]
        assert verify(base, n, m, p)
    else:
        base = trivial_scheme(n, m, p)
    maxt = n*m*p + 10
    mod = cp.RawModule(code=KERNEL.replace("MAXT", str(maxt)).replace("MAXRANK", str(max_rank)).replace("NN", str(n if n == m == p else 8)),
                       options=("-std=c++14",))
    kern = mod.get_function("walk")

    # elite pool (host) keyed by canonical form, grouped by rank
    pool = {}
    def add_pool(sch, tag=""):
        c = canon(sch)
        if c in pool: return False
        pool[c] = len(sch)
        return True
    add_pool(base)
    if n == m == p:
        for img in sym_images(base, n): add_pool(img)
    if pool_path:   # schemes from a walker pool file (each was host-verified when it entered that pool)
        ext = json.load(open(pool_path))["schemes"]
        for s_ in ext: add_pool([tuple(t) for t in s_])
        print(f"loaded {len(ext)} pool schemes -> pool {len(pool)}", flush=True)

    ECAP = 256
    eliteBuf  = cp.zeros((ECAP, maxt, 3), dtype=cp.uint64)
    eliteRank = cp.zeros(ECAP, dtype=cp.int32)
    def push_elites():
        ranks = sorted(set(pool.values()))
        keep = [s for s, r in pool.items() if r <= ranks[0] + 1][:ECAP]
        if len(keep) < ECAP:
            keep += [s for s, r in pool.items() if r > ranks[0] + 1][:ECAP - len(keep)]
        buf = np.zeros((ECAP, maxt, 3), dtype=np.uint64); rk = np.zeros(ECAP, dtype=np.int32)
        for e, s in enumerate(keep):
            rk[e] = len(s)
            for t, (a, b, c) in enumerate(s): buf[e, t] = (a, b, c)
        for e in range(len(keep), ECAP):  # pad by repeating
            buf[e] = buf[e % max(1, len(keep))]; rk[e] = rk[e % max(1, len(keep))]
        eliteBuf.set(buf); eliteRank.set(rk)
        return len(keep)
    n_elite = push_elites()

    block = 128; grid = (walks + block - 1) // block
    walkElite = cp.asarray(rs.randint(0, max(1, n_elite), walks).astype(np.int32))

    # A2 policy population: [plusAfter, climb, greedyPermille, reduceMask]
    def rand_policy():
        return [int(rs.choice([300, 1000, 3000, 10000])), int(rs.choice([1, 2, 3, 5])),
                int(rs.choice([0, 200, 500, 800])), int(rs.choice([3, 7, 15, 31])),
                int(rs.choice([0, 300, 600, 900])) if max_rank > 0 else 0]
    policies = [rand_policy() for _ in range(n_policies)]
    pol_scores = np.zeros(n_policies)
    block_pol = rs.randint(0, n_policies, grid)
    def policy_array():
        arr = np.array([policies[block_pol[g]] for g in range(grid)], dtype=np.int32).ravel()
        return cp.asarray(arr)
    gPol = policy_array()

    gA = cp.zeros((walks, maxt), dtype=cp.uint64); gB = cp.zeros_like(gA); gC = cp.zeros_like(gA)
    r0 = len(base)
    for t, (a, b, c) in enumerate(base):
        gA[:, t] = a; gB[:, t] = b; gC[:, t] = c
    gRank = cp.full(walks, r0, dtype=cp.int32); gSince = cp.zeros(walks, dtype=cp.int32)
    gBestW = cp.full(walks, r0, dtype=cp.int32)
    gSeed = cp.asarray(rs.randint(1, 2**63, size=walks, dtype=np.uint64))
    OUTCAP = 128
    outBuf = cp.zeros((OUTCAP, maxt, 3), dtype=cp.uint64); outRank = cp.zeros(OUTCAP, dtype=cp.int32)
    outSlots = cp.zeros(1, dtype=cp.int32); globalBest = cp.full(1, r0, dtype=cp.int32)

    t0 = time.time(); launches = 0; total = 0; best = r0; verified = {}
    prog = []
    prev_bestw = gBestW.get()
    while time.time() - t0 < seconds:
        kern((grid,), (block,), (gA, gB, gC, gRank, gSince, gBestW, gSeed,
             eliteBuf, eliteRank, walkElite, gPol, np.int32(walks), np.int32(maxt),
             np.int32(steps), np.int32(best), globalBest, np.int32(4096),
             outBuf, outRank, outSlots, np.int32(OUTCAP)))
        cp.cuda.Stream.null.synchronize(); launches += 1; total += walks*steps
        # ---- harvest + verify ----
        slots = min(int(outSlots.get()[0]), OUTCAP)
        new_schemes = []
        if slots:
            buf = outBuf.get()[:slots]; rks = outRank.get()[:slots]; outSlots.fill(0)
            for s in range(slots):
                r = int(rks[s]); sch = [(int(buf[s,t,0]), int(buf[s,t,1]), int(buf[s,t,2])) for t in range(r)]
                if canon(sch) in pool: continue
                if verify(sch, n, m, p):
                    add_pool(sch); new_schemes.append(sch)
                    if r not in verified:
                        verified[r] = sch
                        json.dump({"format":[n,m,p],"rank":r,"field":"F2","verified_brent_f2":True,
                                   "found_by":"gpu_walker2 self-search","scheme_bitmasks":[list(t) for t in sch]},
                                  open(f"{out_prefix}_rank{r}.json","w"), indent=1)
                        print(f"  rank {r}: VERIFIED (new record for this run)", flush=True)
                else:
                    print(f"  rank {r}: REJECTED-BY-HOST", flush=True)
        # ---- A1 subtensor exchange on freshest low-rank elites ----
        if new_schemes and launches % 5 == 0:
            low = sorted(new_schemes, key=len)[:6]
            kids = subtensor_exchange(low, n, m, p)
            for kid in kids:
                if add_pool(kid):
                    print(f"  A1 exchange produced rank {len(kid)} (verified)", flush=True)
                    if len(kid) not in verified:
                        verified[len(kid)] = kid
                        json.dump({"format":[n,m,p],"rank":len(kid),"field":"F2","verified_brent_f2":True,
                                   "found_by":"A1 subtensor exchange","scheme_bitmasks":[list(t) for t in kid]},
                                  open(f"{out_prefix}_rank{len(kid)}.json","w"), indent=1)
        gb = int(globalBest.get()[0])
        if gb < best:
            best = gb; el = time.time()-t0
            print(f"[{el:7.1f}s] best rank {gb}  pool={len(pool)}  {total/el/1e6:.0f}M steps/s", flush=True)
            prog.append({"t": el, "best": gb, "pool": len(pool)})
        # ---- A2 policy scoring + evolution ----
        cur_bestw = gBestW.get()
        gains = np.maximum(prev_bestw - cur_bestw, 0).reshape(grid, block).sum(axis=1)
        for g in range(grid): pol_scores[block_pol[g]] += gains[g]
        prev_bestw = cur_bestw
        if launches % log_every == 0:
            order = np.argsort(-pol_scores)
            worst = order[-2:]; bests = order[:2]
            for wi, bi in zip(worst, bests):   # replace losers with mutated winners
                child = list(policies[bi])
                idx = rs.randint(5); child[idx] = rand_policy()[idx]
                policies[wi] = child; pol_scores[wi] = pol_scores[bi] * 0.5
            block_pol = rs.randint(0, n_policies, grid); gPol = policy_array()
            n_elite = push_elites()
            walkElite = cp.asarray(rs.randint(0, max(1, n_elite), walks).astype(np.int32))
            el = time.time()-t0
            print(f"[{el:7.1f}s] launch {launches}: best={best} pool={len(pool)} elites={n_elite} "
                  f"top-policy={policies[order[0]]} {total/el/1e6:.0f}M steps/s", flush=True)
        if best <= target and target in verified:
            print(f"TARGET {target} HIT and verified.", flush=True); break
    dt = time.time()-t0
    json.dump({"format":[n,m,p],"schemes":[[list(t) for t in s] for s in pool]},
              open(f"{out_prefix}_pool.json","w"))
    res = {"format":[n,m,p],"target":target,"best_rank":best,"verified_ranks":sorted(verified),
           "pool_size":len(pool),"launches":launches,"total_steps":total,"steps_per_sec":total/dt,
           "seconds":dt,"final_policies":policies,"policy_scores":pol_scores.tolist(),"progress":prog}
    json.dump(res, open(f"{out_prefix}_run2.json","w"), indent=1)
    print(json.dumps({k:v for k,v in res.items() if k not in ("progress","final_policies")}, indent=1))
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", nargs=3, type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--walks", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-scheme", type=str, default=None)
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--max-rank", type=int, default=0, help="forbid factors of F2 matrix rank > this (0=off)")
    a = ap.parse_args()
    n, m, p = a.fmt
    res = run(n, m, p, a.target, a.seconds, a.walks, a.steps, a.seed, a.seed_scheme,
              f"../results/g2_{n}{m}{p}{a.tag}", pool_path=a.pool, max_rank=a.max_rank)
    sys.exit(0 if a.target in res["verified_ranks"] else 2)
