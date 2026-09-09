"""
Fused CUDA flip-graph walker for matrix multiplication schemes over F2.

Each GPU thread runs an independent flip-graph random walk (Kauers-Moosbauer
dynamics: flips + reductions + plus transitions + restarts) on a scheme held as
uint64 bitmasks in thread-local memory. Persistent walk state lives in global
buffers between kernel launches; the host polls the global best rank and exactly
re-verifies (Brent equations, Python) every scheme the GPU claims. CUDA is never
trusted: a scheme only counts once the host certificate passes.

Usage:
  python gpu_walker.py --fmt 4 4 4 --target 47 --milestones 49 48 --seconds 300
"""
import argparse, json, time, sys
import numpy as np
import cupy as cp

from flip_graph import verify, trivial_scheme, target_tensor  # exact host verifier

KERNEL = r'''
extern "C" {

__device__ __forceinline__ unsigned long long xs64(unsigned long long &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
__device__ __forceinline__ unsigned int rnd(unsigned long long &s, unsigned int n) {
    return (unsigned int)(xs64(s) % n);
}

// full local reduction pass: drop zero-factor terms, merge terms equal on two axes
__device__ int reduce_all(unsigned long long *A, unsigned long long *B,
                          unsigned long long *C, int rank) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int t = 0; t < rank; t++) {
            if (!A[t] || !B[t] || !C[t]) {
                rank--; A[t]=A[rank]; B[t]=B[rank]; C[t]=C[rank];
                t--; changed = true;
            }
        }
        for (int u = 0; u < rank && !changed; u++) {
            for (int v = u + 1; v < rank; v++) {
                int sa = (A[u]==A[v]), sb = (B[u]==B[v]), sc = (C[u]==C[v]);
                if (sa + sb + sc >= 2) {
                    if (sa && sb)      C[u] ^= C[v];
                    else if (sa && sc) B[u] ^= B[v];
                    else               A[u] ^= A[v];
                    rank--; A[v]=A[rank]; B[v]=B[rank]; C[v]=C[rank];
                    changed = true; break;
                }
            }
        }
    }
    return rank;
}

__global__ void walk(
    unsigned long long *gA, unsigned long long *gB, unsigned long long *gC,
    int *gRank, int *gSince, int *gBestW, unsigned long long *gSeed,
    const unsigned long long *tA, const unsigned long long *tB,
    const unsigned long long *tC,
    int nWalks, int maxt, int rank0, int steps,
    int laBits, int plusAfter, int slack, int restartAt,
    int dumpThresh, int *globalBest,
    unsigned long long *outBuf, int *outRank, int *outSlots, int outCap)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= nWalks) return;

    unsigned long long A[MAXT], B[MAXT], C[MAXT];
    int rank = gRank[w];
    for (int t = 0; t < rank; t++) {
        A[t] = gA[(size_t)w*maxt+t]; B[t] = gB[(size_t)w*maxt+t]; C[t] = gC[(size_t)w*maxt+t];
    }
    unsigned long long seed = gSeed[w];
    int since = gSince[w], bestw = gBestW[w];

    for (int step = 0; step < steps; step++) {
        since++;
        // ---- pick a flippable pair: random tries then linear fallback ----
        int i = -1, j = -1, axis = -1;
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
            if (xs64(seed) & 1) { int tmp=i; i=j; j=tmp; }   // random direction
            if (axis == 0)      { C[i] ^= C[j]; B[j] ^= B[i]; }
            else if (axis == 1) { A[i] ^= A[j]; C[j] ^= C[i]; }
            else                { B[i] ^= B[j]; A[j] ^= A[i]; }
            // reduce on zero factor or occasionally
            bool zero = (!A[i]||!B[i]||!C[i]||!A[j]||!B[j]||!C[j]);
            if (zero || (xs64(seed) & 15u) == 0) {
                int nr = reduce_all(A, B, C, rank);
                if (nr < rank) {
                    rank = nr; since = 0;
                    if (rank < bestw) {
                        bestw = rank;
                        int old = atomicMin(globalBest, rank);
                        if (rank < old && rank <= dumpThresh) {
                            int slot = atomicAdd(outSlots, 1);
                            if (slot < outCap) {
                                outRank[slot] = rank;
                                for (int t = 0; t < rank; t++) {
                                    outBuf[((size_t)slot*maxt+t)*3+0] = A[t];
                                    outBuf[((size_t)slot*maxt+t)*3+1] = B[t];
                                    outBuf[((size_t)slot*maxt+t)*3+2] = C[t];
                                }
                            }
                        }
                    }
                }
            }
        }
        // ---- escape logic ----
        if (since > plusAfter) {
            since = 0;
            if (rank >= restartAt || rank >= bestw + slack || rank >= maxt - 1) {
                rank = rank0;                       // restart from trivial
                for (int t = 0; t < rank; t++) { A[t]=tA[t]; B[t]=tB[t]; C[t]=tC[t]; }
            } else {                                 // plus transition (rank+1)
                int k = rnd(seed, rank);
                unsigned long long mask =
                    (xs64(seed) & ((1ULL << laBits) - 1ULL));
                if (mask && mask != A[k]) {
                    A[rank] = mask ^ A[k]; B[rank] = B[k]; C[rank] = C[k];
                    A[k] = mask; rank++;
                }
            }
        }
    }

    for (int t = 0; t < rank; t++) {
        gA[(size_t)w*maxt+t] = A[t]; gB[(size_t)w*maxt+t] = B[t]; gC[(size_t)w*maxt+t] = C[t];
    }
    gRank[w] = rank; gSince[w] = since; gBestW[w] = bestw; gSeed[w] = seed;
}
} // extern C
'''

def run(n, m, p, target, milestones, seconds, walks, steps_per_launch,
        plus_after, slack, seed, out_prefix, seed_scheme_path=None, climb=6):
    if seed_scheme_path:
        d = json.load(open(seed_scheme_path))
        triv = [tuple(t) for t in d["scheme_bitmasks"]]
        assert verify(triv, n, m, p), "seed scheme failed host verification"
        print(f"seeding {len(triv)}-term verified scheme from {seed_scheme_path}")
    else:
        triv = trivial_scheme(n, m, p)
    rank0 = len(triv)
    maxt = rank0 + 14
    la_bits = n * m

    mod = cp.RawModule(code=KERNEL.replace("MAXT", str(maxt)),
                       options=("-std=c++14",))
    kern = mod.get_function("walk")

    tA = cp.asarray(np.array([t[0] for t in triv], dtype=np.uint64))
    tB = cp.asarray(np.array([t[1] for t in triv], dtype=np.uint64))
    tC = cp.asarray(np.array([t[2] for t in triv], dtype=np.uint64))

    gA = cp.zeros((walks, maxt), dtype=cp.uint64)
    gB = cp.zeros((walks, maxt), dtype=cp.uint64)
    gC = cp.zeros((walks, maxt), dtype=cp.uint64)
    gA[:, :rank0] = tA; gB[:, :rank0] = tB; gC[:, :rank0] = tC
    gRank  = cp.full(walks, rank0, dtype=cp.int32)
    gSince = cp.zeros(walks, dtype=cp.int32)
    gBestW = cp.full(walks, rank0, dtype=cp.int32)
    rs = np.random.RandomState(seed)
    gSeed  = cp.asarray(rs.randint(1, 2**63, size=walks, dtype=np.uint64))

    OUTCAP = 64
    outBuf   = cp.zeros((OUTCAP, maxt, 3), dtype=cp.uint64)
    outRank  = cp.zeros(OUTCAP, dtype=cp.int32)
    outSlots = cp.zeros(1, dtype=cp.int32)
    globalBest = cp.full(1, rank0, dtype=cp.int32)

    restart_at = rank0 + (climb if seed_scheme_path else 0)
    block = 128
    grid = (walks + block - 1) // block
    dump_thresh = max([target] + list(milestones)) if milestones else target

    t0 = time.time()
    total_steps = 0
    best_seen = rank0
    best_verified = {}
    launches = 0
    log = []
    while time.time() - t0 < seconds:
        kern((grid,), (block,),
             (gA, gB, gC, gRank, gSince, gBestW, gSeed, tA, tB, tC,
              np.int32(walks), np.int32(maxt), np.int32(rank0),
              np.int32(steps_per_launch), np.int32(la_bits),
              np.int32(plus_after), np.int32(slack), np.int32(restart_at),
              np.int32(dump_thresh), globalBest,
              outBuf, outRank, outSlots, np.int32(OUTCAP)))
        cp.cuda.Stream.null.synchronize()
        launches += 1
        total_steps += walks * steps_per_launch
        gb = int(globalBest.get()[0])
        slots = min(int(outSlots.get()[0]), OUTCAP)
        if slots:  # verify every dumped scheme exactly on host
            buf = outBuf.get()[:slots]; rks = outRank.get()[:slots]
            outSlots.fill(0)
            for s in range(slots):
                r = int(rks[s])
                if r in best_verified: continue
                sch = [(int(buf[s, t, 0]), int(buf[s, t, 1]), int(buf[s, t, 2]))
                       for t in range(r)]
                ok = verify(sch, n, m, p)
                tag = "VERIFIED" if ok else "REJECTED-BY-HOST"
                print(f"  rank {r}: {tag}")
                if ok:
                    best_verified[r] = sch
                    fn = f"{out_prefix}_rank{r}.json"
                    json.dump({"format": [n, m, p], "rank": r, "field": "F2",
                               "verified_brent_f2": True,
                               "scheme_bitmasks": [[a, b, c] for a, b, c in sch]},
                              open(fn, "w"), indent=1)
        if gb < best_seen:
            best_seen = gb
            el = time.time() - t0
            print(f"[{el:7.1f}s] best rank {gb}  "
                  f"({total_steps/el/1e6:.0f}M steps/s aggregate)")
            log.append({"t": el, "best": gb})
        if best_verified.get(target) is not None:
            print(f"TARGET {target} HIT and verified.")
            break
    dt = time.time() - t0
    return {"format": [n, m, p], "target": target, "best_rank_gpu": best_seen,
            "verified_ranks": sorted(best_verified), "walks": walks,
            "launches": launches, "total_flip_steps": total_steps,
            "steps_per_sec": total_steps / dt, "seconds": dt, "progress": log}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", nargs=3, type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--milestones", nargs="*", type=int, default=[])
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--walks", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--plus-after", type=int, default=20000)
    ap.add_argument("--slack", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-scheme", type=str, default=None)
    ap.add_argument("--climb", type=int, default=6)
    args = ap.parse_args()
    n, m, p = args.fmt
    prefix = f"../results/gpu_{n}{m}{p}"
    res = run(n, m, p, args.target, args.milestones, args.seconds, args.walks,
              args.steps, args.plus_after, args.slack, args.seed, prefix,
              seed_scheme_path=args.seed_scheme, climb=args.climb)
    print(json.dumps({k: v for k, v in res.items() if k != "progress"}, indent=1))
    json.dump(res, open(f"{prefix}_run.json", "w"), indent=1)
    sys.exit(0 if args.target in res["verified_ranks"] else 2)
