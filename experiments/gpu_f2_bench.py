"""Size a GPU-batched flip walker: bitsliced GF(2) throughput on the RTX 5070."""
import cupy as cp, time, json

xp = cp.ElementwiseKernel('uint64 x, uint64 y', 'uint64 z', 'z = __popcll(x ^ y)', 'xorpop')
N = 1 << 24  # 16M uint64 = 128 MB per array
a = cp.random.randint(0, 2**63, N, dtype=cp.uint64)
b = cp.random.randint(0, 2**63, N, dtype=cp.uint64)
xp(a, b); cp.cuda.Stream.null.synchronize()
t = time.time(); R = 50
for _ in range(R): xp(a, b)
cp.cuda.Stream.null.synchronize()
dt = (time.time() - t) / R
bitops = N * 64 / dt
# simulated batched flip step: gather 2 terms x 3 factors for 1M parallel walks, XOR, scatter
W = 1_000_000
terms = cp.random.randint(0, 2**48, (W, 64, 3), dtype=cp.uint64)  # 64 terms/walk, 3 factors
i = cp.random.randint(0, 64, W); j = cp.random.randint(0, 64, W)
w = cp.arange(W)
cp.cuda.Stream.null.synchronize(); t = time.time(); R2 = 20
for _ in range(R2):
    ti = terms[w, i]; tj = terms[w, j]
    terms[w, i, 2] = ti[:, 2] ^ tj[:, 2]
    terms[w, j, 1] = ti[:, 1] ^ tj[:, 1]
cp.cuda.Stream.null.synchronize()
flip_rate = W * R2 / (time.time() - t) * R2 / R2
out = {"xor_popcount_bitops_per_sec": bitops, "batched_flip_steps_per_sec_1M_walks": flip_rate,
       "cpu_python_flips_per_sec_measured": 234176}
print(json.dumps(out, indent=1))
json.dump(out, open("../results/gpu_f2_bench.json", "w"), indent=1)
