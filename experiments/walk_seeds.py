"""Phase 1 step 4: sequential GPU flip walks from every distinct verified 5x5 rank-93 seed, target 92.
Seeds = the record + one file per distinct scheme set among results/orbitn_5_same_rank93_*.json and
results/P1_same*_rank93_*.json (whatever exists at launch). Logs results/P1_walks.jsonl."""
import glob, json, subprocess, sys, time, re
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 600
pattern = sys.argv[2] if len(sys.argv) > 2 else None
logpath = sys.argv[3] if len(sys.argv) > 3 else "../results/P1_walks.jsonl"
N = sys.argv[4] if len(sys.argv) > 4 else "5"
TARGET = {"5": "92", "6": "152"}[N]
# distinct schemes already walked (by term set), from every walk log
walked = set()
for lg in glob.glob("../results/*walks*.jsonl"):
    for l in open(lg):
        try:
            f = json.loads(l)["seed"]; walked.add(frozenset(tuple(t) for t in json.load(open(f))["scheme_bitmasks"]))
        except Exception: pass
seeds = [] if pattern else ["../results/records/lille_555_rank93.json"]
seen = set(walked)
files = sorted(set(glob.glob(pattern))) if pattern else sorted(set(glob.glob("../results/orbitn_5_same_rank93_*.json")))
for f in files:
    S = frozenset(tuple(t) for t in json.load(open(f))["scheme_bitmasks"])
    if S in seen: continue
    seen.add(S); seeds.append(f)
log = open(logpath, "a")
print("seeds:", len(seeds), "seconds each:", secs, flush=True)
for f in seeds:
    t0 = time.time()
    p = subprocess.run([sys.executable, "gpu_walker.py", "--fmt", N, N, N, "--target", TARGET, "--seconds", str(secs), "--walks", "8192",
                        "--steps", "2000", "--seed", "11", "--plus-after", "3000", "--slack", "6", "--climb", "6", "--seed-scheme", f],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    best = re.findall(r"best_rank_gpu[^0-9]*(\d+)", out); sps = re.findall(r"steps_per_sec[^0-9]*([0-9.]+)", out)
    rec = {"seed": f, "seconds": secs, "best_rank": int(best[-1]) if best else None, "steps_per_sec": float(sps[-1]) if sps else None,
           "verified_hits": len(re.findall(r"VERIFIED", out)), "target_hit": "TARGET" in out, "wall": round(time.time() - t0, 1)}
    log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
print("done", flush=True)
