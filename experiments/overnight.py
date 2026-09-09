"""Overnight programme (5-6 Sep 2026), deadline-driven, self-terminating.

Stage A  5x5 BFS continuation: expand every unexpanded pool member (exact |D|<=2 same-rank sweep, 12 workers)
         until the family is exhausted or the stage deadline.
Stage B  5x5 exact |D|<=2 REDUCE (rank 92) around every pool member (6 workers, sequential seeds).
Stage C  6x6 Phase 2: same-rank BFS around the rank-153 record (12 workers) until its deadline.
Stage D  5x5 |D|=3 REDUCE around the pool members farthest from the record (6 workers), after B.
Stage E  6x6 exact |D|<=2 REDUCE (rank 152) around every 6x6 pool member found (6 workers), after D.
GPU      walk loop over every unwalked 5x5 (120 s) and 6x6 (300 s) pool member until the global deadline.

Every subprocess group is killed at its deadline; state is on disk (jsonl logs, scheme files, bfs state).
Prints 'OVERNIGHT DONE' at the end. Digest/report are done by the session afterwards."""
import glob, json, os, subprocess, sys, threading, time
import symgroup as SG
from flip_graph import verify

R = "../results"
LOG = open(os.path.join(R, "overnight.log"), "a")
def log(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()

def ts(hhmm, day_offset=0):
    """absolute epoch for today/tomorrow HH:MM"""
    t = time.localtime(); base = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
    h, m = map(int, hhmm.split(":")); return base + day_offset * 86400 + h * 3600 + m * 60

NOW = time.time()
def dl(hhmm):   # deadline this night: if the time is already past, it means tomorrow
    x = ts(hhmm); return x if x > NOW else ts(hhmm, 1)
DEADLINE_A = dl("03:00"); DEADLINE_B = dl("03:00"); DEADLINE_C = dl("06:00"); DEADLINE_D = dl("06:00"); DEADLINE_E = dl("06:30"); DEADLINE_GPU = dl("06:30")
RECORD = {5: os.path.join(R, "records/lille_555_rank93.json"), 6: os.path.join(R, "records/lille_666_rank153.json")}
RANK = {5: 93, 6: 153}

def run_sweep(n, scheme, mode, dmin, dmax, tag, W, deadline, conf=2000000, distinct=50):
    procs = []
    for o in range(W):
        args = [sys.executable, "-u", "orbit_lns_n.py", "--scheme", scheme, "--n", str(n), "--group", "cyc", "--mode", mode, "--min-delete", str(dmin), "--max-delete", str(dmax),
                "--max-new", "4", "--distinct", str(distinct), "--conf", str(conf), "--tag", tag, "--stride", str(W), "--offset", str(o), "--log", os.path.join(R, "%s_w%d.jsonl" % (tag, o))]
        procs.append(subprocess.Popen(args, stdout=open(os.path.join(R, "%s_w%d.log" % (tag, o)), "a"), stderr=subprocess.STDOUT))
    t0 = time.time(); killed = False
    while any(p.poll() is None for p in procs):
        if time.time() > deadline:
            for p in procs:
                if p.poll() is None: p.terminate()
            killed = True; break
        time.sleep(5)
    for p in procs:
        try: p.wait(10)
        except Exception: pass
    return {"seconds": round(time.time() - t0), "killed_at_deadline": killed}

def pool(n):
    """distinct verified C3-invariant schemes of the record rank: {key: file}"""
    G3 = SG.closure([SG.perm_cyc(n)], n)
    pats = [RECORD[n]] + sorted(set(glob.glob(os.path.join(R, "orbitn_%d_same_rank%d_*.json" % (n, RANK[n]))) + glob.glob(os.path.join(R, "P1_*_rank%d_*.json" % RANK[n])) + glob.glob(os.path.join(R, "ON%d_*_rank%d_*.json" % (n, RANK[n])))))
    out = {}
    for f in pats:
        try: S = [tuple(t) for t in json.load(open(f))["scheme_bitmasks"]]
        except Exception: continue
        if len(S) != RANK[n]: continue
        k = json.dumps(sorted(S))
        if k not in out and verify(S, n, n, n) and SG.is_invariant(S, G3, n): out[k] = f
    return out

def bfs(n, W, deadline, state_path, tagprefix, max_seeds=500):
    st = json.load(open(state_path)) if os.path.exists(state_path) else {"expanded": [], "seeds": {}}
    # 5x5: the Phase-1 state already lists 20 expanded seeds
    while len(st["expanded"]) < max_seeds and time.time() < deadline - 120:
        P = pool(n); todo = [k for k in P if k not in st["expanded"]]
        log("BFS n=%d: pool %d distinct, expanded %d, todo %d" % (n, len(P), len(st["expanded"]), len(todo)))
        if not todo:
            log("BFS n=%d: FAMILY EXHAUSTED under 2-orbit same-rank rewrites (every member expanded)" % n); return "exhausted"
        k = todo[0]; f = P[k]; tag = "%s%02d" % (tagprefix, len(st["expanded"]))
        r = run_sweep(n, f, "same", 1, 2, tag, W, deadline)
        if r["killed_at_deadline"]:
            log("BFS n=%d: seed %s killed at deadline (partial, not counted as expanded)" % (n, tag)); return "deadline"
        st["expanded"].append(k); st["seeds"][tag] = {"file": f, "seconds": r["seconds"]}
        json.dump(st, open(state_path, "w"), indent=1)
        log("BFS n=%d: seed %s (%s) expanded in %ds -> %d files" % (n, tag, os.path.basename(f), r["seconds"], len(glob.glob(os.path.join(R, "%s_%d_same_rank%d_*.json" % (tag, n, RANK[n]))))))
    return "cap" if time.time() < deadline else "deadline"

def reduce_seeds(n, seeds, dmin, dmax, W, deadline, tagprefix):
    done = []
    for i, f in enumerate(seeds):
        if time.time() > deadline - 60: break
        tag = "%s%02d" % (tagprefix, i)
        r = run_sweep(n, f, "reduce", dmin, dmax, tag, W, deadline)
        sat = len(glob.glob(os.path.join(R, "%s_%d_reduce_rank%d_*.json" % (tag, n, RANK[n] - 1))))
        log("REDUCE n=%d |D|=%d..%d seed %s (%s): %ds, killed=%s, rank-%d files: %d" % (n, dmin, dmax, tag, os.path.basename(f), r["seconds"], r["killed_at_deadline"], RANK[n] - 1, sat))
        done.append(tag)
        if r["killed_at_deadline"]: break
    return done

def far_seeds(n, k):
    """pool members ordered by fewest terms in common with the record (farthest first), then most shared factors"""
    rec = set(tuple(t) for t in json.load(open(RECORD[n]))["scheme_bitmasks"])
    P = pool(n); rows = []
    for key, f in P.items():
        S = [tuple(t) for t in json.loads(key)]
        if set(S) == rec: continue
        import collections
        sh = sum(sum(v for v in collections.Counter(t[ax] for t in S).values() if v > 1) for ax in range(3))
        rows.append((len(set(S) & rec), -sh, f))
    rows.sort(); return [f for _, _, f in rows[:k]]

def gpu_loop(deadline):
    walked = set()
    for lg in glob.glob(os.path.join(R, "*walks*.jsonl")):
        for l in open(lg):
            try: walked.add(json.loads(l)["seed_key"] if "seed_key" in l else json.dumps(sorted(tuple(t) for t in json.load(open(json.loads(l)["seed"]))["scheme_bitmasks"])))
            except Exception: pass
    out = open(os.path.join(R, "ON_walks.jsonl"), "a")
    import re
    while time.time() < deadline - 60:
        cand = []
        for n, secs in ((6, 300), (5, 120)):
            for k, f in pool(n).items():
                if k not in walked: cand.append((n, secs, k, f))
        if not cand:
            time.sleep(120); continue
        n, secs, k, f = cand[0]
        secs = min(secs, max(30, int(deadline - time.time() - 90)))
        t0 = time.time()
        p = subprocess.run([sys.executable, "gpu_walker.py", "--fmt", str(n), str(n), str(n), "--target", str(RANK[n] - 1), "--seconds", str(secs), "--walks", "8192", "--steps", "2000", "--seed", "11", "--plus-after", "3000", "--slack", "6", "--climb", "6", "--seed-scheme", f], capture_output=True, text=True)
        o = p.stdout + p.stderr
        best = re.findall(r"best_rank_gpu[^0-9]*(\d+)", o); sps = re.findall(r"steps_per_sec[^0-9]*([0-9.]+)", o)
        rec = {"n": n, "seed": f, "seed_key": k, "seconds": secs, "best_rank": int(best[-1]) if best else None, "steps_per_sec": float(sps[-1]) if sps else None, "target_hit": "TARGET" in o, "verified_hits": len(re.findall(r"VERIFIED", o)), "wall": round(time.time() - t0, 1), "rc": p.returncode}
        out.write(json.dumps(rec) + "\n"); out.flush(); walked.add(k)
        log("GPU walk n=%d %s: best %s, %s M steps/s%s" % (n, os.path.basename(f), rec["best_rank"], round((rec["steps_per_sec"] or 0) / 1e6, 1), " *** TARGET HIT ***" if rec["target_hit"] else ""))

def main():
    log("OVERNIGHT START; deadlines A/B %s, C/D %s, E/GPU %s" % (time.strftime("%H:%M", time.localtime(DEADLINE_A)), time.strftime("%H:%M", time.localtime(DEADLINE_C)), time.strftime("%H:%M", time.localtime(DEADLINE_E))))
    gpu = threading.Thread(target=gpu_loop, args=(DEADLINE_GPU,), daemon=True); gpu.start()
    # Stage A and B in parallel (12 + 6 workers)
    resA = {}
    tA = threading.Thread(target=lambda: resA.update(r=bfs(5, 12, DEADLINE_A, os.path.join(R, "P1_bfs_state.json"), "ON5bfs")), daemon=True); tA.start()
    P5 = pool(5); seeds5 = [f for k, f in P5.items() if f != RECORD[5]]
    log("Stage B: 5x5 |D|<=2 reduce around %d pool members" % len(seeds5))
    doneB = reduce_seeds(5, seeds5, 1, 2, 6, DEADLINE_B, "ON5red2_")
    log("Stage B done: %d seeds swept" % len(doneB))
    tA.join(); log("Stage A done: %s" % resA.get("r"))
    # Stage C (6x6 BFS, 12 workers) and Stage D (5x5 |D|=3 reduce on far seeds, 6 workers) in parallel
    resC = {}
    tC = threading.Thread(target=lambda: resC.update(r=bfs(6, 12, DEADLINE_C, os.path.join(R, "ON6_bfs_state.json"), "ON6bfs")), daemon=True); tC.start()
    far = far_seeds(5, 4); log("Stage D: 5x5 |D|=3 reduce around %d far seeds: %s" % (len(far), [os.path.basename(f) for f in far]))
    doneD = reduce_seeds(5, far, 3, 3, 6, DEADLINE_D, "ON5red3_")
    log("Stage D done: %d seeds" % len(doneD))
    # Stage E: 6x6 |D|<=2 reduce around every 6x6 pool member found so far (6 workers) until 06:30
    P6 = pool(6); log("Stage E: 6x6 |D|<=2 reduce around %d pool members" % len(P6))
    doneE = reduce_seeds(6, list(P6.values()), 1, 2, 6, DEADLINE_E, "ON6red2_")
    log("Stage E done: %d seeds" % len(doneE))
    tC.join(); log("Stage C done: %s" % resC.get("r"))
    while time.time() < DEADLINE_GPU and gpu.is_alive(): time.sleep(10)
    log("OVERNIGHT DONE")

if __name__ == "__main__":
    main()
