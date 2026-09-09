"""Continuation (6 Sep daytime), deadline-driven, corrected pool globs.
  T1  5x5 BFS continuation (6 workers) until DEADLINE
  T2  6x6 BFS continuation (6 workers) until DEADLINE
  T3  6x6 exact |D|<=2 rank-152 reduce around pool members, farthest from the record first (6 workers);
      then |D|=3 around the farthest member with the remaining time
  GPU walk loop over unwalked members (6x6 90 s, 5x5 45 s)
Prints 'CONTINUE DONE'."""
import glob, json, os, sys, threading, time
import overnight as ON
import symgroup as SG
from flip_graph import verify
R = ON.R
DEADLINE = ON.ts("15:45") if time.time() < ON.ts("15:45") else ON.ts("15:45", 1)

def pool(n):
    rank = ON.RANK[n]; G3 = SG.closure([SG.perm_cyc(n)], n)
    pats = {5: ["orbitn_5_same_rank93_*.json", "P1_*_rank93_*.json", "ON5bfs*_5_same_rank93_*.json", "C25bfs*_5_same_rank93_*.json"],
            6: ["ON6bfs*_6_same_rank153_*.json", "C26bfs*_6_same_rank153_*.json"]}[n]
    files = [ON.RECORD[n]] + sorted(set(f for p in pats for f in glob.glob(os.path.join(R, p))))
    out = {}
    for f in files:
        try: S = [tuple(t) for t in json.load(open(f))["scheme_bitmasks"]]
        except Exception: continue
        k = json.dumps(sorted(S))
        if k not in out and len(S) == rank and verify(S, n, n, n) and SG.is_invariant(S, G3, n): out[k] = f
    return out
ON.pool = pool

def far_seeds(n):
    rec = set(tuple(t) for t in json.load(open(ON.RECORD[n]))["scheme_bitmasks"])
    rows = []
    import collections
    for k, f in pool(n).items():
        S = [tuple(t) for t in json.loads(k)]
        if set(S) == rec: continue
        sh = sum(sum(v for v in collections.Counter(t[ax] for t in S).values() if v > 1) for ax in range(3))
        rows.append((len(set(S) & rec), -sh, f))
    rows.sort(); return [f for _, _, f in rows]

def t3():
    seeds = far_seeds(6)
    ON.log("T3: 6x6 |D|<=2 rank-152 reduce around %d members, farthest first" % len(seeds))
    done = ON.reduce_seeds(6, seeds, 1, 2, 6, DEADLINE, "C26red2_")
    ON.log("T3: |D|<=2 done for %d seeds" % len(done))
    if time.time() < DEADLINE - 600 and seeds:
        ON.log("T3: 6x6 |D|=3 rank-152 reduce around the farthest member %s until deadline" % os.path.basename(seeds[0]))
        ON.reduce_seeds(6, seeds[:1], 3, 3, 6, DEADLINE, "C26red3_")

ON.log("CONTINUE START, deadline %s; pools: 5x5 %d, 6x6 %d" % (time.strftime("%H:%M", time.localtime(DEADLINE)), len(pool(5)), len(pool(6))))
threads = [threading.Thread(target=lambda: ON.log("T1 5x5 BFS: %s" % ON.bfs(5, 6, DEADLINE, os.path.join(R, "P1_bfs_state.json"), "C25bfs", max_seeds=10000)), daemon=True),
           threading.Thread(target=lambda: ON.log("T2 6x6 BFS: %s" % ON.bfs(6, 6, DEADLINE, os.path.join(R, "ON6_bfs_state.json"), "C26bfs", max_seeds=10000)), daemon=True),
           threading.Thread(target=t3, daemon=True),
           threading.Thread(target=ON.gpu_loop, args=(DEADLINE,), daemon=True)]
for t in threads: t.start()
for t in threads[:3]: t.join()
while time.time() < DEADLINE and threads[3].is_alive(): time.sleep(10)
ON.log("CONTINUE DONE")
