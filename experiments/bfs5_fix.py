"""Corrected 5x5 BFS continuation: overnight.pool() missed its own ON5bfsNN_5_same_rank93_*.json files
(glob 'ON5_*'), so Stage A only expanded the original 65 members and bfs5_finish would have done the same.
Correct glob; runs until exhaustion or 09:00 (12 workers)."""
import glob, json, os, time
import overnight as ON
import symgroup as SG
from flip_graph import verify
R = ON.R
_orig = ON.pool
def pool5():
    G3 = SG.closure([SG.perm_cyc(5)], 5)
    files = [ON.RECORD[5]] + sorted(set(glob.glob(os.path.join(R, "orbitn_5_same_rank93_*.json")) + glob.glob(os.path.join(R, "P1_*_rank93_*.json")) + glob.glob(os.path.join(R, "ON5bfs*_5_same_rank93_*.json"))))
    out = {}
    for f in files:
        S = [tuple(t) for t in json.load(open(f))["scheme_bitmasks"]]
        k = json.dumps(sorted(S))
        if k not in out and len(S) == 93 and verify(S, 5, 5, 5) and SG.is_invariant(S, G3, 5): out[k] = f
    return out
ON.pool = lambda n: pool5() if n == 5 else _orig(n)
ON.log("BFS5-FIX start: corrected pool has %d distinct 5x5 rank-93 schemes" % len(pool5()))
res = ON.bfs(5, 12, ON.ts("09:00") if time.time() < ON.ts("09:00") else ON.ts("09:00", 1), os.path.join(R, "P1_bfs_state.json"), "ON5bfs")
ON.log("BFS5-FIX done: %s" % res)
