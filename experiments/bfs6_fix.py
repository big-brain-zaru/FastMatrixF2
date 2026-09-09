"""Corrected 6x6 BFS continuation: overnight.pool() missed files named ON6bfsNN_6_same_rank153_*.json
(glob 'ON6_*'). Same logic with the glob fixed; runs until the Stage-C deadline (06:00)."""
import glob, json, os, sys, time
import overnight as ON
import symgroup as SG
from flip_graph import verify
R = ON.R
def pool6():
    G3 = SG.closure([SG.perm_cyc(6)], 6)
    files = [ON.RECORD[6]] + sorted(set(glob.glob(os.path.join(R, "ON6bfs*_6_same_rank153_*.json"))))
    out = {}
    for f in files:
        S = [tuple(t) for t in json.load(open(f))["scheme_bitmasks"]]
        k = json.dumps(sorted(S))
        if k not in out and len(S) == 153 and verify(S, 6, 6, 6) and SG.is_invariant(S, G3, 6): out[k] = f
    return out
ON.pool = lambda n: pool6() if n == 6 else ON.__dict__["pool"](n)
_orig_pool = ON.pool
def pool(n):
    return pool6() if n == 6 else _orig_pool(n)
ON.pool = pool
ON.log("BFS6-FIX start: corrected pool has %d distinct 6x6 rank-153 schemes" % len(pool6()))
res = ON.bfs(6, 12, ON.dl("06:00") if time.time() < ON.ts("06:00") else ON.ts("06:00", 1), os.path.join(R, "ON6_bfs_state.json"), "ON6bfs")
ON.log("BFS6-FIX done: %s" % res)
