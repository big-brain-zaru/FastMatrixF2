"""Finish the 5x5 family expansion: expand every remaining unexpanded pool member (exact |D|<=2 same-rank
sweep, 12 workers) until exhausted or 09:00, using overnight.bfs with the P1 state file."""
import os, time
import overnight as ON
R = ON.R
ON.log("BFS5-FINISH start")
res = ON.bfs(5, 12, ON.ts("09:00") if time.time() < ON.ts("09:00") else ON.ts("09:00", 1), os.path.join(R, "P1_bfs_state.json"), "ON5bfs")
ON.log("BFS5-FINISH done: %s" % res)
