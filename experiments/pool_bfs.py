"""Phase 1 pool growth by breadth-first expansion: for each distinct verified 5x5 rank-93 scheme not yet
expanded, run the exact |D|<=2 same-rank orbit sweep around it (12 workers), collect new distinct schemes,
repeat until no seed is left or --max-seeds reached. State: results/P1_bfs_state.json"""
import glob, json, os, subprocess, sys, time
import symgroup as SG
from flip_graph import verify
W = 12; MAXSEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
st_path = "../results/P1_bfs_state.json"
st = json.load(open(st_path)) if os.path.exists(st_path) else {"expanded": [], "seeds": {}}
def key(S): return json.dumps(sorted(S))
def collect():
    """all distinct verified C3-invariant 93s known so far -> {key: file}"""
    G3 = SG.closure([SG.perm_cyc(5)], 5)
    files = ["../results/records/lille_555_rank93.json"] + sorted(set(glob.glob("../results/orbitn_5_same_rank93_*.json") + glob.glob("../results/P1_*_rank93_*.json")))
    out = {}
    for f in files:
        S = [tuple(t) for t in json.load(open(f))["scheme_bitmasks"]]
        k = key(S)
        if k not in out and verify(S, 5, 5, 5) and SG.is_invariant(S, G3, 5): out[k] = f
    return out
log = open("../results/P1_bfs.log", "a")
while len(st["expanded"]) < MAXSEEDS:
    pool = collect()
    todo = [k for k in pool if k not in st["expanded"]]
    msg = "[%s] pool %d distinct, expanded %d, todo %d" % (time.strftime("%H:%M:%S"), len(pool), len(st["expanded"]), len(todo))
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if not todo: break
    k = todo[0]; f = pool[k]; sid = len(st["expanded"])
    tag = "P1_bfs%02d" % sid
    procs = [subprocess.Popen([sys.executable, "-u", "orbit_lns_n.py", "--scheme", f, "--n", "5", "--group", "cyc", "--mode", "same", "--min-delete", "1", "--max-delete", "2", "--max-new", "4", "--distinct", "50", "--conf", "2000000", "--tag", tag, "--stride", str(W), "--offset", str(o), "--log", "../results/%s_w%d.jsonl" % (tag, o)],
                              stdout=open("../results/%s_w%d.log" % (tag, o), "w"), stderr=subprocess.STDOUT) for o in range(W)]
    t0 = time.time()
    for p in procs: p.wait()
    st["expanded"].append(k); st["seeds"][tag] = {"file": f, "seconds": round(time.time() - t0)}
    json.dump(st, open(st_path, "w"), indent=1)
    new = len(glob.glob("../results/%s_5_same_rank93_*.json" % tag))
    msg = "[%s] seed %s (%s) expanded in %.0fs -> %d new files" % (time.strftime("%H:%M:%S"), tag, os.path.basename(f), time.time() - t0, new)
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
print("bfs done", flush=True)
