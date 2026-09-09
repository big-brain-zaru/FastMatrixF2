"""Phase 1 digest: pool (distinct verified C3-invariant 5x5 rank-93 schemes and their invariants), BFS
layer statistics, exact reduction sweep (rank 92) outcome, GPU walk outcomes. Writes results/p1_dissection.json."""
import glob, json, collections, time, os
import symgroup as SG
import pool_invariants as PI
from flip_graph import verify

R = "../results"
def jl(p):
    out = []
    for f in sorted(glob.glob(os.path.join(R, p))):
        for l in open(f):
            try: out.append(json.loads(l))
            except Exception: pass
    return out
def key(r): return (tuple(r["deleted_orbits"]), tuple(r["refill_sizes"]), tuple(r["refill_subgroup_orders"]))

out = {"generated": time.strftime("%Y-%m-%d %H:%M")}
# ---- pool
rec = [tuple(t) for t in json.load(open(os.path.join(R, "records/lille_555_rank93.json")))["scheme_bitmasks"]]
files = sorted(set(glob.glob(os.path.join(R, "orbitn_5_same_rank93_*.json")) + glob.glob(os.path.join(R, "P1_*_rank93_*.json"))))
G3 = SG.closure([SG.perm_cyc(5)], 5); tau = SG.perm_tau(5)
sets = collections.OrderedDict(); sets[frozenset(rec)] = "record"
for f in files:
    S = frozenset(tuple(t) for t in json.load(open(f))["scheme_bitmasks"])
    sets.setdefault(S, os.path.basename(f))
pool = []
for i, (S, f) in enumerate(sets.items()):
    L = sorted(S)
    row = {"id": i, "file": f, "verified": verify(L, 5, 5, 5), "C3": SG.is_invariant(L, G3, 5), "tau": SG.is_invariant(L, [tau], 5),
           "signature": PI.signature(L), "weights": PI.weights(L), "shared": PI.shared(L), "common_with_record": len(S & frozenset(rec))}
    if i < 0:   # (final run: Z/4 sample already computed in p1_dissect_prelim.log: 10 of 10 liftable) the exact Z/4 lift test is slow in pure Python (minutes per scheme); sample the first 10
        lift, rk, nv, ne = PI.z4_lift(L); row["z4_lift"] = lift
    else:
        row["z4_lift"] = None
    pool.append(row)
sigs = collections.Counter(json.dumps(r["signature"]) for r in pool)
sw = collections.Counter(json.dumps([r["signature"], r["weights"]]) for r in pool)
out["pool"] = {"files": len(files), "distinct": len(pool), "all_verified": all(r["verified"] for r in pool), "all_C3": all(r["C3"] for r in pool),
               "tau_invariant": sum(r["tau"] for r in pool), "z4_tested": sum(1 for r in pool if r["z4_lift"] is not None), "z4_liftable": sum(1 for r in pool if r["z4_lift"]),
               "distinct_signatures": len(sigs), "distinct_signature_weight_classes": len(sw),
               "shared_factor_hist": {"/".join(map(str, k)): v for k, v in collections.Counter(tuple(r["shared"]) for r in pool).most_common()},
               "common_with_record_hist": dict(sorted(collections.Counter(r["common_with_record"] for r in pool).items())), "members": pool}
print("POOL: %d files -> %d distinct schemes; verified %s; C3 %s; tau-invariant %d; Z/4-liftable %d of %d tested; GL signatures %d; (signature,weights) classes %d" % (
    len(files), len(pool), out["pool"]["all_verified"], out["pool"]["all_C3"], out["pool"]["tau_invariant"], out["pool"]["z4_liftable"], out["pool"]["z4_tested"], len(sigs), len(sw)))
print("  shared-factor histogram:", out["pool"]["shared_factor_hist"]); print("  terms in common with record:", out["pool"]["common_with_record_hist"])

# ---- BFS
st = json.load(open(os.path.join(R, "P1_bfs_state.json"))) if os.path.exists(os.path.join(R, "P1_bfs_state.json")) else {"expanded": [], "seeds": {}}
layer = []
for tag, info in st["seeds"].items():
    recs = jl("%s_w*.jsonl" % tag)
    o = collections.Counter(r["results"][-1]["outcome"] for r in recs)
    newf = len(glob.glob(os.path.join(R, "%s_5_same_rank93_*.json" % tag)))
    layer.append({"tag": tag, "seed": os.path.basename(info["file"]), "instances": len(recs), "final_outcomes": dict(o), "new_files": newf, "seconds": info["seconds"]})
out["bfs"] = {"seeds_expanded": len(st["expanded"]), "per_seed": layer, "total_instances": sum(l["instances"] for l in layer),
              "all_terminated_unsat": all(set(l["final_outcomes"]) <= {"unsat"} for l in layer)}
print("BFS: %d seeds expanded, %d instances, every instance terminated UNSAT after enumeration: %s" % (len(st["expanded"]), out["bfs"]["total_instances"], out["bfs"]["all_terminated_unsat"]))
for l in layer: print("   %s <- %s: %d inst, %s, %d new files, %ss" % (l["tag"], l["seed"][:40], l["instances"], l["final_outcomes"], l["new_files"], l["seconds"]))

# ---- exact reduction sweep (rank 92)
red = jl("P1_reduce3_w*.jsonl")
by = {}
for r in red: by.setdefault(key(r), set()).add(r["results"][-1]["outcome"])
oc = collections.Counter(r["results"][-1]["outcome"] for r in red)
ts = sorted(r["results"][-1]["t"] for r in red)
out["reduce3"] = {"instances": len(red), "distinct_keys": len(by), "outcomes": dict(oc), "median_t": ts[len(ts) // 2] if ts else None, "max_t": ts[-1] if ts else None,
                  "solver_hours": round(sum(ts) / 3600, 2), "sat_files": glob.glob(os.path.join(R, "P1_reduce3_5_reduce_rank92_*.json"))}
print("REDUCE3 (rank 92, |D|=3, <=4 refill orbits): %d instances (%d distinct), %s, median %.1fs, max %.0fs, %.1f solver-h, rank-92 files: %d" % (
    len(red), len(by), dict(oc), out["reduce3"]["median_t"] or 0, out["reduce3"]["max_t"] or 0, out["reduce3"]["solver_hours"], len(out["reduce3"]["sat_files"])))

# ---- walks
w = jl("P1_walks.jsonl") + jl("P1_walks2.jsonl")
out["walks"] = {"seeds": len(w), "best_ranks": collections.Counter(x["best_rank"] for x in w), "target_hits": sum(1 for x in w if x["target_hit"]),
                "steps_per_sec_median": sorted(x["steps_per_sec"] for x in w if x["steps_per_sec"])[len(w) // 2] if w else None, "seconds_each": sorted(collections.Counter(x["seconds"] for x in w).items()) if w else None,
                "total_steps": sum((x["steps_per_sec"] or 0) * x["seconds"] for x in w)}
out["walks"]["best_ranks"] = dict(out["walks"]["best_ranks"])
print("WALKS: %d seeds, seconds each %s, best ranks %s, target hits %d, median %.1f M steps/s, total %.2e steps" % (
    len(w), out["walks"]["seconds_each"], out["walks"]["best_ranks"], out["walks"]["target_hits"], (out["walks"]["steps_per_sec_median"] or 0) / 1e6, out["walks"]["total_steps"]))
json.dump(out, open(os.path.join(R, "p1_dissection.json"), "w"), indent=1)
print("saved", os.path.join(R, "p1_dissection.json"))
