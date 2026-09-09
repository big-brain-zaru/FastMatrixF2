"""Overnight digest: 5x5 family (final), 5x5 reduce sweeps (|D|<=2 around all members; |D|=3 around far
seeds), 6x6 Phase 2 (pool, expansion, reduce, walks). Writes results/on_dissection.json."""
import glob, json, collections, os, time
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
def sig(S, n):
    def f2rank(m):
        rows = [(m >> (n * i)) & ((1 << n) - 1) for i in range(n)]; r = 0
        while rows:
            p = rows.pop()
            if not p: continue
            r += 1; hb = p.bit_length() - 1; rows = [x ^ p if (x >> hb) & 1 else x for x in rows if x]
        return r
    return json.dumps([sorted(collections.Counter(f2rank(t[ax]) for t in S).items()) for ax in range(3)])
def wts(S): return json.dumps(sorted(collections.Counter(bin(t[0]).count("1") + bin(t[1]).count("1") + bin(t[2]).count("1") for t in S).items()))
def shared(S): return tuple(sum(v for v in collections.Counter(t[ax] for t in S).values() if v > 1) for ax in range(3))

out = {"generated": time.strftime("%Y-%m-%d %H:%M")}
for n, rank, recf, pats, tagbfs in [(5, 93, "records/lille_555_rank93.json", ["orbitn_5_same_rank93_*.json", "P1_*_rank93_*.json", "ON5bfs*_rank93_*.json", "C25bfs*_rank93_*.json"], "ON5bfs"),
                                     (6, 153, "records/lille_666_rank153.json", ["ON6bfs*_rank153_*.json", "C26bfs*_rank153_*.json"], "ON6bfs")]:
    rec = [tuple(t) for t in json.load(open(os.path.join(R, recf)))["scheme_bitmasks"]]
    files = sorted(set(f for p in pats for f in glob.glob(os.path.join(R, p))))
    G3 = SG.closure([SG.perm_cyc(n)], n)
    sets = collections.OrderedDict(); sets[frozenset(rec)] = "record"
    for f in files:
        S = frozenset(tuple(t) for t in json.load(open(f))["scheme_bitmasks"]); sets.setdefault(S, os.path.basename(f))
    members = []
    for S, f in sets.items():
        L = sorted(S); members.append({"file": f, "verified": verify(L, n, n, n), "C3": SG.is_invariant(L, G3, n), "sig": sig(L, n), "wts": wts(L), "shared": shared(L), "common": len(S & frozenset(rec))})
    st_path = os.path.join(R, "P1_bfs_state.json" if n == 5 else "ON6_bfs_state.json")
    st = json.load(open(st_path)) if os.path.exists(st_path) else {"expanded": [], "seeds": {}}
    exp_files = [v["file"] for v in st["seeds"].values()]
    bfs_recs = jl("P1_bfs*_w*.jsonl") + jl("%s*_w*.jsonl" % tagbfs) if n == 5 else jl("%s*_w*.jsonl" % tagbfs)
    # only count instances belonging to completed expansions: tags in state
    tags = set(st["seeds"].keys())
    def tag_of(f): return os.path.basename(f).split("_w")[0]
    recs_done = []
    for f in sorted(set(glob.glob(os.path.join(R, "P1_bfs*_w*.jsonl")) + glob.glob(os.path.join(R, "%s*_w*.jsonl" % tagbfs)) + glob.glob(os.path.join(R, "C2%dbfs*_w*.jsonl" % n)))):
        if tag_of(f) in tags: recs_done += [json.loads(l) for l in open(f) if l.strip()]
    fam = {"distinct": len(members), "all_verified": all(m["verified"] for m in members), "all_C3": all(m["C3"] for m in members),
           "expanded": len(st["expanded"]), "unexpanded": len(members) - len(st["expanded"]) if all(m["verified"] for m in members) else None,
           "expansion_instances": len(recs_done), "expansion_all_terminated_unsat": all(r["results"][-1]["outcome"] == "unsat" for r in recs_done) if recs_done else None,
           "signatures": len({m["sig"] for m in members}), "sig_weight_classes": len({(m["sig"], m["wts"]) for m in members}),
           "shared_hist": {"/".join(map(str, k)): v for k, v in collections.Counter(m["shared"] for m in members).most_common()},
           "common_hist": dict(sorted(collections.Counter(m["common"] for m in members).items()))}
    out["family_%d" % n] = fam
    print("FAMILY n=%d rank %d: %d distinct (verified %s, C3 %s); expanded %d, unexpanded %d; expansion instances %d, all terminated UNSAT: %s; signatures %d; sig+weight classes %d" % (
        n, rank, fam["distinct"], fam["all_verified"], fam["all_C3"], fam["expanded"], fam["unexpanded"], fam["expansion_instances"], fam["expansion_all_terminated_unsat"], fam["signatures"], fam["sig_weight_classes"]))
    print("   shared factors:", fam["shared_hist"]); print("   terms in common with record:", fam["common_hist"])

# reduce sweeps
def sweep_stats(pattern, label):
    rows = {}
    for f in sorted(glob.glob(os.path.join(R, pattern))):
        tag = os.path.basename(f).split("_w")[0]
        rs = [json.loads(l) for l in open(f) if l.strip()]
        d = rows.setdefault(tag, {"instances": 0, "outcomes": collections.Counter(), "t": 0.0})
        d["instances"] += len(rs); d["t"] += sum(r["results"][-1]["t"] for r in rs)
        for r in rs: d["outcomes"][r["results"][-1]["outcome"]] += 1
    tot = sum(d["instances"] for d in rows.values()); oc = collections.Counter()
    for d in rows.values(): oc.update(d["outcomes"])
    res = {"seeds": len(rows), "instances": tot, "outcomes": dict(oc), "solver_hours": round(sum(d["t"] for d in rows.values()) / 3600, 2),
           "per_seed": {k: {"instances": v["instances"], "outcomes": dict(v["outcomes"])} for k, v in rows.items()}}
    print("%s: %d seeds, %d instances, %s, %.1f solver-h" % (label, len(rows), tot, dict(oc), res["solver_hours"]))
    return res
out["reduce5_D2_all"] = sweep_stats("ON5red2_*_w*.jsonl", "5x5 |D|<=2 rank-92 reduce around pool members")
out["reduce5_D3_far"] = sweep_stats("ON5red3_*_w*.jsonl", "5x5 |D|=3 rank-92 reduce around far seeds")
out["reduce6_D2"] = sweep_stats("*6red2_*_w*.jsonl", "6x6 |D|<=2 rank-152 reduce")
out["reduce6_D3_far"] = sweep_stats("C26red3_*_w*.jsonl", "6x6 |D|=3 rank-152 reduce around the farthest member")
out["rank92_files"] = glob.glob(os.path.join(R, "ON5red*_rank92_*.json")); out["rank152_files"] = glob.glob(os.path.join(R, "*6red*_rank152_*.json"))
print("rank-92 files:", len(out["rank92_files"]), "rank-152 files:", len(out["rank152_files"]))
# walks
w = jl("ON_walks*.jsonl")
by_n = {}
for x in w: by_n.setdefault(x.get("n", 6 if "rank153" in x["seed"] else 5), []).append(x)
out["walks"] = {}
for n, xs in sorted(by_n.items()):
    out["walks"][n] = {"walks": len(xs), "best_ranks": dict(collections.Counter(x["best_rank"] for x in xs)), "hits": sum(1 for x in xs if x["target_hit"]),
                       "median_msteps": round(sorted(x["steps_per_sec"] for x in xs if x["steps_per_sec"])[len(xs) // 2] / 1e6, 1), "total_steps": sum((x["steps_per_sec"] or 0) * x["seconds"] for x in xs)}
    print("WALKS n=%d: %s" % (n, out["walks"][n]))
json.dump(out, open(os.path.join(R, "on_dissection.json"), "w"), indent=1)
print("saved on_dissection.json")
