"""Digest of the filtered census run B2: filter verdicts + full-SAT outcomes, per group and overall.
Writes results/b2_dissection.json and prints a markdown summary."""
import json, glob, collections, time

wl = json.load(open("../results/B2_worklist.json"))
S = {r["group_index"]: r for r in wl["summary"]}
filt = {}
for l in open("../results/B2_filter.jsonl"):   # later records (post-hoc tightening) supersede earlier ones
    r = json.loads(l); filt[(r["group_index"], tuple(r["class_key"]))] = r
filt = list(filt.values())
runs = []
for f in sorted(glob.glob("../results/B2_w*.jsonl")):
    runs += [json.loads(l) for l in open(f)]
# dedupe runs by faithful key (group, class key)
seen = {}
for r in runs:
    seen[(r["group_index"], tuple(r["class_key"]))] = r
runs = list(seen.values())

out = collections.Counter(r["outcome"] for r in runs)
tot_t = sum(r["t"] for r in runs)
print("## B2 full-SAT outcomes: %d instances decided-or-budgeted, %s, %.1f solver-hours" % (len(runs), dict(out), tot_t / 3600))
by_out_t = collections.defaultdict(list)
for r in runs:
    by_out_t[r["outcome"]].append(r["t"])
for k, v in by_out_t.items():
    v.sort(); print("  %s: n=%d, median %.0fs, min %.0fs, max %.0fs" % (k, len(v), v[len(v) // 2], v[0], v[-1]))

# filter tallies
fc = collections.Counter(r["filter"] for r in filt)
print("## filter verdicts over %d multisets: %s" % (len(filt), dict(fc)))

# per group table
rows = []
for gi in sorted(S):
    s = S[gi]
    rr = [r for r in runs if r["group_index"] == gi]
    c = collections.Counter(r["outcome"] for r in rr)
    reach = bool(s["fixed_coords"]) and s["fixed_coords"][0] > 0
    rows.append({"group": gi, "order": s["order"], "reachable": reach, "fixed_coords": s["fixed_coords"][0] if s["fixed_coords"] else None,
                 "count_bound": s.get("count_bound_tightened", s["count_bound"]), "multisets": s["multisets"],
                 "filter_eliminated": s["count"] + s["unsat"] + s.get("count7", 0), "survivors_run": len(rr),
                 "unsat": c.get("unsat", 0), "sat": c.get("sat", 0), "budget": c.get("budget", 0)})
print("\n| group | order | reachable | fixed coords | bound | multisets | filter-eliminated | run | UNSAT | SAT | budget |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| %d | %d | %s | %s | %d | %d | %d | %d | %d | %d | %d |" % (
        r["group"], r["order"], "yes" if r["reachable"] else "no", r["fixed_coords"], r["count_bound"], r["multisets"],
        r["filter_eliminated"], r["survivors_run"], r["unsat"], r["sat"], r["budget"]))

# exact statements: per group, how many leading multisets (fewest slots first) are fully decided UNSAT
# (filter-eliminated or full UNSAT) with no budget among them
work_pos = {(w["group_index"], tuple(w["class_key"])): w["pos_in_group"] for w in wl["work"]}
print("\n## coverage: per group, the leading surviving multisets are decided (UNSAT) up to position:")
cov = {}
for gi in sorted(S):
    rr = sorted([r for r in runs if r["group_index"] == gi], key=lambda r: work_pos[(gi, tuple(r["class_key"]))])
    k = 0
    for r in rr:
        if work_pos[(gi, tuple(r["class_key"]))] != k: break
        if r["outcome"] != "unsat": break
        k += 1
    cov[gi] = k
print(dict(cov))
print("groups with >=1 leading survivor refuted:", sum(1 for v in cov.values() if v > 0), "of", len(cov))

sats = [r for r in runs if r["outcome"] == "sat"]
print("\n## SAT decodes: %d (verified: %d)" % (len(sats), sum(1 for r in sats if r.get("verified"))))
for r in sats: print("  ", r)

json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "outcomes": dict(out), "solver_hours": round(tot_t / 3600, 2),
           "filter_verdicts": dict(fc), "per_group": rows, "leading_coverage": cov, "sat": sats},
          open("../results/b2_dissection.json", "w"), indent=1)
print("\nsaved ../results/b2_dissection.json")
