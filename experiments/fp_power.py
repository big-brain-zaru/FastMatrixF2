"""Measure the fixed-point filter on the actual census: (a) the three instances already run in B,
(b) up to --limit class-multisets per group for orders >= --min-order."""
import json, time, argparse, sys
import symgroup as SG, fp_filter as FP
from census_run import bounded_slots
from flip_graph import target_tensor
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=300); ap.add_argument("--min-order", type=int, default=6)
ap.add_argument("--out", default="../results/fp_power.json"); a = ap.parse_args()
n = 4; T = set(target_tensor(n, n, n))
census = json.load(open("../results/census_n4_r46_cyclic.json"))
groups = [w for w in census["worklist"] if "elements" in w]
print("(a) the three B instances already run")
for gi, mi, prev in [(6, 0, "unsat 7.1s"), (2, 1, "budget 664s"), (2, 2, "budget 1019s")]:
    G = [tuple(g) for g in groups[gi]["elements"]]
    slots = bounded_slots(G, n, 46, 24, 3)[mi]
    v, dt, st = FP.run_filter(n, G, slots, T)
    print("  group %d m%d sizes %s: filter=%s (%.2fs) [full run: %s] surv=%s" % (
        gi, mi, [len(G)//len(H) for H in slots], v, dt, prev, [s["surviving_terms"] for s in st["per_involution"]]))
print("(b) power sweep")
rows = []; t0 = time.time()
for gi, w in enumerate(groups):
    if w["order"] < a.min_order: continue
    G = [tuple(g) for g in w["elements"]]
    invs = FP.involution_classes(G)
    subs = FP.all_subgroups(G, n); classes = FP.subgroup_classes(G, subs)
    cands = FP.slots_by_class(G, n, 46, 46, a.limit, classes)
    cnt = {"unsat": 0, "pass": 0, "budget": 0}; tt = 0.0
    for slots, key in cands:
        v, dt, st = FP.run_filter(n, G, slots, T); cnt[v] += 1; tt += dt
    rows.append({"group_index": gi, "order": w["order"], "involution_classes": len(invs), "subgroups": len(subs),
                 "subgroup_classes": len(classes), "multisets_tested": len(cands), **cnt, "filter_seconds": round(tt, 1)})
    print("  group %2d order %2d inv-classes %d subs %2d classes %2d: %d multisets -> unsat %d pass %d budget %d (%.1fs)" % (
        gi, w["order"], len(invs), len(subs), len(classes), len(cands), cnt["unsat"], cnt["pass"], cnt["budget"], tt), flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
print("total %.1fs" % (time.time() - t0))
