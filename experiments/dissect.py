"""Full dissection of the day-3 orbit-level campaign: results AND process metrics.
Reads every orbit_lns_*.jsonl, emits a machine-readable summary + a human report."""
import json, glob, collections, os, statistics, sys, time

RES = "../results"
STAGES = [
    ("A1_le4",        ["orbit_lns_reduce_w*.jsonl", "stage1_pre_dedup/orbit_lns_reduce_w*.jsonl"], None,
     "A: delete <=2 orbits, refill <=4 orbits, one term fewer"),
    ("A1_complete",   ["orbit_lns_reduceB_w*.jsonl", "orbit_lns_reduceC_w*.jsonl"], 7729,
     "A: delete <=2 orbits, refill 5..17 orbits (completes the <=2 statement)"),
    ("A2_sweep",      ["orbit_lns_reduce3_w*.jsonl", "orbit_lns_reduce3t_w*.jsonl"], 5765,
     "A: delete 3 orbits, refill <=4 orbits, one term fewer"),
    ("A2_budgetrerun",["orbit_lns_reduce3b_w*.jsonl"], None,
     "A stage-2 budget-exhausted instances re-run at 100M conflicts"),
    ("N1_le2",        ["orbit_lns_same_w*.jsonl", "stage1_pre_dedup/orbit_lns_same_w*.jsonl"], None,
     "N1: delete <=2 orbits, refill same rank, block the original as a set"),
    ("N1_3",          ["orbit_lns_same3_w*.jsonl", "orbit_lns_same3b_w*.jsonl", "orbit_lns_same3c_w*.jsonl"], 5737,
     "N1: delete 3 orbits, refill same rank"),
    ("N1_hardrerun",  ["orbit_lns_same3d_w*.jsonl"], 17,
     "N1 budget-exhausted instances re-run at 100M conflicts"),
    ("C_555_reduce",  ["C_555_reduce_w*.jsonl"], 648,
     "C: 5x5 rank-93 record, delete <=2 C3-orbits, refill one term fewer (rank-92 attempt)"),
    ("C_666_reduce",  ["C_666_reduce_w*.jsonl"], 1578,
     "C: 6x6 rank-153 record, delete <=2 C3-orbits, refill one term fewer (rank-152 attempt)"),
    ("C_555_same",    ["C_555_same_w*.jsonl"], 1116,
     "C: 5x5 rank-93 record, same-rank diversification"),
]

def load(pats, dedupe=False):
    """Load records. dedupe=True keys by (deleted orbits, refill sizes, stabilizer orders) and is
    used ONLY for the C stages, where overlapping launches duplicated records and where the group
    (C3) has exactly one subgroup class per order so the key is faithful. The A/N1 stages were each
    launched once per instance (stride + resume), and there several subgroup classes share an
    order, so keying on orders would wrongly merge distinct instances -- raw counts are used."""
    if not dedupe:
        out = []
        for p in pats:
            for f in glob.glob(os.path.join(RES, p)):
                for l in open(f):
                    try: out.append(json.loads(l))
                    except Exception: pass
        return out
    d = {}
    for p in pats:
        for f in glob.glob(os.path.join(RES, p)):
            for l in open(f):
                try: r = json.loads(l)
                except Exception: continue
                k = (tuple(r.get("deleted_orbits", [])), tuple(r.get("refill_sizes", [])),
                     tuple(r.get("refill_subgroup_orders", [])), r.get("mode"), r.get("n"))
                d[k] = r
    return list(d.values())

def pct(v, q):
    v = sorted(v)
    return v[min(len(v)-1, int(len(v)*q))] if v else None

report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}, "totals": {}}
tot_inst = tot_sec = 0; tot_out = collections.Counter()
lines = []
for key, pats, total, desc in STAGES:
    r = load(pats, dedupe=key.startswith('C_'))
    if not r:
        report["stages"][key] = {"instances": 0, "description": desc}; continue
    out = collections.Counter(x["results"][-1]["outcome"] for x in r if x["results"])
    ts = [x["results"][-1]["t"] for x in r if x["results"]]
    new = sum(1 for x in r for y in x["results"] if y.get("outcome") == "sat" and not y.get("identical"))
    ident = sum(1 for x in r for y in x["results"] if y.get("identical"))
    Ds = collections.Counter(tuple(x["deleted_orbits"]) for x in r)
    # what makes an instance slow: correlate time with deleted total and refill orbit count
    slow = collections.defaultdict(list)
    for x in r:
        if x["results"]: slow[(len(x["refill_sizes"]), sum(x["deleted_sizes"]))].append(x["results"][-1]["t"])
    slow_tbl = sorted(((k, len(v), round(statistics.median(v),1), round(max(v),1)) for k,v in slow.items()),
                      key=lambda z: -z[2])[:6]
    undecided = [x for x in r if x["results"] and x["results"][-1]["outcome"] == "budget"]
    st = {"description": desc, "instances": len(r), "total_planned": total,
          "outcomes": dict(out), "new_schemes": new, "relabelings_blocked": ident,
          "solver_hours": round(sum(ts)/3600, 2),
          "time_s": {"median": round(statistics.median(ts),1), "p90": round(pct(ts,0.9),1), "max": round(max(ts),1)},
          "distinct_deletion_sets": len(Ds),
          "slowest_classes_refillcount_deletedterms": slow_tbl,
          "undecided": [{"D": x["deleted_orbits"], "deleted_sizes": x["deleted_sizes"],
                         "refill": x["refill_sizes"], "sub_orders": x["refill_subgroup_orders"],
                         "t": x["results"][-1]["t"]} for x in undecided]}
    report["stages"][key] = st
    tot_inst += len(r); tot_sec += sum(ts); tot_out.update(out)
    cov = f"{len(r)}/{total}" if total else str(len(r))
    lines.append(f"{key:16s} {cov:>10s}  {dict(out)}  new={new}  median={st['time_s']['median']}s max={st['time_s']['max']}s  {st['solver_hours']}h")
report["totals"] = {"instances": tot_inst, "outcomes": dict(tot_out), "solver_hours": round(tot_sec/3600,2),
                    "new_schemes": sum(s.get("new_schemes",0) for s in report["stages"].values()),
                    "relabelings_blocked": sum(s.get("relabelings_blocked",0) for s in report["stages"].values())}
json.dump(report, open(os.path.join(RES, "day3_dissection.json"), "w"), indent=1)
print("stage            decided     outcomes  new  timing  solver-time")
for l in lines: print(" ", l)
print("\nTOTAL:", report["totals"])
