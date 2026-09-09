"""Approach B, filtered rerun. Two phases:

  --prepare : for every census group (order >= --min-order) enumerate stabilizer-CLASS multisets
              (fewest slots first, <= --max-multisets per group), apply the fixed-point filter
              (fp_filter.run_filter) to each, log every verdict to <tag>_filter.jsonl and write the
              survivors to <tag>_worklist.json, ordered largest group first, fewest slots first.
  worker    : take survivors [offset::stride], run the full from-scratch encoding (sym_break.encode,
              symmetry breaking OFF) with a conflict budget, host-verify any SAT decode, log to
              <tag>_w<offset>.jsonl.

Usage:
  python census_run2.py --prepare --census ../results/census_n4_r46_cyclic.json --tag B2 --max-multisets 200
  python census_run2.py --census ... --tag B2 --conf 1000000 --stride 10 --offset 0
"""
import argparse, json, time
from pysat.solvers import Cadical153
import symgroup as SG
import sym_break as SB
import fp_filter as FP
import encode_opt as EO
from pysat.solvers import Cadical195
from flip_graph import verify, target_tensor


def prepare(a):
    n = a.n; T = set(target_tensor(n, n, n))
    census = json.load(open(a.census))
    groups = [w for w in census["worklist"] if "elements" in w]
    suffix = "" if a.pstride == 1 else "_p%d" % a.poffset
    flog = open("../results/%s_filter%s.jsonl" % (a.tag, suffix), "w")
    work = []; summary = []
    t_all = time.time()
    lb_table = FP.small_rank_table()
    print("small-format rank lower bounds in use: %s" % lb_table, flush=True)
    for gi, w in enumerate(groups):
        if w["order"] < a.min_order or (gi % a.pstride) != a.poffset:
            continue
        G = [tuple(g) for g in w["elements"]]
        chk = SG.check_generators(n, G, ["g%d" % i for i in range(len(G))])
        if not all(chk.values()):
            print("group %d order %d FAILED generator check -- skipped" % (gi, w["order"]), flush=True)
            continue
        invs = FP.involution_classes(G)
        nfix = [len(FP.fixed_coords(g, n)) for g in invs]
        subs = FP.all_subgroups(G, n); classes = FP.subgroup_classes(G, subs)
        cands = FP.slots_by_class(G, n, a.rank, a.max_slots, a.max_multisets, classes)
        cnt = {"count": 0, "unsat": 0, "pass": 0, "budget": 0}; t0 = time.time()
        bound, bdesc = (0, "no involution with fixed coordinates")
        if invs and nfix[0] > 0:
            bound, bdesc = FP.count_bound(invs[0], n, T, lb_table)
        for mi, (slots, key) in enumerate(cands):
            sizes = [len(G) // len(H) for H in slots]
            fc = FP.fixed_count(G, slots, invs[0]) if invs else None
            if invs and nfix[0] > 0 and fc < bound:
                v, dt, st = "count", 0.0, None          # zero-cost elimination: too few fixed terms
            else:
                v, dt, st = FP.run_filter(n, G, slots, T, conf=a.filter_conf)
            cnt[v] += 1
            rec = {"group_index": gi, "group_order": w["order"], "class_key": list(key), "orbit_sizes": sizes,
                   "fixed_terms": fc, "count_bound": bound, "filter": v, "filter_t": round(dt, 3)}
            flog.write(json.dumps(rec) + "\n")
            if v not in ("unsat", "count"):
                work.append({"group_index": gi, "group_order": w["order"], "class_key": list(key),
                             "orbit_sizes": sizes, "filter": v, "fixed_terms": fc, "count_bound": bound})
        flog.flush()
        row = {"group_index": gi, "order": w["order"], "involution_classes": len(invs), "fixed_coords": nfix,
               "count_bound": bound, "bound_desc": bdesc,
               "subgroup_classes": len(classes), "multisets": len(cands), **cnt, "seconds": round(time.time() - t0, 1)}
        summary.append(row)
        print("group %2d order %2d fixed-coords %s bound %d classes %d: %d multisets -> count-eliminated %d, sat-eliminated %d, survive %d (%.1fs)" % (
            gi, w["order"], nfix, bound, len(classes), len(cands), cnt["count"], cnt["unsat"], cnt["pass"] + cnt["budget"], time.time() - t0), flush=True)
    work.sort(key=lambda r: (-r["group_order"], len(r["orbit_sizes"])))
    json.dump({"n": n, "rank": a.rank, "census": a.census, "min_order": a.min_order,
               "max_multisets": a.max_multisets, "filter_conf": a.filter_conf,
               "lb_table": {"x".join(map(str, k)): v for k, v in lb_table.items()},
               "summary": summary, "work": work},
              open("../results/%s_worklist%s.json" % (a.tag, suffix), "w"), indent=1)
    tot = sum(r["multisets"] for r in summary); el = sum(r["unsat"] + r["count"] for r in summary)
    print("PREPARED%s: %d groups, %d multisets, %d eliminated (%.1f%%), %d survivors (%.1fs)" % (
        suffix, len(summary), tot, el, 100.0 * el / max(tot, 1), len(work), time.time() - t_all), flush=True)


def merge(a):
    import glob
    parts = sorted(glob.glob("../results/%s_worklist_p*.json" % a.tag))
    summary = []; work = []; meta = None
    for p in parts:
        d = json.load(open(p)); meta = d; summary += d["summary"]; work += d["work"]
    summary.sort(key=lambda r: r["group_index"])
    # post-hoc tightening with the small-format ranks proved AFTER the prepare pass started
    # (results/small_rank_table.json): a <2,2,2>-type involution needs >= 7 fixed terms.
    lb = FP.small_rank_table()
    tight = {}
    for r in summary:
        if "preserve fixed-format <" in r.get("bound_desc", ""):
            fmt = tuple(sorted(int(x) for x in r["bound_desc"].split("<")[1].split(">")[0].split(",")))
            b = r["count_bound"]
            for k, v in lb.items():
                if all(k[i] <= fmt[i] for i in range(3)):
                    b = max(b, v)
            if b > r["count_bound"]:
                tight[r["group_index"]] = b; r["count_bound_tightened"] = b
    kept = []; dropped = []
    for r in work:
        b = tight.get(r["group_index"])
        if b is not None and r["fixed_terms"] is not None and r["fixed_terms"] < b:
            r["filter_before_tightening"] = r["filter"]; r["filter"] = "count7"; r["count_bound"] = b; dropped.append(r)
        else:
            kept.append(r)
    work = kept
    for r in summary:
        d = [x for x in dropped if x["group_index"] == r["group_index"]]
        r["count7"] = len(d)
        for x in d:
            r[x["filter_before_tightening"]] -= 1
    print("post-hoc <2,2,2> bound 7: %d more survivors eliminated in groups %s" % (len(dropped), sorted(tight)))
    # breadth-first order: the k-th surviving multiset of every group before the (k+1)-th of any,
    # larger groups first within a round (so a fixed time budget covers every class evenly)
    pos = {}
    for r in work:                      # each part is already in fewest-slots-first order per group
        r["pos_in_group"] = pos.get(r["group_index"], 0); pos[r["group_index"]] = r["pos_in_group"] + 1
    work.sort(key=lambda r: (r["pos_in_group"], -r["group_order"], r["group_index"]))
    meta["summary"] = summary; meta["work"] = work
    json.dump(meta, open("../results/%s_worklist.json" % a.tag, "w"), indent=1)
    with open("../results/%s_filter.jsonl" % a.tag, "w") as out:
        for p in sorted(glob.glob("../results/%s_filter_p*.jsonl" % a.tag)):
            out.write(open(p).read())
        for r in dropped:
            out.write(json.dumps({k: v for k, v in r.items() if k != "pos_in_group"}) + "\n")
    tot = sum(r["multisets"] for r in summary)
    print("MERGED %d parts: %d groups, %d multisets, count-eliminated %d, sat-eliminated %d, survivors %d" % (
        len(parts), len(summary), tot, sum(r["count"] for r in summary), sum(r["unsat"] for r in summary), len(work)))


def worker(a):
    n = a.n; T = set(target_tensor(n, n, n))
    wl = json.load(open("../results/%s_worklist.json" % a.tag))
    census = json.load(open(a.census))
    groups = [w for w in census["worklist"] if "elements" in w]
    mine = wl["work"][a.offset::a.stride]
    print("worker %d/%d: %d instances" % (a.offset, a.stride, len(mine)), flush=True)
    log = open("../results/%s_w%d.jsonl" % (a.tag, a.offset), "a")
    class_cache = {}
    for r in mine:
        gi = r["group_index"]; G = [tuple(g) for g in groups[gi]["elements"]]
        if gi not in class_cache:   # same deterministic enumeration as prepare (sorted by size, then elements)
            class_cache[gi] = FP.subgroup_classes(G, FP.all_subgroups(G, n))
        classes = class_cache[gi]
        slots = [classes[ci][0] for ci in r["class_key"]]
        assert [len(G) // len(H) for H in slots] == r["orbit_sizes"]
        sig = r["orbit_sizes"]
        t0 = time.time()
        # 09-05: optimised encoder (orbit equations + gate sharing, encode_opt.py) and CaDiCaL 1.9.5;
        # validated against sym_break.encode (identical verdicts) -- see docs/OPTIMIZATION-DIGEST.md
        cl, reps, terms, nv, neq = EO.encode(n, G, slots, T, True, True)
        print("  group %d (order %d) key %s sizes %s: encoded %d vars / %d clauses / %d eqs in %.1fs" % (
            gi, r["group_order"], r["class_key"], sig, nv, len(cl), neq, time.time() - t0), flush=True)
        s = Cadical195(bootstrap_with=cl); s.conf_budget(a.conf)
        ok = s.solve_limited(); dt = time.time() - t0
        rec = {"group_index": gi, "group_order": r["group_order"], "class_key": r["class_key"],
               "orbit_sizes": sig, "vars": nv, "t": round(dt, 1), "conf": a.conf, "filter": r["filter"], "enc": "E1+E2", "solver": "cadical195"}
        if ok is None:
            rec["outcome"] = "budget"
        elif ok is False:
            rec["outcome"] = "unsat"
        else:
            sch = EO.decode(set(l for l in s.get_model() if l > 0), terms, n)
            okv = verify(sch, n, n, n)
            rec["outcome"] = "sat"; rec["rank"] = len(sch); rec["verified"] = okv
            if okv:
                fn = "../results/%s_rank%d_g%d_%s.json" % (a.tag, len(sch), gi, "-".join(map(str, r["class_key"])))
                json.dump({"format": [n, n, n], "rank": len(sch), "field": "F2", "verified_brent_f2": True,
                           "found_by": "census_run2 group %d (order %d) sizes %s" % (gi, r["group_order"], sig),
                           "group_elements": [list(g) for g in G],
                           "scheme_bitmasks": [list(t) for t in sch]}, open(fn, "w"), indent=1)
                rec["file"] = fn
                print("  *** NEW rank %d VERIFIED under group order %d -> %s" % (len(sch), r["group_order"], fn), flush=True)
        s.delete()
        print("group %d (order %d) sizes %s: %s (%.1fs, %d vars)" % (gi, r["group_order"], sig, rec["outcome"], dt, nv), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--tag", default="B2")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rank", type=int, default=46)
    ap.add_argument("--min-order", type=int, default=2)
    ap.add_argument("--max-slots", type=int, default=46)
    ap.add_argument("--max-multisets", type=int, default=200)
    ap.add_argument("--filter-conf", type=int, default=50000, help="conflict budget per projected instance; budget = survive")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pstride", type=int, default=1, help="prepare: process groups gi with gi %% pstride == poffset")
    ap.add_argument("--poffset", type=int, default=0)
    ap.add_argument("--conf", type=int, default=1000000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    if a.prepare:
        prepare(a)
    elif a.merge:
        merge(a)
    else:
        worker(a)


if __name__ == "__main__":
    main()
