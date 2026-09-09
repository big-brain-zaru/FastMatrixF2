"""Approach B, step 2: for each census group class, run a from-scratch search for a G-invariant
scheme of rank <= R with N3 symmetry breaking (SB1+SB2), largest groups first.

Per group we enumerate slot multisets (orbit-type compositions) whose sizes sum to exactly R, in
order of fewest slots first, and solve each with a bounded conflict budget. Every SAT decode is
host-verified; every outcome (sat/unsat/budget) is logged per (group, multiset) to JSONL.

Usage:
  python census_run.py --census ../results/census_n4_r46_full.json --n 4 --rank 46 \
      --min-order 6 --max-multisets 40 --conf 3000000 --stride 8 --offset 0
"""
import argparse, itertools, json, time, sys
from pysat.solvers import Cadical153
import symgroup as SG
import sym_break as SB
from flip_graph import verify, target_tensor


def bounded_slots(G, n, rank, max_slots, limit):
    """Yield up to `limit` slot multisets (lists of stabilizer subgroups) with sizes summing to
    `rank`, enumerated by increasing number of slots, then by stabilizer choice."""
    import itertools as it
    ident = tuple(range(SG.L_of(n)))
    subs = {frozenset([ident])}
    for x in G:
        subs.add(frozenset(SG.closure([x], n)))
        for y in G:
            subs.add(frozenset(SG.closure([x, y], n)))
    by_size = {}
    for H in subs:
        by_size.setdefault(len(G) // len(H), []).append(H)
    sizes = sorted(by_size, reverse=True)
    out = []
    for parts in range(1, max_slots + 1):
        # compositions of `rank` into exactly `parts` parts from `sizes`, non-increasing
        def rec(t, maxs, k):
            if k == 0:
                if t == 0: yield []
                return
            for s_ in sizes:
                if s_ <= maxs and s_ * 1 <= t:
                    for rest in rec(t - s_, s_, k - 1):
                        yield [s_] + rest
        for comp in rec(rank, max(sizes), parts):
            blocks = [(sz, comp.count(sz)) for sz in sorted(set(comp), reverse=True)]
            per = [list(it.combinations_with_replacement(by_size[sz], cnt)) for sz, cnt in blocks]
            for choice in it.product(*per):
                out.append([H for blk in choice for H in blk])
                if len(out) >= limit: return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rank", type=int, default=46)
    ap.add_argument("--min-order", type=int, default=2)
    ap.add_argument("--max-slots", type=int, default=24)
    ap.add_argument("--max-multisets", type=int, default=40, help="per group, fewest-slots first")
    ap.add_argument("--conf", type=int, default=3000000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--sb", action="store_true", help="enable lex symmetry breaking (SB1+SB2). Default OFF: the 3x3 control showed SB is sound but 10-64x slower on SATISFIABLE instances; use only for pure refutation sweeps")
    a = ap.parse_args()
    n = a.n
    T = set(target_tensor(n, n, n))
    census = json.load(open(a.census))
    groups = [w for w in census["worklist"] if w["order"] >= a.min_order and "elements" in w]
    groups.sort(key=lambda w: -w["order"])
    print("census groups with elements: %d (orders %s)" % (len(groups), sorted({w["order"] for w in groups}, reverse=True)), flush=True)

    work = []
    for gi, w in enumerate(groups):
        G = [tuple(g) for g in w["elements"]]
        # re-verify the group actually preserves the tensor (guard against any census bug)
        chk = SG.check_generators(n, G, ["g%d" % i for i in range(len(G))])
        if not all(chk.values()):
            print("  group %d order %d FAILED generator check -- skipped" % (gi, w["order"]), flush=True)
            continue
        # bounded enumeration: fewest slots first, stop at max-multisets (the full enumeration for
        # large groups explodes combinatorially and hung the first sanity run)
        cands = bounded_slots(G, n, a.rank, a.max_slots, a.max_multisets)
        for mi, slots in enumerate(cands):
            work.append((gi, w["order"], G, mi, slots))
    print("total (group, multiset) instances: %d" % len(work), flush=True)
    mine = work[a.offset::a.stride]
    print("this worker: %d" % len(mine), flush=True)
    log = open(a.log or ("../results/census_run_w%d.jsonl" % a.offset), "a")

    for (gi, order, G, mi, slots) in mine:
        sig = [len(G) // len(H) for H in slots]
        t0 = time.time()
        cl, reps, terms, nv = SB.encode(n, G, slots, T, a.sb, a.sb)
        print("  group %d (order %d) multiset %d sizes %s: encoded %d vars / %d clauses in %.1fs" % (
            gi, order, mi, sig, nv, len(cl), time.time() - t0), flush=True)
        s = Cadical153(bootstrap_with=cl)
        s.conf_budget(a.conf)
        ok = s.solve_limited()
        dt = time.time() - t0
        rec = {"group_index": gi, "group_order": order, "multiset_index": mi, "orbit_sizes": sig,
               "vars": nv, "t": round(dt, 1), "sb": a.sb, "conf": a.conf}
        if ok is None:
            rec["outcome"] = "budget"
        elif ok is False:
            rec["outcome"] = "unsat"
        else:
            sch = SB.decode(set(l for l in s.get_model() if l > 0), terms, n)
            okv = verify(sch, n, n, n)
            rec["outcome"] = "sat"; rec["rank"] = len(sch); rec["verified"] = okv
            if okv:
                fn = "../results/census_rank%d_g%d_m%d.json" % (len(sch), gi, mi)
                json.dump({"format": [n, n, n], "rank": len(sch), "field": "F2",
                           "verified_brent_f2": True,
                           "found_by": "census_run group %d (order %d) multiset %s" % (gi, order, sig),
                           "group_elements": [list(g) for g in G],
                           "scheme_bitmasks": [list(t) for t in sch]}, open(fn, "w"), indent=1)
                rec["file"] = fn
                print("  *** NEW rank %d VERIFIED under group order %d -> %s" % (len(sch), order, fn), flush=True)
        s.delete()
        print("group %d (order %d) multiset %d sizes %s: %s (%.1fs, %d vars)" % (
            gi, order, mi, sig, rec["outcome"], dt, nv), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
