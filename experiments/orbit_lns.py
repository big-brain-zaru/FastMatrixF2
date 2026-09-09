"""
Orbit-level LNS around a Gamma-invariant scheme (Approach A) and same-rank diversification (N1).

Given the 47 and its symmetry group Gamma (order 12), delete a set D of whole orbits (total size s)
and ask SAT for a refill by orbit types with sizes summing to s-1 (mode reduce) or s (mode same),
enumerating every stabilizer subgroup choice. Kept orbits are asserted as unit clauses. In mode
same, the original scheme is blocked AS A SET: after each SAT we compare the decoded term set with
the input; identical => add a blocking clause on the found representative bits and continue, so
relabelings cannot masquerade as new schemes. Every instance is appended to a JSONL log.

Usage:
  python orbit_lns.py --mode reduce --max-delete 2 --max-new 4 --stride 8 --offset 0
  python orbit_lns.py --mode same   --max-delete 3 --max-new 4 --distinct 20 --stride 8 --offset 0
"""
import argparse, itertools, json, time, sys
from pysat.solvers import Cadical153
import h1_group_sat as H
from flip_graph import verify, target_tensor

def bits_of(t):
    a, b, c = t
    return [(a >> i) & 1 for i in range(16)] + [(b >> i) & 1 for i in range(16)] + [(c >> i) & 1 for i in range(16)]

def build_solver(specs, G, units):
    """Same encoding as h1_group_sat.encode but returns (solver, terms) for incremental blocking."""
    T = target_tensor(4, 4, 4)
    nv = 0; clauses = list(units)
    def new():
        nonlocal nv; nv += 1; return nv
    def AND(x, y):
        z = new(); clauses.extend([[-z, x], [-z, y], [z, -x, -y]]); return z
    reps = []; terms = []
    for Hs in specs:
        v = [new() for _ in range(48)]; reps.append(v)
        for h in Hs:
            for i in range(48):
                j = h[i]
                if j != i: clauses.extend([[-v[i], v[j]], [v[i], -v[j]]])
        clauses.append(v[0:16]); clauses.append(v[16:32]); clauses.append(v[32:48])
        seen = set(); cosets = []
        for g in G:
            key = frozenset(H.compose(g, h) for h in Hs)
            if key in seen: continue
            seen.add(key); cosets.append(g)
        for g in cosets:
            inv = H.inverse_index(g); terms.append([v[inv[j]] for j in range(48)])
    for ia in range(16):
        for ib in range(16):
            for ic in range(16):
                lits = []
                for tv in terms:
                    ab = AND(tv[ia], tv[16+ib]); lits.append(AND(ab, tv[32+ic]))
                rhs = 1 if (ia, ib, ic) in T else 0
                cur = lits[0]
                for l in lits[1:]:
                    z = new(); clauses.extend([[-z, cur, l], [-z, -cur, -l], [z, -cur, l], [z, cur, -l]]); cur = z
                clauses.append([cur] if rhs else [-cur])
    return Cadical153(bootstrap_with=clauses), reps, terms

def decode(model, terms):
    out = []
    for tv in terms:
        b = [1 if x in model else 0 for x in tv]
        out.append((sum(b[i] << i for i in range(16)), sum(b[16+i] << i for i in range(16)), sum(b[32+i] << i for i in range(16))))
    return out

def compositions(total, sizes, max_parts):
    sizes = sorted(sizes, reverse=True)
    def rec(t, maxs, parts):
        if t == 0: yield []; return
        if parts == 0: return
        for s in sizes:
            if s <= maxs and s <= t:
                for rest in rec(t - s, s, parts - 1): yield [s] + rest
    yield from rec(total, max(sizes), max_parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="../results/alphatensor_444_rank47.json")
    ap.add_argument("--mode", choices=["reduce", "same"], default="reduce")
    ap.add_argument("--max-delete", type=int, default=2, help="max number of orbits deleted at once")
    ap.add_argument("--min-delete", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=4, help="max number of refill orbits")
    ap.add_argument("--min-new", type=int, default=1)
    ap.add_argument("--distinct", type=int, default=10, help="mode same: distinct schemes to enumerate per instance")
    ap.add_argument("--conf", type=int, default=20000000)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--resume-glob", default=None, help='chained: "glob:stride;glob:stride"')
    ap.add_argument("--rerun-budget", default=None, help='"glob:stride" of the stage whose budget-exhausted instances to re-run')
    a = ap.parse_args()
    G = H.build_group(); subs = H.subgroups(G)
    # deduplicate stabilizer subgroups up to Gamma-conjugacy: an orbit with stabilizer Hs at rep t has
    # stabilizer g Hs g^-1 at rep g.t, so conjugate choices give isomorphic SAT instances.
    def conj(Hs, g):
        ginv = H.inverse_index(g)
        return frozenset(H.compose(H.compose(g, h), tuple(ginv)) for h in Hs)
    classes = []
    for Hs in subs:
        if any(conj(Hs, g) in cls for cls in classes for g in G): continue
        classes.append({conj(Hs, g) for g in G})
    reps_sub = [min(cls, key=lambda x: sorted(x)) for cls in classes]
    by_size = {}
    for Hs in reps_sub: by_size.setdefault(len(G)//len(Hs), []).append(Hs)
    print("stabilizer conjugacy classes per orbit size:", {k: len(v) for k, v in sorted(by_size.items())}, flush=True)
    S = H.load(a.scheme); Sset = set(S)
    seen = set(); specs = []; reps = []; orbs = []
    for t in S:
        if t in seen: continue
        orb = {H.apply(g, t) for g in G}; seen |= orb
        specs.append(frozenset(g for g in G if H.apply(g, t) == t)); reps.append(t); orbs.append(sorted(orb))
    sizes_orig = [len(o) for o in orbs]
    log = open(a.log or f"../results/orbit_lns_{a.mode}_w{a.offset}.jsonl", "a")
    # enumerate instances
    instances = []
    for nd in range(a.min_delete, a.max_delete + 1):
        for D in itertools.combinations(range(len(orbs)), nd):
            s = sum(sizes_orig[k] for k in D)
            target = s if a.mode == "same" else s - 1
            if target <= 0: continue
            for comp in compositions(target, list(by_size), a.max_new):
                if len(comp) < a.min_new: continue
                # group equal sizes: choose a multiset of subgroup classes per size
                sizes_sorted = sorted(comp, reverse=True)
                blocks = [(sz, sizes_sorted.count(sz)) for sz in sorted(set(sizes_sorted), reverse=True)]
                per_block = [list(itertools.combinations_with_replacement(by_size[sz], cnt)) for sz, cnt in blocks]
                for choice in itertools.product(*per_block):
                    combo = tuple(Hs for blk in choice for Hs in blk)
                    instances.append((D, sizes_sorted, combo))
    instances.sort(key=lambda x: (len(x[1]), -sum(sizes_orig[k] for k in x[0])))
    todo = instances
    if a.resume_glob:
        # chained resume: "glob:stride;glob:stride" -- each stage's worker w decided todo[w::stride][:n_w]
        # over the todo list *as it stood at that stage*.
        import glob as _g, re as _re
        for stage in a.resume_glob.split(";"):
            pat, stride = stage.rsplit(":", 1); stride = int(stride)
            done_idx = set()
            for f in _g.glob(pat):
                m = _re.search(r"_w(\d+)\.jsonl$", f)
                if not m: continue
                w = int(m.group(1)); n_w = sum(1 for _ in open(f))
                done_idx.update(w + stride * i for i in range(n_w))
            todo = [x for i, x in enumerate(todo) if i not in done_idx]
            print(f"resume stage {pat}: skipped {len(done_idx)}; remaining {len(todo)}", flush=True)
    if a.rerun_budget:
        # re-run exactly the budget-exhausted instances of a previous stage: "glob:stride" whose todo
        # list is the one AFTER the resume chain given in --resume-glob (excluding that stage itself)
        import glob as _g, re as _re
        pat, stride = a.rerun_budget.rsplit(":", 1); stride = int(stride); picked = []
        for f in _g.glob(pat):
            m = _re.search(r"_w(\d+)\.jsonl$", f)
            if not m: continue
            w = int(m.group(1))
            for i, line in enumerate(open(f)):
                r = json.loads(line)
                if r["results"] and r["results"][-1]["outcome"] == "budget": picked.append(todo[w + stride * i])
        todo = picked
        print(f"rerun-budget: {len(todo)} budget-exhausted instances selected", flush=True)
    mine = todo[a.offset::a.stride]
    print(f"orbit sizes {sizes_orig}; total instances {len(instances)}; this worker {len(mine)}", flush=True)
    for (D, comp, combo) in mine:
        kept = [k for k in range(len(orbs)) if k not in D]
        spec_list = [specs[k] for k in kept] + list(combo)
        units = []
        for pos, k in enumerate(kept):
            for i, bit in enumerate(bits_of(reps[k])): units.append([48*pos+i+1] if bit else [-(48*pos+i+1)])
        t0 = time.time()
        solver, rep_vars, terms = build_solver(spec_list, G, units)
        rec = {"mode": a.mode, "deleted_orbits": list(D), "deleted_sizes": [sizes_orig[k] for k in D],
               "refill_sizes": comp, "refill_subgroup_orders": [len(Hs) for Hs in combo], "results": []}
        found_distinct = 0; identical = 0
        for it in range(a.distinct if a.mode == "same" else 1):
            solver.conf_budget(a.conf); ok = solver.solve_limited(); dt = time.time() - t0
            if ok is None: rec["results"].append({"outcome": "budget", "t": round(dt, 1)}); break
            if ok is False: rec["results"].append({"outcome": "unsat", "t": round(dt, 1)}); break
            model = set(l for l in solver.get_model() if l > 0)
            scheme = decode(model, terms); okv = verify(scheme, 4, 4, 4)
            same = set(scheme) == Sset
            shared = [sum(1 for u, v in itertools.combinations(scheme, 2) if u[ax] == v[ax]) for ax in range(3)]
            rec["results"].append({"outcome": "sat", "t": round(dt, 1), "rank": len(scheme), "verified": okv, "identical": same, "shared": shared})
            if okv and not same:
                found_distinct += 1
                fn = f"../results/orbit_lns_{a.mode}_rank{len(scheme)}_D{'-'.join(map(str,D))}_{found_distinct}.json"
                json.dump({"format": [4,4,4], "rank": len(scheme), "field": "F2", "verified_brent_f2": True,
                           "found_by": f"orbit_lns {a.mode}: deleted orbits {list(D)} sizes {rec['deleted_sizes']}, refill {comp}",
                           "scheme_bitmasks": [list(t) for t in scheme]}, open(fn, "w"), indent=1)
                rec["results"][-1]["file"] = fn
                print(f"  *** NEW rank {len(scheme)} scheme, shared factors {shared}: {fn}", flush=True)
                if a.mode == "reduce": break
            else:
                identical += int(same)
            # block this exact assignment of the new orbits' representative bits
            new_reps = rep_vars[len(kept):]
            solver.add_clause([-v if v in model else v for rv in new_reps for v in rv])
        solver.delete()
        summ = rec["results"][-1]["outcome"] if rec["results"] else "none"
        print(f"D={list(D)} sizes={rec['deleted_sizes']} refill={comp} subs={rec['refill_subgroup_orders']}: {summ} distinct={found_distinct} identical={identical} ({time.time()-t0:.1f}s)", flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

if __name__ == "__main__":
    main()
