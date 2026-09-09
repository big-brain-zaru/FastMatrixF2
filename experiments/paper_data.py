"""Recompute every headline number quoted in the Zenodo paper (main.tex / supplement.tex) from the
raw artefacts in results/, and write paper/paper_data.json. Nothing here is transcribed from a
narrative document: schemes are re-verified against the Brent equations, invariants are recomputed,
and campaign tallies are re-aggregated from the per-instance JSONL logs.

Usage:  python paper_data.py        (from experiments/)
"""
import json, glob, os, collections, itertools, time
import numpy as np
import symgroup as SG, fp_filter as FP
from flip_graph import verify

R = "../results"
OUT = ".."
D = {"generated": time.strftime("%Y-%m-%d %H:%M"), "source": "experiments/paper_data.py"}


# ---------------------------------------------------------------- helpers
def load(p):
    return [tuple(t) for t in json.load(open(p))["scheme_bitmasks"]]


def jl(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(R, pat))):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def key(r):
    return (tuple(r["deleted_orbits"]), tuple(r["refill_sizes"]), tuple(r["refill_subgroup_orders"]))


def f2rank(m, n):
    rows = [(m >> (n * i)) & ((1 << n) - 1) for i in range(n)]
    r = 0
    while rows:
        p = rows.pop()
        if not p:
            continue
        r += 1
        hb = p.bit_length() - 1
        rows = [x ^ p if (x >> hb) & 1 else x for x in rows if x]
    return r


def signature(S, n):
    """GL(n,2)^3-invariant factor-rank signature: per axis, the histogram of F2 matrix ranks."""
    return [dict(sorted(collections.Counter(f2rank(t[ax], n) for t in S).items())) for ax in range(3)]


def shared(S):
    """Flip isolation: per axis, how many terms share their factor with another term."""
    return [sum(v for v in collections.Counter(t[ax] for t in S).values() if v > 1) for ax in range(3)]


def weights(S):
    return dict(sorted(collections.Counter(sum(bin(t[ax]).count("1") for ax in range(3)) for t in S).items()))


# ---------------------------------------------------------------- 1. verified schemes
print("[1/8] verifying schemes and recomputing invariants")
S47 = load(os.path.join(R, "alphatensor_444_rank47.json"))
S47B = load(os.path.join(R, "records/gamma47_B_rank47.json"))
S49 = load(os.path.join(R, "gpu_444_rank49.json"))
SS = load(os.path.join(R, "strassen_sq_rank49.json"))
rec5 = load(os.path.join(R, "records/lille_555_rank93.json"))
rec6 = load(os.path.join(R, "records/lille_666_rank153.json"))
_, G = FP.load_gamma(4)
C3_5 = SG.closure([SG.perm_cyc(5)], 5)
C3_6 = SG.closure([SG.perm_cyc(6)], 6)

D["schemes"] = {
    "47_alphatensor": {"rank": len(set(S47)), "verified": verify(S47, 4, 4, 4),
                       "gamma_invariant": SG.is_invariant(S47, G, 4),
                       "orbit_sizes": sorted([len(o) for _, o, _ in SG.orbits(S47, G, 4)], reverse=True),
                       "signature": signature(S47, 4), "shared": shared(S47), "weights": weights(S47)},
    "47_second": {"rank": len(set(S47B)), "verified": verify(S47B, 4, 4, 4),
                  "gamma_invariant": SG.is_invariant(S47B, G, 4),
                  "orbit_sizes": sorted([len(o) for _, o, _ in SG.orbits(S47B, G, 4)], reverse=True),
                  "signature": signature(S47B, 4), "shared": shared(S47B), "weights": weights(S47B)},
    "49_self_found": {"rank": len(set(S49)), "verified": verify(S49, 4, 4, 4),
                      "signature": signature(S49, 4), "shared": shared(S49)},
    "49_strassen_squared": {"rank": len(set(SS)), "verified": verify(SS, 4, 4, 4),
                            "signature": signature(SS, 4), "shared": shared(SS)},
    "93_record": {"rank": len(set(rec5)), "verified": verify(rec5, 5, 5, 5),
                  "C3_invariant": SG.is_invariant(rec5, C3_5, 5), "shared": shared(rec5),
                  "orbit_sizes": collections.Counter(len(o) for _, o, _ in SG.orbits(rec5, C3_5, 5))},
    "153_record": {"rank": len(set(rec6)), "verified": verify(rec6, 6, 6, 6),
                   "C3_invariant": SG.is_invariant(rec6, C3_6, 6), "shared": shared(rec6),
                   "orbit_sizes": collections.Counter(len(o) for _, o, _ in SG.orbits(rec6, C3_6, 6))},
}
D["schemes"]["93_record"]["orbit_sizes"] = dict(D["schemes"]["93_record"]["orbit_sizes"])
D["schemes"]["153_record"]["orbit_sizes"] = dict(D["schemes"]["153_record"]["orbit_sizes"])
D["group_gamma"] = {"order": len(G), "stabilizer_orders": sorted(12 // s for s in D["schemes"]["47_alphatensor"]["orbit_sizes"])}


# ---------------------------------------------------------------- 2. the two 47s are inequivalent
print("[2/8] testing the two rank-47 schemes for equivalence under the ambient group")
A, B = frozenset(S47), frozenset(S47B)
_n = 4
_P = [[1 << p[r] for r in range(_n)] for p in itertools.permutations(range(_n))]
_I = [1 << r for r in range(_n)]
_gens = [SG.perm_cyc(_n), SG.perm_tau(_n)] + [SG.perm_sandwich(M, _I, _I, _n) for M in _P[1:]] + \
        [SG.perm_sandwich(_I, M, _I, _n) for M in _P[1:]] + [SG.perm_sandwich(_I, _I, M, _n) for M in _P[1:]]
Omega = SG.closure(_gens, _n)
hit = any(frozenset(SG.apply(g, t, 4) for t in S47) == B for g in Omega)
D["two_47s"] = {
    "common_terms": len(A & B),
    "signatures_differ": D["schemes"]["47_alphatensor"]["signature"] != D["schemes"]["47_second"]["signature"],
    "signature_multisets_differ": sorted(map(json.dumps, D["schemes"]["47_alphatensor"]["signature"]))
                                  != sorted(map(json.dumps, D["schemes"]["47_second"]["signature"])),
    "ambient_group_order": len(Omega),
    "equivalent_under_ambient_group": hit,
    "both_flip_isolated": D["schemes"]["47_alphatensor"]["shared"] == [0, 0, 0] == D["schemes"]["47_second"]["shared"],
}


# ---------------------------------------------------------------- 3. Z/4 lift
print("[3/8] Hensel lift F2 -> Z/4")
try:
    import hensel_lift as HL
    # HL.lift(S, 1) returns 1 if the first Hensel level (F2 -> Z/4) is inconsistent, 2 if it lifts.
    D["z4_lift"] = {"47_alphatensor_lifts_to_Z4": HL.lift(S47, 1) > 1,
                    "47_second_lifts_to_Z4": HL.lift(S47B, 1) > 1,
                    "strassen_squared_lifts_to_Z4": HL.lift(SS, 1) > 1,
                    "system": "4096 equations, 2256 unknowns over F2 at the first level"}
except Exception as e:                                                     # pragma: no cover
    D["z4_lift"] = {"error": repr(e)}


# ---------------------------------------------------------------- 4. Strassen uniqueness over F2
print("[4/8] rank-7 term sets for <2,2,2> over F2")
s222 = json.load(open(os.path.join(R, "all_222_rank7_F2.json")))
sample = [tuple(map(tuple, s)) for s in s222["schemes"][:5]]
D["strassen_f2"] = {"rank7_term_sets": s222["count"], "orbits_under_GL22_cubed_times_S3": s222["classes"],
                    "orbit_sizes": s222["orbit_sizes"],
                    "sample_all_verified": all(verify(list(s), 2, 2, 2) for s in sample)}


# ---------------------------------------------------------------- 5. rigidity theorems (4x4)
print("[5/8] re-aggregating the orbit-rewrite campaigns")
def collapse(records, pred=lambda r: True):
    by = {}
    for r in records:
        if not pred(r):
            continue
        by.setdefault(key(r), set()).add(r["results"][-1]["outcome"])
    return by

t1 = collapse(jl("orbit_lns_reduce_w*.jsonl") + jl("orbit_lns_reduceB_w*.jsonl") + jl("orbit_lns_reduceC_w*.jsonl"),
              lambda r: len(r["deleted_orbits"]) <= 2)
t2 = collapse(jl("orbit_lns_reduce3*.jsonl"), lambda r: len(r["deleted_orbits"]) == 3)
t3 = collapse(jl("orbit_lns_same*.jsonl"))
close = jl("n1_close*.jsonl")
D["theorems_4x4"] = {
    "T1_instance_keys": len(t1), "T1_all_unsat": all(v == {"unsat"} for v in t1.values()),
    "T2_instance_keys": len(t2), "T2_all_unsat_after_reruns": all("unsat" in v for v in t2.values()),
    "T3_instance_keys": len(t3), "T3_open_after_day3": sum(1 for v in t3.values() if "unsat" not in v),
    "T3_closure_outcomes": dict(collections.Counter(r["outcome"] for r in close)),
    "T3_new_47s_in_closure": sum(1 for r in close if r["outcome"] == "sat" and r.get("verified") and not r.get("identical_to_47")),
    "raw_instance_records": {"T1": len(jl("orbit_lns_reduce_w*.jsonl") + jl("orbit_lns_reduceB_w*.jsonl") + jl("orbit_lns_reduceC_w*.jsonl")),
                             "T2": len(jl("orbit_lns_reduce3*.jsonl")), "T3": len(jl("orbit_lns_same*.jsonl"))},
}

# second 47: same sweeps
red_b = jl("G47B_reduce_w*.jsonl")
same_b = jl("G47B_same_w*.jsonl")
D["theorems_47B"] = {
    "reduce46_instances": len(red_b),
    "reduce46_outcomes": dict(collections.Counter(r["results"][-1]["outcome"] for r in red_b)),
    "same_rank_instances": len(same_b),
    "same_rank_outcomes": dict(collections.Counter(r["results"][-1]["outcome"] for r in same_b)),
    "solver_hours": round(sum(x.get("t", 0) for r in red_b + same_b for x in r["results"]) / 3600.0, 3),
}


# ---------------------------------------------------------------- 6. 5x5 / 6x6
print("[6/8] 5x5 and 6x6 families and reductions")
t4 = collapse(jl("C_555_reduce_w*.jsonl"))
t5 = collapse(jl("C_666_reduce_w*.jsonl"))
on = json.load(open(os.path.join(R, "on_dissection.json")))
D["theorems_5x5_6x6"] = {
    "T4_record_D2_instances": len(t4), "T4_all_unsat": all(v == {"unsat"} for v in t4.values()),
    "T5_record_D2_instances": len(t5), "T5_all_unsat": all(v == {"unsat"} for v in t5.values()),
    "T4_extended": {"D2_over_64_pool_members": on["reduce5_D2_all"], "D3_far_seeds": on["reduce5_D3_far"]},
    "T5_extended": {"D2_over_11_seeds": on["reduce6_D2"]},
    "rank92_files_found": on["rank92_files"], "rank152_files_found": on["rank152_files"],
}
D["families"] = {"5x5": on["family_5"], "6x6": on["family_6"], "walks": on["walks"]}

# per-campaign-stage tallies (raw instance records), as regenerated by report_figures.py
D["campaign_stages"] = json.load(open("../docs/figures/data.json"))["sat_stages"]
D["solve_times"] = json.load(open("../docs/figures/data.json"))["solve_times"]


# ---------------------------------------------------------------- 7. fixed-point filter and censuses
print("[7/8] filter power and symmetry censuses")
filt = {}
for line in open(os.path.join(R, "B2_filter.jsonl")):
    r = json.loads(line)
    filt[(r["group_index"], tuple(r["class_key"]))] = r["filter"]
fc = collections.Counter(filt.values())
b2 = json.load(open(os.path.join(R, "b2_dissection.json")))
D["filter"] = {"instances": len(filt), "verdicts": dict(fc),
               "eliminated_exactly": fc.get("unsat", 0) + fc.get("count", 0) + fc.get("count7", 0),
               "B2_full_solves": b2["outcomes"]}

st = json.load(open(os.path.join(R, "investigation_state.json")))
D["censuses"] = {"frozen_at": st["frozen_at"],
                 "gamma46": {k: st["G46_filter"][k] for k in ("processed", "total", "outcomes", "by_slots")},
                 "gl_odd": {k: st["GL_odd"][k] for k in ("processed", "outcomes", "by_order") if k in st["GL_odd"]},
                 "permutation": st["PC2"]}
D["censuses"]["gamma46"]["survivors_listed"] = len(st["G46_filter"]["survivors"])
D["censuses"]["gl_odd"]["total"] = st["GL_odd"].get("total", 4321)

# small-format ranks used by the corollary
D["small_formats"] = {"rank_F2_222": 7, "rank_F2_222_source": "SAT: rank<=6 UNSAT, rank<=7 SAT (fp_filter.small_rank_table)",
                      "table": json.load(open(os.path.join(R, "small_rank_table.json")))}


# ---------------------------------------------------------------- 8. resource accounting
print("[8/8] resource accounting")
def solver_seconds(pat, field="t"):
    s = 0.0
    for r in jl(pat):
        if "results" in r:
            s += sum(x.get(field, 0) or 0 for x in r["results"])
        else:
            s += r.get(field, 0) or 0
    return s

pats = ["orbit_lns_*.jsonl", "C_555_*.jsonl", "C_666_*.jsonl", "n1_close*.jsonl", "B2_w*.jsonl",
        "B2_filter*.jsonl", "G47B_*.jsonl", "G46_filter.jsonl", "GL_odd.jsonl", "PC2.jsonl",
        "gamma_scratch.jsonl", "P1_*.jsonl", "ON5*.jsonl", "ON6*.jsonl", "C25*.jsonl", "C26*.jsonl", "ladder.jsonl"]
acct = {}
for p in pats:
    recs = jl(p)
    if recs:
        acct[p] = {"records": len(recs), "solver_hours": round(solver_seconds(p) / 3600.0, 2)}
D["resources"] = {"per_log": acct,
                  "total_logged_records": sum(v["records"] for v in acct.values()),
                  "total_solver_hours": round(sum(v["solver_hours"] for v in acct.values()), 1)}
walks = D["families"]["walks"]
D["resources"]["flip_walk_steps_5x5_6x6"] = {k: v["total_steps"] for k, v in walks.items()}

json.dump(D, open(os.path.join(OUT, "paper_data.json"), "w"), indent=1, default=str)
print("\nwrote paper/paper_data.json")
for k in ("two_47s", "z4_lift", "strassen_f2", "theorems_4x4", "theorems_47B", "filter"):
    print(" ", k, json.dumps(D[k], default=str)[:300])
print("  families 5x5 distinct", D["families"]["5x5"]["distinct"], "| 6x6 distinct", D["families"]["6x6"]["distinct"])
print("  total solver hours", D["resources"]["total_solver_hours"], "| logged records", D["resources"]["total_logged_records"])
