"""Generate the figures and data tables for docs/SCIENTIFIC-REPORT.md from the raw result files.
Every number is read from results/, nothing is typed in. Output: docs/figures/*.png + docs/figures/data.json"""
import json, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results"; OUT = "../docs/figures"; os.makedirs(OUT, exist_ok=True)
data = {}

def jl(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(R, pattern))):
        for l in open(f):
            l = l.strip()
            if l:
                try: out.append(json.loads(l))
                except Exception: pass
    return out

# ---------------------------------------------------------------- 1. walker trajectories (best rank vs time)
runs = {
    "gen-1 flips, trivial start (4x4)": "gpu_444_run.json",
    "gen-2 population + policies (4x4)": "g2_444_run2.json",
    "tau-quotient walker (4x4)": "z2_444_runz2_v3.json",
    "rank<=3 gated walker (4x4)": "g2_444_rk3_run2.json",
    "released from symmetric pool (4x4)": "g2_444_released_run2.json",
}
plt.figure(figsize=(8, 4.5))
traj = {}
for label, f in runs.items():
    p = os.path.join(R, f)
    if not os.path.exists(p): continue
    d = json.load(open(p))
    prog = d.get("progress", [])
    if not prog: continue
    t = [x["t"] for x in prog]; b = [x["best"] for x in prog]
    # extend to run end
    t.append(d.get("seconds", t[-1])); b.append(b[-1])
    plt.step(t, b, where="post", label="%s -> floor %d" % (label, min(b)))
    traj[label] = {"floor": int(min(b)), "seconds": d.get("seconds"), "steps": d.get("total_steps", d.get("total_flip_steps")),
                   "steps_per_sec": d.get("steps_per_sec")}
plt.axhline(47, color="k", ls="--", lw=1); plt.text(5, 47.3, "record 47 (AlphaTensor, F2)", fontsize=8)
plt.axhline(49, color="gray", ls=":", lw=1); plt.text(5, 49.3, "published plain-flip floor 49", fontsize=8, color="gray")
plt.xscale("log"); plt.xlabel("wall time (s, log)"); plt.ylabel("best verified rank"); plt.title("4x4 walker dynamics: best rank vs time")
plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_walker_trajectories.png"), dpi=150); plt.close()
data["walker_runs"] = traj

# ---------------------------------------------------------------- 2. factor-rank signatures
def f2rank(m, n):
    rows = [(m >> (n * i)) & ((1 << n) - 1) for i in range(n)]
    r = 0
    while rows:
        piv = rows.pop()
        if not piv: continue
        r += 1; hb = piv.bit_length() - 1
        rows = [x ^ piv if (x >> hb) & 1 else x for x in rows if x]
    return r
def signature(scheme, n):
    return [collections.Counter(f2rank(t[ax], n) for t in scheme) for ax in range(3)]
S47 = [tuple(t) for t in json.load(open(os.path.join(R, "alphatensor_444_rank47.json")))["scheme_bitmasks"]]
S49 = [tuple(t) for t in json.load(open(os.path.join(R, "gpu_444_rank49.json")))["scheme_bitmasks"]]
SS = [tuple(t) for t in json.load(open(os.path.join(R, "strassen_sq_rank49.json")))["scheme_bitmasks"]]
sig = {"47": signature(S47, 4), "self-found 49": signature(S49, 4), "Strassen x Strassen": signature(SS, 4)}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, axis_name, ai in zip(axes, ["a-factors", "b-factors", "c-factors"], range(3)):
    w = 0.25
    for k, (name, s) in enumerate(sig.items()):
        ax.bar(np.arange(1, 5) + (k - 1) * w, [s[ai].get(r, 0) for r in range(1, 5)], w, label=name)
    ax.set_title(axis_name); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("F2 rank of factor"); ax.grid(alpha=.3, axis="y")
axes[0].set_ylabel("number of terms"); axes[0].legend(fontsize=7)
fig.suptitle("GL-invariant factor-rank signatures: the 47 vs the rank-49 family"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_signatures.png"), dpi=150); plt.close()
data["signatures"] = {k: [dict(c) for c in v] for k, v in sig.items()}

# ---------------------------------------------------------------- 3. SAT campaigns: instance counts and outcomes
d3 = json.load(open(os.path.join(R, "day3_dissection.json")))["stages"]
b2 = json.load(open(os.path.join(R, "b2_dissection.json")))
stages = [("A1 |D|<=2, refill<=4", "A1_le4"), ("A1 completeness, refill 5-17", "A1_complete"), ("A2 |D|=3, refill<=4", "A2_sweep"),
          ("N1 |D|<=2 same-rank", "N1_le2"), ("N1 |D|=3 same-rank", "N1_3"), ("C 5x5 rank-92", "C_555_reduce"),
          ("C 6x6 rank-152", "C_666_reduce"), ("C 5x5 same-rank", "C_555_same")]
labels, uns, sat, bud, hrs = [], [], [], [], []
for lab, key in stages:
    if key not in d3: continue
    o = d3[key]["outcomes"]; labels.append(lab); uns.append(o.get("unsat", 0)); sat.append(o.get("sat", 0)); bud.append(o.get("budget", 0)); hrs.append(d3[key]["solver_hours"])
labels.append("B2 census (3 h)"); uns.append(b2["outcomes"].get("unsat", 0)); sat.append(0); bud.append(b2["outcomes"].get("budget", 0)); hrs.append(b2["solver_hours"])
fig, ax = plt.subplots(figsize=(9, 4.2))
y = np.arange(len(labels))
ax.barh(y, uns, color="#4a7", label="UNSAT (exact)"); ax.barh(y, sat, left=uns, color="#36c", label="SAT (verified)"); ax.barh(y, bud, left=np.array(uns) + np.array(sat), color="#c94", label="budget exhausted")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("instances (log)")
for i, h in enumerate(hrs): ax.text(max(uns[i] + sat[i] + bud[i], 1) * 1.15, i, "%.1f solver-h" % h, va="center", fontsize=7)
ax.legend(fontsize=8, loc="lower right"); ax.set_title("Exact SAT campaigns: outcomes per stage"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_sat_campaigns.png"), dpi=150); plt.close()
data["sat_stages"] = [{"stage": l, "unsat": u, "sat": s, "budget": b, "solver_hours": h} for l, u, s, b, h in zip(labels, uns, sat, bud, hrs)]

# ---------------------------------------------------------------- 4. solve-time distributions: UNSAT vs budget, A/N1 (orbit LNS) and B2 (from scratch)
def times(recs, key="t"):
    out = collections.defaultdict(list)
    for r in recs:
        for res in (r.get("results") or [r]):
            o = res.get("outcome"); t = res.get(key)
            if o and t is not None: out[o].append(float(t))
    return out
orbit = times(jl("orbit_lns_reduce*.jsonl") + jl("orbit_lns_same*.jsonl"))
census = times(jl("B2_w*.jsonl"))
c5 = times(jl("C_555_*.jsonl") + jl("C_666_*.jsonl"))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (title, tt) in zip(axes, [("4x4 orbit-level LNS around the 47 (A, N1)", orbit), ("5x5 / 6x6 orbit-level LNS (C)", c5), ("4x4 from-scratch census (B2)", census)]):
    bins = np.logspace(-1, 4, 40)
    for o, col in [("unsat", "#4a7"), ("sat", "#36c"), ("budget", "#c94")]:
        if tt.get(o): ax.hist(tt[o], bins=bins, alpha=.7, color=col, label="%s (n=%d, median %.0f s)" % (o, len(tt[o]), np.median(tt[o])))
    ax.set_xscale("log"); ax.set_xlabel("solver time per instance (s)"); ax.set_title(title, fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3)
axes[0].set_ylabel("instances")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_solve_times.png"), dpi=150); plt.close()
data["solve_times"] = {name: {o: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(max(v))} for o, v in tt.items()}
                       for name, tt in [("orbit_lns_4x4", orbit), ("orbit_lns_5x5_6x6", c5), ("census_B2", census)]}

# ---------------------------------------------------------------- 5. census filter: elimination vs fixed coordinates
wl = json.load(open(os.path.join(R, "B2_worklist.json")))
S = wl["summary"]
fc = [s["fixed_coords"][0] if s["fixed_coords"] else -1 for s in S]
elim = [(s["count"] + s["unsat"] + s.get("count7", 0)) / max(s["multisets"], 1) * 100 for s in S]
orders = [s["order"] for s in S]
fig, ax = plt.subplots(figsize=(8, 4))
cats = collections.OrderedDict([("no involution (odd order)", lambda f: f == -1), ("involution, 0 fixed coords", lambda f: f == 0), ("64 fixed coords (<2,2,2> / matrix)", lambda f: f == 64), ("256 fixed coords (<2,2,4>)", lambda f: f == 256), ("1024 fixed coords (<2,4,4>)", lambda f: f == 1024)])
xs = [];
for i, (name, pred) in enumerate(cats.items()):
    vals = [e for e, f in zip(elim, fc) if pred(f)]
    ax.scatter([i + np.random.uniform(-.15, .15) for _ in vals], vals, s=[20 + 3 * o for o, f in zip(orders, fc) if pred(f)], alpha=.7)
    ax.text(i, -6, "%d groups" % len(vals), ha="center", fontsize=8)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(list(cats.keys()), fontsize=7, rotation=12); ax.set_ylabel("% of the group's multisets eliminated exactly"); ax.set_ylim(-10, 70)
ax.set_title("Fixed-point filter power by involution type (marker size = group order)"); ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_filter_power.png"), dpi=150); plt.close()
data["filter_groups"] = [{"group": s["group_index"], "order": s["order"], "fixed_coords": f, "multisets": s["multisets"], "eliminated": s["count"] + s["unsat"] + s.get("count7", 0), "budget": s["budget"]} for s, f in zip(S, fc)]

# ---------------------------------------------------------------- 6. the 47's orbit structure under Gamma
import sys; sys.path.insert(0, ".")
import symgroup as SG, fp_filter as FP
S47l, G = FP.load_gamma(4)
orbs = SG.orbits(S47l, G, 4)
sizes = sorted([len(o) for (_, o, _) in orbs], reverse=True)
stabs = sorted([len(H) for (_, _, H) in orbs])
data["gamma_orbits"] = {"group_order": len(G), "orbit_sizes": sizes, "stabilizer_orders": sorted([12 // s for s in sizes])}
rec5 = [tuple(t) for t in json.load(open(os.path.join(R, "records/lille_555_rank93.json")))["scheme_bitmasks"]]
rec6 = [tuple(t) for t in json.load(open(os.path.join(R, "records/lille_666_rank153.json")))["scheme_bitmasks"]]
def shared(scheme):
    out = []
    for ax in range(3):
        c = collections.Counter(t[ax] for t in scheme); out.append(sum(v for v in c.values() if v > 1))
    return out
data["flip_isolation"] = {"4x4 rank 47": shared(S47l), "4x4 rank 49 (self-found)": shared(S49), "5x5 rank 93 (record)": shared(rec5), "6x6 rank 153 (record)": shared(rec6)}
new93 = sorted(glob.glob(os.path.join(R, "orbitn_5_same_rank93_*.json")))
sets93 = {frozenset(tuple(t) for t in json.load(open(f))["scheme_bitmasks"]) for f in new93}
data["new_93s"] = {"files": len(new93), "distinct_schemes": len(sets93), "shared_factors": sorted(shared(list(S)) for S in sets93)}

# pools
pools = {}
for f in ["z2_444_pool.json", "gated50_pool.json", "newbasin_49_pool.json", "g2_444_released_pool.json", "z2_444_rk3_pool.json", "g2_444_rk3_pool.json"]:
    if not os.path.exists(os.path.join(R, f)):
        continue     # bulk walker pools are not published; every figure works without them
    d = json.load(open(os.path.join(R, f))); sch = d["schemes"]
    ranks = collections.Counter(len(s) if isinstance(s, list) else len(s.get("scheme_bitmasks", s.get("scheme", []))) for s in sch)
    pools[f] = {"n": len(sch), "ranks": dict(ranks)}
data["pools"] = pools

json.dump(data, open(os.path.join(OUT, "data.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in data.items() if k not in ("filter_groups",)}, indent=1)[:6000])
print("figures written to", OUT)
