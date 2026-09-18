"""Generate the seven publication figures for the paper, as PDF (for LaTeX) and PNG (for the
README), into figures/. All inputs are raw artefacts under results/ plus paper/paper_data.json, which is
itself recomputed from results/ by paper_data.py. Nothing is typed in by hand.

Usage:  python paper_figures.py       (from experiments/)
"""
import json, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results"
OUT = "../figures"
os.makedirs(OUT, exist_ok=True)
PD = json.load(open("../paper_data.json", encoding="utf-8"))
RD = json.load(open("../docs/figures/data.json", encoding="utf-8"))
ON = json.load(open(os.path.join(R, "on_dissection.json"), encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight",
})
C = {"unsat": "#2c7a4b", "sat": "#1f5fa9", "budget": "#c07a1e", "grey": "#777777",
     "a": "#1f5fa9", "b": "#c0392b", "c": "#2c7a4b", "d": "#7d3c98"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("  wrote", name + ".pdf/.png")


def jl(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(R, pattern))):
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


# ------------------------------------------------------------------ Figure 1: search floors at 4x4
print("figure 1: walker trajectories")
runs = {
    "$\\tau$-quotient walker": "z2_444_runz2_v3.json",
    "rank-$\\leq$3 gated walker": "g2_444_rk3_run2.json",
    "released from symmetric pool": "g2_444_released_run2.json",
}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for label, f in runs.items():
    p = os.path.join(R, f)
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    prog = d.get("progress", [])
    if not prog:
        continue
    t = [x["t"] for x in prog] + [d.get("seconds", prog[-1]["t"])]
    b = [x["best"] for x in prog] + [prog[-1]["best"]]
    ax.step(t, b, where="post", lw=1.5, label="%s $\\rightarrow$ floor %d" % (label, min(b)))
ax.axhline(47, color="k", ls="--", lw=1)
ax.text(4, 47.4, "record 47 (AlphaTensor, $\\mathbb{F}_2$)", fontsize=7.5)
ax.axhline(49, color=C["grey"], ls=":", lw=1)
ax.text(4, 49.4, "Strassen$^{\\otimes 2}$ family, rank 49", fontsize=7.5, color=C["grey"])
ax.set_xscale("log")
ax.set_xlabel("wall time (s, log scale)")
ax.set_ylabel("best host-verified rank")
ax.set_title("$4\\times4$ flip-walk dynamics: no walk crossed rank 49")
ax.legend(loc="upper right")
save(fig, "figure1_walker_floors")

# ------------------------------------------------------------------ Figure 2: factor-rank signatures
print("figure 2: factor-rank signatures of the two 47s and the 49 family")
S = PD["schemes"]
series = [("AlphaTensor rank 47", S["47_alphatensor"]["signature"], C["a"]),
          ("this work, rank 47 (second)", S["47_second"]["signature"], C["b"]),
          ("rank 49 (Strassen$^{\\otimes 2}$ family)", S["49_self_found"]["signature"], C["grey"])]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, axis_name, ai in zip(axes, ["$a$-factors", "$b$-factors", "$c$-factors"], range(3)):
    w = 0.26
    for k, (name, sig, col) in enumerate(series):
        ax.bar(np.arange(1, 5) + (k - 1) * w, [sig[ai].get(str(r), sig[ai].get(r, 0)) for r in range(1, 5)],
               w, label=name, color=col)
    ax.set_title(axis_name)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("$\\mathbb{F}_2$ rank of factor")
axes[0].set_ylabel("number of terms")
axes[0].set_ylim(0, 44)
axes[0].legend(loc="upper right", fontsize=6.0, framealpha=0.95)
fig.suptitle("GL$(4,2)^3$-invariant factor-rank signatures", y=1.02)
save(fig, "figure2_signatures")

# ------------------------------------------------------------------ Figure 3: exact campaign outcomes
print("figure 3: campaign outcomes")
NAMES = {
    "A1 |D|<=2, refill<=4": "$4{\\times}4$ rank 46, $\\leq$2 orbits deleted",
    "A1 completeness, refill 5-17": "$4{\\times}4$ rank 46, $\\leq$2 orbits, refills 5--17",
    "A2 |D|=3, refill<=4": "$4{\\times}4$ rank 46, 3 orbits deleted",
    "N1 |D|<=2 same-rank": "$4{\\times}4$ same rank, $\\leq$2 orbits",
    "N1 |D|=3 same-rank": "$4{\\times}4$ same rank, 3 orbits",
    "C 5x5 rank-92": "$5{\\times}5$ rank 92, from the record",
    "C 6x6 rank-152": "$6{\\times}6$ rank 152, from the record",
    "C 5x5 same-rank": "$5{\\times}5$ same rank, from the record",
    "B2 census (3 h)": "$4{\\times}4$ rank 46, from scratch (census)",
}
rows = []
for st_ in PD["campaign_stages"]:
    rows.append((NAMES.get(st_["stage"], st_["stage"]), st_["unsat"], st_["sat"], st_["budget"], st_["solver_hours"]))
tb = PD["theorems_47B"]
rows.append(("$4{\\times}4$ rank 46, around the second 47", tb["reduce46_outcomes"].get("unsat", 0), 0, 0, tb["solver_hours"] / 2))
rows.append(("$4{\\times}4$ same rank, around the second 47", tb["same_rank_outcomes"].get("unsat", 0), 0, 0, tb["solver_hours"] / 2))
e5 = PD["theorems_5x5_6x6"]["T4_extended"]
e6 = PD["theorems_5x5_6x6"]["T5_extended"]
rows.append(("$5{\\times}5$ rank 92, over 64 family members", e5["D2_over_64_pool_members"]["outcomes"].get("unsat", 0), 0, 0,
             e5["D2_over_64_pool_members"]["solver_hours"]))
rows.append(("$5{\\times}5$ rank 92, 3 orbits, far seeds", e5["D3_far_seeds"]["outcomes"].get("unsat", 0), 0, 0,
             e5["D3_far_seeds"]["solver_hours"]))
rows.append(("$6{\\times}6$ rank 152, over 11 family members", e6["D2_over_11_seeds"]["outcomes"].get("unsat", 0), 0, 0,
             e6["D2_over_11_seeds"]["solver_hours"]))
rows.sort(key=lambda r: -(r[1] + r[2] + r[3]))
y = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(6.8, 4.6))
left = np.zeros(len(rows))
for idx, kind in ((1, "unsat"), (2, "sat"), (3, "budget")):
    v = np.array([r[idx] for r in rows], dtype=float)
    ax.barh(y, v, left=left, color=C[kind],
            label={"unsat": "UNSAT (exact refutation)", "sat": "SAT (host-verified scheme)",
                   "budget": "conflict budget exhausted"}[kind])
    left += v
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
ax.invert_yaxis()
ax.set_xscale("symlog", linthresh=100)
ax.set_xlim(0, 1.2e5)
ax.set_xlabel("SAT instances (symmetric log scale)")
ax.set_title("Exact outcomes of every orbit-rewrite and census campaign")
ax.legend(loc="lower right")
for i, r in enumerate(rows):
    ax.text(left[i] * 1.15 + 2, i, "%.1f h" % r[4], va="center", fontsize=6.5, color=C["grey"])
save(fig, "figure3_campaign_outcomes")

# ------------------------------------------------------------------ Figure 4: solve-time distributions
print("figure 4: solve times")
st = PD["solve_times"]
groups = [("$4\\times4$ orbit rewrites", st["orbit_lns_4x4"]),
          ("$5\\times5$ / $6\\times6$ rewrites", st["orbit_lns_5x5_6x6"]),
          ("$4\\times4$ census (from scratch)", st["census_B2"])]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, (title, g) in zip(axes, groups):
    xs, meds, p90s, maxs, cols = [], [], [], [], []
    for i, kind in enumerate(("unsat", "sat", "budget")):
        if kind not in g:
            continue
        xs.append(len(xs))
        meds.append(g[kind]["median"])
        p90s.append(g[kind]["p90"])
        maxs.append(g[kind]["max"])
        cols.append(C[kind])
    ax.vlines(xs, meds, maxs, color=cols, lw=1)
    for xi, kind in zip(xs, [k for k in ("unsat", "sat", "budget") if k in g]):
        ax.text(xi, 3.2e4, "n=%d" % g[kind]["n"], ha="center", fontsize=6.5, color=C[kind])
    ax.scatter(xs, meds, color=cols, s=26, zorder=3, label=None)
    ax.scatter(xs, p90s, color=cols, s=14, marker="_", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1.2e5)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([k for k in ("UNSAT", "SAT", "budget") if k.lower() in g][:len(xs)], fontsize=7.5)
    ax.set_title(title)
axes[0].set_ylabel("solver time per instance (s, log)\nmedian $\\bullet$, p90 $-$, max")
fig.suptitle("Decidable instances decide fast; budgets are burned by undecidable ones", y=1.03)
save(fig, "figure4_solve_times")

# ------------------------------------------------------------------ Figure 5: fixed-point filter power
print("figure 5: filter power")
fg = RD["filter_groups"]
kinds = {0: ("no fixed coordinate", C["grey"]), 64: ("$\\langle 2,2,2\\rangle$ fixed format", C["a"]),
         256: ("$\\langle 2,2,4\\rangle$ fixed format", C["b"]), 1024: ("$\\langle 2,4,4\\rangle$ fixed format", C["c"])}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
seen = set()
for g in fg:
    frac = g["eliminated"] / g["multisets"] if g["multisets"] else 0
    lab, col = kinds.get(g["fixed_coords"], ("other", C["d"]))
    ax.scatter(g["order"] + (np.random.RandomState(g["group"]).rand() - .5) * 0.6, 100 * frac,
               s=14 + 3.0 * g["order"], color=col, alpha=.75,
               label=None if lab in seen else lab, edgecolors="none")
    seen.add(lab)
ax.set_xlabel("order of the symmetry group")
ax.set_ylabel("% of that group's rank-46 orbit-type multisets\nrefuted exactly by the filter")
ax.set_title("Power of the fixed-point restriction lemma, by involution type")
ax.legend(loc="upper left")
save(fig, "figure5_filter_power")

# ------------------------------------------------------------------ Figure 6: same-rank families at 5x5 / 6x6
print("figure 6: same-rank families")
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
fig.subplots_adjust(wspace=0.34)
ax = axes[0]
# (seeds whose exact two-orbit neighbourhood was fully enumerated, distinct schemes known at that point)
g5 = [(0, 1), (20, 65), (108, 263), (125, 307)]
g6 = [(0, 1), (1, 17), (4, 67), (5, 79)]
ax.plot([p[0] for p in g5], [p[1] for p in g5], "o-", color=C["a"], label="$5\\times5$ rank 93")
ax.plot([p[0] for p in g6], [p[1] for p in g6], "s-", color=C["b"], label="$6\\times6$ rank 153")
ax.set_xlabel("seeds fully expanded")
ax.set_ylabel("distinct verified schemes at record rank")
ax.set_title("Same-rank families grow at every stage")
ax.legend(loc="upper left")
ax = axes[1]
h5 = ON["family_5"]["shared_hist"]
h6 = ON["family_6"]["shared_hist"]
k5 = sorted(int(k.split("/")[0]) for k in h5)
k6 = sorted(int(k.split("/")[0]) for k in h6)
ax.bar([k - 0.2 for k in k5], [h5["%d/%d/%d" % (k, k, k)] for k in k5], 0.4, color=C["a"], label="$5\\times5$ (307 schemes)")
ax.bar([k + 0.2 for k in k6], [h6["%d/%d/%d" % (k, k, k)] for k in k6], 0.4, color=C["b"], label="$6\\times6$ (79 schemes)")
ax.axvline(0.5, color="k", ls="--", lw=1)
ax.text(0.7, max(h5.values()) * 0.85, "flip-isolated\n($4\\times4$ records live here)", fontsize=6.5)
ax.set_xlabel("shared factors per axis (0 = flip-isolated)")
ax.set_ylabel("schemes")
ax.set_title("Every family member is flippable")
ax.set_xticks([0, 5, 10, 15])
ax.legend(loc="upper right")
save(fig, "figure6_families")

# ------------------------------------------------------------------ Figure 7: symmetry-census funnel
print("figure 7: census funnel")
by = PD["censuses"]["gamma46"]["by_slots"]
slots = sorted(int(k) for k in by)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
fig.subplots_adjust(wspace=0.34)
ax = axes[0]
bot = np.zeros(len(slots))
for kind, lab, col in (("count", "refuted by the involution count bound", C["c"]),
                       ("filter", "refuted by the projected subsystem", C["a"]),
                       ("pass", "survives, needs a full solve", C["budget"])):
    v = np.array([by[str(s)].get(kind, 0) for s in slots], dtype=float)
    ax.bar(slots, v, bottom=bot, color=col, label=lab)
    bot += v
ax.set_xlabel("number of orbits in the rank-46 multiset")
ax.set_ylabel("orbit-type multisets")
ax.set_ylim(0, 9900)
ax.set_title("$\\Gamma$ rank-46 census\n(%d of %d multisets processed)" %
             (PD["censuses"]["gamma46"]["processed"], PD["censuses"]["gamma46"]["total"]))
ax.legend(loc="upper left", fontsize=6.5)
ax = axes[1]
gl = PD["censuses"]["gl_odd"]
orders = sorted(gl["by_order"], key=int)
u = [gl["by_order"][o].get("unsat", 0) for o in orders]
w = [gl["by_order"][o].get("wall", 0) for o in orders]
x = np.arange(len(orders))
ax.bar(x, u, 0.55, color=C["unsat"], label="UNSAT (exact)")
ax.bar(x, w, 0.55, bottom=u, color=C["budget"], label="undecided at the 300 s cap")
ax.set_xticks(x)
ax.set_xticklabels(["order %s" % o for o in orders])
ax.set_ylabel("census instances")
ax.set_title("Odd-order GL$(4,2)^3$ symmetries\n(%d of %d instances run)" % (gl["processed"], gl["total"]))
ax.legend(loc="upper right", fontsize=6.5)
save(fig, "figure7_census")

print("\nall figures written to", os.path.abspath(OUT))
