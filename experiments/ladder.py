"""Fixed-point ladder: search for a g-invariant scheme (g = block-preserving involution with fixed format
<2,2,2>) whose g-fixed terms restrict to a PRESCRIBED 2x2 core (list of 2x2 terms, None = vanishing
restriction). Slots: f fixed terms (stabilizer <g>) + p pairs (trivial stabilizer), rank = f + 2p.
Encoder: encode_opt (E1+E2) over G = <g>; core pinned by unit clauses on the representative bits at the
fixed indices. Every SAT decode host-verified.

  python ladder.py --rank 47 --core 47      (calibration: the 47's own involution and core -> must be SAT)
  python ladder.py --rank 46 --core 47      (a 46 sharing the 47's fixed structure?)
  python ladder.py --rank 46 --core strassen --which 0..35
"""
import argparse, json, time, itertools
from pysat.solvers import Cadical195
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import verify, target_tensor

n = 4; N = 16
S47, Gam = FP.load_gamma(4)
g = [x for x in FP.involution_classes(Gam) if FP.involution_type(x, n)[0] == "preserve"][0]
G = SG.closure([g], n)
F = FP.fixed_coords(g, n); A = sorted({P[0] for P in F}); B = sorted({P[1] for P in F})
FX = sorted({p // 4 for p in A}); FY = sorted({p % 4 for p in A}); FZ = sorted({p % 4 for p in B})
T = set(target_tensor(n, n, n))

def restrict(t):
    a, b, c = t
    return (sum((((a >> (i*4+j)) & 1) << (ii*2+jj)) for ii, i in enumerate(FX) for jj, j in enumerate(FY)),
            sum((((b >> (j*4+k)) & 1) << (jj*2+kk)) for jj, j in enumerate(FY) for kk, k in enumerate(FZ)),
            sum((((c >> (k*4+i)) & 1) << (kk*2+ii)) for kk, k in enumerate(FZ) for ii, i in enumerate(FX)))

def pin_units(rep_vars, core_term):
    """unit literals pinning the representative's bits at fixed indices to core_term (or all zero if None)"""
    units = []
    ra, rb, rc = core_term if core_term else (0, 0, 0)
    for ii, i in enumerate(FX):
        for jj, j in enumerate(FY):
            v = rep_vars[i*4+j]; units.append(v if (ra >> (ii*2+jj)) & 1 else -v)
    for jj, j in enumerate(FY):
        for kk, k in enumerate(FZ):
            v = rep_vars[N + j*4+k]; units.append(v if (rb >> (jj*2+kk)) & 1 else -v)
    for kk, k in enumerate(FZ):
        for ii, i in enumerate(FX):
            v = rep_vars[2*N + k*4+i]; units.append(v if (rc >> (kk*2+ii)) & 1 else -v)
    return units

def run(rank, core, conf, tag):
    f = len(core); assert (rank - f) % 2 == 0; p = (rank - f) // 2
    Hg = frozenset(G); Htriv = frozenset([tuple(range(48))])
    slots = [Hg] * f + [Htriv] * p
    t0 = time.time(); cl, reps, terms, nv, neq = EO.encode(n, G, slots, T, True, True)
    units = []
    for i in range(f): units += pin_units(reps[i], core[i])
    cl += [[u] for u in units]
    s = Cadical195(bootstrap_with=cl); s.conf_budget(conf); ok = s.solve_limited(); dt = time.time() - t0
    rec = {"tag": tag, "rank": rank, "f": f, "pairs": p, "vars": nv, "eqs": neq, "conf": conf, "t": round(dt, 1), "outcome": "budget" if ok is None else ("sat" if ok else "unsat")}
    if ok:
        sch = EO.decode(set(l for l in s.get_model() if l > 0), terms, n); rec["verified"] = verify(sch, 4, 4, 4); rec["rank_found"] = len(set(sch)); rec["is_the_47"] = set(sch) == set(S47)
        if rec["verified"] and len(set(sch)) < 47:
            fn = "../results/ladder_rank%d_%s.json" % (len(set(sch)), tag); json.dump({"format": [4,4,4], "rank": len(set(sch)), "field": "F2", "verified_brent_f2": True, "found_by": "ladder %s" % tag, "scheme_bitmasks": [list(t) for t in sorted(set(sch))]}, open(fn, "w"), indent=1); rec["file"] = fn
    s.delete(); open("../results/ladder.jsonl", "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True); return rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=47); ap.add_argument("--core", default="47"); ap.add_argument("--which", type=int, default=0); ap.add_argument("--conf", type=int, default=2_000_000); ap.add_argument("--zeros", type=int, default=None, help="number of vanishing fixed terms to add (default: rank-parity fill)")
    a = ap.parse_args()
    if a.core == "47":
        core = [restrict(t) for t in S47 if SG.apply(g, t, n) == t]; core = [c if all(c) else None for c in core]   # 8 nonzero + 1 None
        if a.rank == 46: core = [c for c in core if c]      # drop the vanishing one: 8 fixed + 19 pairs
    else:
        S = json.load(open("../results/all_222_rank7_F2.json"))["schemes"][a.which]; core = [tuple(t) for t in S]
        core += [None] * ((a.rank - len(core)) % 2)   # parity fill
    if a.zeros is not None: core = [c for c in core if c] + [None] * a.zeros
    print("involution fixed sets X%s Y%s Z%s; core f=%d (%d vanishing), pairs %d" % (FX, FY, FZ, len(core), sum(1 for c in core if c is None), (a.rank - len(core)) // 2), flush=True)
    run(a.rank, core, a.conf, "%s_r%d_w%d" % (a.core, a.rank, a.which))
