"""
Incremental / single-instance group-SAT for Gamma-invariant 4x4 F2 schemes.

Algorithmic fixes over h1_group_sat.py (which rebuilt a 1.9M-clause instance per orbit type):
  1. ONE encoding with K orbit slots x 12 image slots. Each orbit slot has 48 representative bits
     and a stabilizer CHOICE encoded by selector literals sel[k][H] (exactly one H per slot, or the
     slot is empty). H-invariance equalities and image activity are implied by the selectors.
     An image (coset representative g) is active iff the chosen H's coset set contains g as the
     canonical rep. Inactive images contribute nothing to the tensor sum.
  2. The rank is a CARDINALITY constraint (sum of active images <= R), so the solver chooses the
     orbit-type multiset itself instead of us enumerating 3,584 x 6 instances.
  3. Symmetry breaking: slots are used in order (slot k empty => slot k+1 empty).
  4. Learned clauses persist: one CaDiCaL object, solved incrementally with assumptions for R.

Usage: python group_sat_inc.py --max-rank 46 --slots 12 --conf 200000000
"""
import argparse, itertools, json, time, sys
from pysat.solvers import Cadical153
from pysat.card import CardEnc, EncType
import h1_group_sat as H
from flip_graph import verify, target_tensor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rank", type=int, default=46)
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--conf", type=int, default=200000000)
    ap.add_argument("--assert-47", action="store_true", help="completeness test: assert the 47's configuration")
    ap.add_argument("--out", default="../results/group_sat_inc")
    a = ap.parse_args()
    G = H.build_group(); subs = H.subgroups(G); ident = tuple(range(48))
    Gl = list(G); gidx = {g: i for i, g in enumerate(Gl)}
    # canonical coset reps per subgroup: for each H, the set of g that are min-index in their left coset gH
    coset_rep = {}
    for Hs in subs:
        reps = set()
        for g in Gl:
            coset = {H.compose(g, h) for h in Hs}
            reps.add(min(coset, key=lambda x: gidx[x]))
        coset_rep[Hs] = reps
    T = target_tensor(4, 4, 4)
    nv = 0; clauses = []
    def new():
        nonlocal nv; nv += 1; return nv
    def AND(x, y):
        z = new(); clauses.extend([[-z, x], [-z, y], [z, -x, -y]]); return z
    K = a.slots
    V = [[new() for _ in range(48)] for _ in range(K)]                 # representative bits
    SEL = [{Hs: new() for Hs in subs} for _ in range(K)]              # stabilizer choice
    EMPTY = [new() for _ in range(K)]
    ACT = [[new() for _ in Gl] for _ in range(K)]                     # image g of slot k active
    for k in range(K):
        opts = list(SEL[k].values()) + [EMPTY[k]]
        clauses.append(opts)                                          # at least one
        for x, y in itertools.combinations(opts, 2): clauses.append([-x, -y])   # at most one
        if k > 0: clauses.append([-EMPTY[k-1], EMPTY[k]])              # slots used in order
        for Hs, sl in SEL[k].items():
            for h in Hs:                                              # sel -> H-invariance
                for i in range(48):
                    j = h[i]
                    if j != i: clauses.extend([[-sl, -V[k][i], V[k][j]], [-sl, V[k][i], -V[k][j]]])
            for g in Gl:                                              # sel -> activity pattern
                clauses.append([-sl, ACT[k][gidx[g]]] if g in coset_rep[Hs] else [-sl, -ACT[k][gidx[g]]])
        for g in Gl: clauses.append([-EMPTY[k], -ACT[k][gidx[g]]])
        # nonzero factors when not empty
        clauses.append([EMPTY[k]] + V[k][0:16]); clauses.append([EMPTY[k]] + V[k][16:32]); clauses.append([EMPTY[k]] + V[k][32:48])
    # image bit literals: image of slot k under g has bit j = V[k][g^-1(j)]
    inv_idx = {g: H.inverse_index(g) for g in Gl}
    # tensor equation with gated terms: contribution = ACT & a & b & c
    for ia in range(16):
        for ib in range(16):
            for ic in range(16):
                lits = []
                for k in range(K):
                    for g in Gl:
                        ii = inv_idx[g]
                        ab = AND(V[k][ii[ia]], V[k][ii[16+ib]])
                        abc = AND(ab, V[k][ii[32+ic]])
                        lits.append(AND(abc, ACT[k][gidx[g]]))
                rhs = 1 if (ia, ib, ic) in T else 0
                cur = lits[0]
                for l in lits[1:]:
                    z = new(); clauses.extend([[-z, cur, l], [-z, -cur, -l], [z, -cur, l], [z, cur, -l]]); cur = z
                clauses.append([cur] if rhs else [-cur])
    # cardinality: total active images <= max_rank
    act_all = [ACT[k][i] for k in range(K) for i in range(len(Gl))]
    card = CardEnc.atmost(lits=act_all, bound=a.max_rank, top_id=nv, encoding=EncType.seqcounter)
    clauses.extend(card.clauses); nv = max(nv, card.nv)
    print(f"encoding: slots={K} vars={nv} clauses={len(clauses)} max_rank={a.max_rank}", flush=True)
    units = []
    if a.assert_47:
        S47 = H.load("../results/alphatensor_444_rank47.json"); seen = set(); k = 0
        for t in S47:
            if t in seen: continue
            orb = {H.apply(g, t) for g in G}; seen |= orb
            Hs = frozenset(g for g in G if H.apply(g, t) == t)
            units.append([SEL[k][Hs]])
            bits = [(t[0] >> i) & 1 for i in range(16)] + [(t[1] >> i) & 1 for i in range(16)] + [(t[2] >> i) & 1 for i in range(16)]
            for i, b in enumerate(bits): units.append([V[k][i]] if b else [-V[k][i]])
            k += 1
        for kk in range(k, K): units.append([EMPTY[kk]])
    s = Cadical153(bootstrap_with=clauses + units); s.conf_budget(a.conf)
    t0 = time.time(); ok = s.solve_limited(); dt = time.time() - t0
    if ok:
        model = set(l for l in s.get_model() if l > 0)
        scheme = []
        for k in range(K):
            for g in Gl:
                if ACT[k][gidx[g]] in model:
                    ii = inv_idx[g]
                    bitsv = [1 if V[k][ii[j]] in model else 0 for j in range(48)]
                    scheme.append((sum(bitsv[i] << i for i in range(16)), sum(bitsv[16+i] << i for i in range(16)), sum(bitsv[32+i] << i for i in range(16))))
        okv = verify(scheme, 4, 4, 4)
        print(f"SAT in {dt:.1f}s: rank {len(scheme)} VERIFIED={okv}", flush=True)
        if okv:
            fn = f"{a.out}_rank{len(scheme)}.json"
            json.dump({"format":[4,4,4],"rank":len(scheme),"field":"F2","verified_brent_f2":True,
                       "found_by":"group_sat_inc single-instance cardinality search","scheme_bitmasks":[list(t) for t in scheme]}, open(fn,"w"), indent=1)
            print("SAVED", fn, flush=True)
    else:
        print(("UNSAT" if ok is False else "budget exhausted") + f" ({dt:.1f}s)", flush=True)
    s.delete()

if __name__ == "__main__":
    main()
