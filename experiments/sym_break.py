"""N3: lexicographic symmetry breaking for from-scratch G-invariant scheme search (n-generic).

Two provably satisfiability-preserving constraints:

  (SB1) orbit-representative canonicity: for each orbit slot with representative bit-vector v and
        each g in G, assert  v <=_lex  g.v .  Every orbit contains a lex-minimal element, so any
        G-invariant scheme has a representative choice satisfying this; it removes the |orbit|-fold
        choice of which element names the orbit.

  (SB2) inter-orbit ordering: for consecutive orbit slots carrying the SAME stabilizer subgroup,
        assert  v_i <=_lex v_{i+1} .  Orbits are an unordered set, so this only removes the k!
        permutations of identical-type slots.

Both are standard; each is applied only where its precondition holds. `--check` runs the
satisfiability-preservation test: solve a known-satisfiable instance with and without the
constraints and confirm both are SAT and the constrained solution still verifies.

Usage:
  python sym_break.py --check --n 3 --group cyc --rank 23
"""
import argparse, itertools, json, time
from pysat.solvers import Cadical153
import symgroup as SG
from flip_graph import verify, target_tensor


def lex_leq(clauses, new, X, Y):
    """Assert X <=_lex Y for equal-length bit-vector variable lists (MSB-first as given).
    Standard chain: e_i means 'prefix up to i is equal'."""
    L = len(X)
    e_prev = None
    for i in range(L):
        # if prefix equal so far then X_i <= Y_i  i.e.  not(X_i and not Y_i)
        if e_prev is None:
            clauses.append([-X[i], Y[i]])
        else:
            clauses.append([-e_prev, -X[i], Y[i]])
        if i == L - 1:
            break
        e_i = new()
        # e_i  <->  e_prev and (X_i == Y_i)
        if e_prev is None:
            clauses.append([-e_i, -X[i], Y[i]])
            clauses.append([-e_i, X[i], -Y[i]])
            clauses.append([e_i, X[i], Y[i]])
            clauses.append([e_i, -X[i], -Y[i]])
        else:
            clauses.append([-e_i, e_prev])
            clauses.append([-e_i, -X[i], Y[i]])
            clauses.append([-e_i, X[i], -Y[i]])
            clauses.append([e_i, -e_prev, X[i], Y[i]])
            clauses.append([e_i, -e_prev, -X[i], -Y[i]])
        e_prev = e_i


def encode(n, G, slot_subs, rank_terms_R, use_sb1, use_sb2):
    """Encode a from-scratch G-invariant search over the given orbit slots.
    rank_terms_R: residual coordinate set (target tensor if nothing is kept)."""
    N = n * n
    L = 3 * N
    st = {"nv": 0}
    clauses = []

    def new():
        st["nv"] += 1
        return st["nv"]

    def AND(x, y):
        z = new()
        clauses.append([-z, x]); clauses.append([-z, y]); clauses.append([z, -x, -y])
        return z

    reps = []; terms = []
    for H in slot_subs:
        v = [new() for _ in range(L)]
        reps.append(v)
        for h in H:
            for i in range(L):
                j = h[i]
                if j != i:
                    clauses.append([-v[i], v[j]]); clauses.append([v[i], -v[j]])
        clauses.append(v[0:N]); clauses.append(v[N:2 * N]); clauses.append(v[2 * N:3 * N])
        seen = set(); cosets = []
        for g in G:
            key = frozenset(SG.compose(g, h) for h in H)
            if key in seen: continue
            seen.add(key); cosets.append(g)
        for g in cosets:
            inv = SG.inverse_index(g)
            terms.append([v[inv[j]] for j in range(L)])

    if use_sb1:
        for v, H in zip(reps, slot_subs):
            for g in G:
                if g in H:  # g fixes this orbit's rep by construction
                    continue
                inv = SG.inverse_index(g)
                gv = [v[inv[j]] for j in range(L)]
                lex_leq(clauses, new, v, gv)
    if use_sb2:
        for i in range(len(slot_subs) - 1):
            if slot_subs[i] == slot_subs[i + 1]:
                lex_leq(clauses, new, reps[i], reps[i + 1])

    for ia in range(N):
        for ib in range(N):
            ab = [AND(tv[ia], tv[N + ib]) for tv in terms]
            for ic in range(N):
                lits = [AND(ab[t], terms[t][2 * N + ic]) for t in range(len(terms))]
                rhs = 1 if (ia, ib, ic) in rank_terms_R else 0
                cur = lits[0]
                for l in lits[1:]:
                    z = new()
                    clauses.append([-z, cur, l]); clauses.append([-z, -cur, -l])
                    clauses.append([z, -cur, l]); clauses.append([z, cur, -l])
                    cur = z
                clauses.append([cur] if rhs else [-cur])
    return clauses, reps, terms, st["nv"]


def decode(model, terms, n):
    N = n * n
    out = []
    for tv in terms:
        b = [1 if x in model else 0 for x in tv]
        out.append((sum(b[i] << i for i in range(N)),
                    sum(b[N + i] << i for i in range(N)),
                    sum(b[2 * N + i] << i for i in range(N))))
    return out


def slots_for(G, n, rank, max_slots):
    """Enumerate orbit-slot stabilizer multisets whose indices sum to `rank`."""
    ident = tuple(range(SG.L_of(n)))
    subs = {frozenset([ident])}
    for a in G:
        subs.add(frozenset(SG.closure([a], n)))
        for b in G:
            subs.add(frozenset(SG.closure([a, b], n)))
    by_size = {}
    for H in subs:
        by_size.setdefault(len(G) // len(H), []).append(sorted(H))
    sizes = sorted(by_size, reverse=True)

    def comps(t, maxs, parts):
        if t == 0:
            yield []; return
        if parts == 0:
            return
        for s in sizes:
            if s <= maxs and s <= t:
                for rest in comps(t - s, s, parts - 1):
                    yield [s] + rest
    out = []
    for comp in comps(rank, max(sizes), max_slots):
        blocks = [(sz, comp.count(sz)) for sz in sorted(set(comp), reverse=True)]
        per = [list(itertools.combinations_with_replacement([tuple(map(tuple, h)) for h in by_size[sz]], cnt))
               for sz, cnt in blocks]
        for choice in itertools.product(*per):
            out.append([frozenset(tuple(x) for x in H) for blk in choice for H in blk])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--group", default="cyc", choices=["cyc", "tau", "s3"])
    ap.add_argument("--rank", type=int, default=23)
    ap.add_argument("--max-slots", type=int, default=12)
    ap.add_argument("--conf", type=int, default=20000000)
    ap.add_argument("--check", action="store_true", help="satisfiability-preservation test")
    ap.add_argument("--limit", type=int, default=6, help="how many slot-multisets to try")
    a = ap.parse_args()
    n = a.n
    gens = {"cyc": [SG.perm_cyc(n)], "tau": [SG.perm_tau(n)], "s3": [SG.perm_cyc(n), SG.perm_tau(n)]}[a.group]
    assert all(SG.check_generators(n, gens, [a.group] * len(gens)).values())
    G = SG.closure(gens, n)
    T = set(target_tensor(n, n, n))
    cand = slots_for(G, n, a.rank, a.max_slots)
    print("n=%d group=%s |G|=%d rank=%d slot-multisets=%d" % (n, a.group, len(G), a.rank, len(cand)), flush=True)

    modes = [("plain", False, False), ("SB1", True, False), ("SB1+SB2", True, True)] if a.check else [("SB1+SB2", True, True)]
    for slots in cand[:a.limit]:
        sig = [len(G) // len(H) for H in slots]
        row = []
        for label, s1, s2 in modes:
            cl, reps, terms, nv = encode(n, G, slots, T, s1, s2)
            s = Cadical153(bootstrap_with=cl); s.conf_budget(a.conf)
            t0 = time.time(); ok = s.solve_limited(); dt = time.time() - t0
            res = "budget" if ok is None else ("SAT" if ok else "UNSAT")
            verified = None
            if ok:
                sch = decode(set(l for l in s.get_model() if l > 0), terms, n)
                verified = verify(sch, n, n, n)
                if verified:
                    fn = "../results/symbreak_n%d_%s_rank%d.json" % (n, a.group, len(sch))
                    json.dump({"format": [n, n, n], "rank": len(sch), "field": "F2",
                               "verified_brent_f2": True, "found_by": "sym_break %s" % label,
                               "scheme_bitmasks": [list(t) for t in sch]}, open(fn, "w"), indent=1)
            s.delete()
            row.append("%s=%s%s %.1fs/%dv" % (label, res, "" if verified is None else ("/verified" if verified else "/BADDECODE"), dt, nv))
        print("  slots %s: %s" % (sig, " | ".join(row)), flush=True)


if __name__ == "__main__":
    main()
