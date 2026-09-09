"""
H1 follow-up: from-scratch SAT search for 4x4 F2 matmul schemes invariant under the order-12
group Gamma = <S3 permutation sandwiches, tau> discovered as the stabilizer of the known 47.

A Gamma-invariant scheme is a union of orbits. Each orbit is determined by a representative term
and its stabilizer subgroup H <= Gamma (orbit size 12/|H|). Unknowns = representative bits (48 per
orbit) with H-invariance equalities; the tensor equation sums the distinct images (coset reps).

Usage:
  python h1_group_sat.py --orbits 12,6,6,6,6,3,3,3,1,1 --conf 20000000     # the 47's orbit type
  python h1_group_sat.py --rank 46 --enumerate --conf 5000000               # all orbit multisets for 46
"""
import argparse, itertools, json, sys, time
from pysat.solvers import Cadical153
from flip_graph import verify, target_tensor
n = 4; LA = 16
def load(p): return [tuple(t) for t in json.load(open(p))["scheme_bitmasks"]]

# ---- group as permutations of the 48 bit positions (a:0-15, b:16-31, c:32-47) ----
def perm_from_matrix_sandwich(X, Y, Z):
    """X,Y,Z permutation matrices (row lists). Returns bit permutation p with image_bit = p[src_bit]
    for (a,b,c) -> (X a Y^-1, Y b Z^-1, Z c X^-1)."""
    def perm_of(P):  # P e_j = e_{pi(j)} : column j has its 1 at row pi(j)
        pi = [None]*n
        for r in range(n):
            for c in range(n):
                if (P[r] >> c) & 1: pi[c] = r
        return pi
    def inv_perm(pi): q=[0]*n; [q.__setitem__(pi[i], i) for i in range(n)]; return q
    px, py, pz = perm_of(X), perm_of(Y), perm_of(Z)
    p = [None]*48
    # a: X a Y^-1 : entry (i,j) -> (px[i], py[j])   (Y^-1 on the right permutes columns by py)
    for i in range(n):
        for j in range(n):
            p[i*n+j]        = px[i]*n + py[j]
            p[16 + i*n+j]   = 16 + py[i]*n + pz[j]
            p[32 + i*n+j]   = 32 + pz[i]*n + px[j]
    return tuple(p)
def perm_tau():
    p = [None]*48
    for i in range(n):
        for j in range(n):
            p[i*n+j] = 16 + j*n+i        # a -> b^T
            p[16+i*n+j] = j*n+i          # b -> a^T
            p[32+i*n+j] = 32 + j*n+i     # c -> c^T
    return tuple(p)
def compose(p, q):  # (p o q)[i] = p[q[i]]
    return tuple(p[q[i]] for i in range(48))
def closure(gens):
    ident = tuple(range(48)); G = {ident}; frontier = [ident]
    while frontier:
        g = frontier.pop()
        for h in gens:
            k = compose(h, g)
            if k not in G: G.add(k); frontier.append(k)
    return sorted(G)
def apply(p, term):
    a, b, c = term; src = [(a >> i) & 1 for i in range(16)] + [(b >> i) & 1 for i in range(16)] + [(c >> i) & 1 for i in range(16)]
    dst = [0]*48
    for i in range(48):
        if src[i]: dst[p[i]] = 1
    A = sum(dst[i] << i for i in range(16)); B = sum(dst[16+i] << i for i in range(16)); C = sum(dst[32+i] << i for i in range(16))
    return (A, B, C)

def build_group():
    stab = json.load(open("../results/h1_stabilizer_alphatensor_444_rank47.json"))["elements"]
    gens = [perm_from_matrix_sandwich(*[list(m) for m in g]) for g in stab] + [perm_tau()]
    G = closure(gens)
    return G
def subgroups(G):
    """all subgroups of a small group given as permutation tuples"""
    ident = tuple(range(48)); subs = {frozenset([ident])}
    elems = list(G)
    for a in elems:
        for b in elems:
            H = closure([a, b]); subs.add(frozenset(H))
    return sorted(subs, key=len)

def encode(orbit_specs, G, conf, verbose=True, units=None):
    """orbit_specs: list of (H frozenset). Build SAT; return scheme or None. units: extra unit clauses."""
    T = target_tensor(4, 4, 4)
    nv = 0; clauses = list(units) if units else []
    def new():
        nonlocal nv; nv += 1; return nv
    def AND(x, y):
        z = new(); clauses.extend([[-z, x], [-z, y], [z, -x, -y]]); return z
    terms = []   # each term: list of 48 literals (variables)
    for H in orbit_specs:
        v = [new() for _ in range(48)]
        # H-invariance: v[h(i)] == v[i]
        for h in H:
            for i in range(48):
                j = h[i]
                if j != i: clauses.extend([[-v[i], v[j]], [v[i], -v[j]]])
        # nonzero factors for the rep
        clauses.append(v[0:16]); clauses.append(v[16:32]); clauses.append(v[32:48])
        # coset representatives: distinct images of v under G/H
        seen = set(); cosets = []
        for g in G:
            key = frozenset(compose(g, h) for h in H)
            if key in seen: continue
            seen.add(key); cosets.append(g)
        for g in cosets:
            terms.append([v[g_inv_index] for g_inv_index in inverse_index(g)])
    # tensor equation
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
    if verbose: print(f"  vars={nv} clauses={len(clauses)} terms={len(terms)}", flush=True)
    s = Cadical153(bootstrap_with=clauses); s.conf_budget(conf)
    t0 = time.time(); ok = s.solve_limited(); dt = time.time() - t0
    if not ok:
        s.delete(); return ("budget" if ok is None else "unsat"), dt
    model = set(l for l in s.get_model() if l > 0); s.delete()
    scheme = []
    for tv in terms:
        bitsv = [1 if x in model else 0 for x in tv]
        a = sum(bitsv[i] << i for i in range(16)); b = sum(bitsv[16+i] << i for i in range(16)); c = sum(bitsv[32+i] << i for i in range(16))
        scheme.append((a, b, c))
    return scheme, dt

def inverse_index(g):
    """term image under g has bit p[i] = src bit i  =>  image[j] = src[g^-1[j]]"""
    inv = [0]*48
    for i in range(48): inv[g[i]] = i
    return inv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbits", default=None, help="comma list of orbit sizes, e.g. 12,6,6,6,6,3,3,3,1,1")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--enumerate", action="store_true", help="try all orbit-size multisets (and subgroup choices) for --rank")
    ap.add_argument("--conf", type=int, default=5000000)
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument("--out", default="../results/group_sat")
    ap.add_argument("--from-scheme", default=None, help="derive exact orbit stabilizers from this Gamma-invariant scheme")
    ap.add_argument("--cube-orbit", type=int, default=None, help="cube-and-conquer over the free bits of this orbit index")
    ap.add_argument("--cube-stride", type=int, default=1)
    ap.add_argument("--cube-offset", type=int, default=0)
    ap.add_argument("--cube-all", action="store_true", help="do not stop at first SAT cube")
    a = ap.parse_args()
    G = build_group(); print("group order:", len(G))
    subs = subgroups(G); print("subgroups:", sorted(len(H) for H in subs))
    S47 = load("../results/alphatensor_444_rank47.json")
    # sanity: 47 invariant under G
    assert all(apply(g, t) in set(S47) for g in G for t in S47), "47 not invariant under built group"
    by_size = {}
    for H in subs: by_size.setdefault(len(G)//len(H), []).append(H)
    print("orbit sizes available:", {k: len(v) for k, v in by_size.items()})
    def try_spec(sizes, tag):
        # choose one subgroup per orbit size (first of each conjugacy-ish class); iterate over combos limited
        options = [by_size[s] for s in sizes]
        tried = 0
        for combo in itertools.product(*options):
            if tried >= a.max_tries: break
            tried += 1
            print(f"[{tag}] sizes={sizes} subgroup-orders={[len(H) for H in combo]}", flush=True)
            res, dt = encode(list(combo), G, a.conf)
            if isinstance(res, list):
                ok = verify(res, 4, 4, 4)
                print(f"  SAT in {dt:.1f}s -> rank {len(res)} VERIFIED={ok}", flush=True)
                if ok:
                    fn = f"{a.out}_rank{len(res)}_{tag}.json"
                    json.dump({"format":[4,4,4],"rank":len(res),"field":"F2","verified_brent_f2":True,
                               "found_by":"h1_group_sat from scratch under the order-12 group","orbit_sizes":sizes,
                               "scheme_bitmasks":[list(t) for t in res]}, open(fn,"w"), indent=1)
                    print("  SAVED", fn, flush=True); return True
            else:
                print(f"  {res} ({dt:.1f}s)", flush=True)
        return False
    if a.from_scheme and a.cube_orbit is not None:
        S = load(a.from_scheme); Sset = set(S); seen = set(); specs = []
        for t in S:
            if t in seen: continue
            orb = {apply(g, t) for g in G}; seen |= orb
            specs.append(frozenset(g for g in G if apply(g, t) == t))
        k = a.cube_orbit; Hk = specs[k]
        # bit-orbits of Hk = free coordinates of the representative
        seen_b = set(); borbs = []
        for i in range(48):
            if i in seen_b: continue
            ob = {i}; fr = [i]
            while fr:
                x = fr.pop()
                for h in Hk:
                    y = h[x]
                    if y not in ob: ob.add(y); fr.append(y)
            seen_b |= ob; borbs.append(sorted(ob))
        print(f"cube on orbit {k}: size {len(G)//len(Hk)}, free bits {len(borbs)} -> {2**len(borbs)} raw cubes", flush=True)
        cubes = []
        for bitsv in itertools.product([0, 1], repeat=len(borbs)):
            full = [0]*48
            for ob, bv in zip(borbs, bitsv):
                for i in ob: full[i] = bv
            if not any(full[0:16]) or not any(full[16:32]) or not any(full[32:48]): continue
            cubes.append(full)
        cubes = cubes[a.cube_offset::a.cube_stride]
        print(f"worker {a.cube_offset}/{a.cube_stride}: {len(cubes)} cubes", flush=True)
        for ci, full in enumerate(cubes):
            units = [[48*k+i+1] if full[i] else [-(48*k+i+1)] for i in range(48)]
            res, dt = encode(specs, G, a.conf, verbose=False, units=units)
            if isinstance(res, list):
                ok = verify(res, 4, 4, 4)
                print(f"  cube {ci}: SAT in {dt:.1f}s -> rank {len(res)} VERIFIED={ok} identical-to-input={set(res)==Sset}", flush=True)
                if ok:
                    fn = f"{a.out}_rank{len(res)}_cube{a.cube_offset}_{ci}.json"
                    json.dump({"format":[4,4,4],"rank":len(res),"field":"F2","verified_brent_f2":True,
                               "found_by":"h1_group_sat cube-and-conquer","scheme_bitmasks":[list(t) for t in res]}, open(fn,"w"), indent=1)
                    print("  SAVED", fn, flush=True)
                    if not a.cube_all: return
            else:
                print(f"  cube {ci}: {res} ({dt:.1f}s)", flush=True)
        return
    if a.from_scheme:
        # derive each orbit's exact stabilizer subgroup from a known Gamma-invariant scheme
        S = load(a.from_scheme); Sset = set(S); seen = set(); specs = []
        for t in S:
            if t in seen: continue
            orb = {apply(g, t) for g in G}; seen |= orb
            H = frozenset(g for g in G if apply(g, t) == t)
            specs.append(H)
        sizes = [len(G)//len(H) for H in specs]
        print("derived orbit sizes:", sizes, "subgroup orders:", [len(H) for H in specs], flush=True)
        res, dt = encode(specs, G, a.conf)
        if isinstance(res, list):
            ok = verify(res, 4, 4, 4); same = set(res) == Sset
            print(f"  SAT in {dt:.1f}s -> rank {len(res)} VERIFIED={ok} identical-to-input={same}", flush=True)
            if ok:
                fn = f"{a.out}_rank{len(res)}_fromspec.json"
                json.dump({"format":[4,4,4],"rank":len(res),"field":"F2","verified_brent_f2":True,
                           "found_by":"h1_group_sat from scratch (orbit stabilizers read off the 47)",
                           "scheme_bitmasks":[list(t) for t in res]}, open(fn,"w"), indent=1); print("  SAVED", fn, flush=True)
        else:
            print(f"  {res} ({dt:.1f}s)", flush=True)
    if a.orbits:
        sizes = [int(x) for x in a.orbits.split(",")]
        try_spec(sizes, "spec")
    if a.enumerate and a.rank:
        sizes_avail = sorted(by_size, reverse=True)
        def partitions(target, maxs):
            if target == 0: yield []; return
            for s in [x for x in sizes_avail if x <= maxs and x <= target]:
                for rest in partitions(target - s, s): yield [s] + rest
        parts = list(partitions(a.rank, max(sizes_avail)))
        print(f"{len(parts)} orbit-size multisets for rank {a.rank}", flush=True)
        # prefer few orbits (large symmetry) first
        parts.sort(key=len)
        for i, sizes in enumerate(parts):
            if try_spec(sizes, f"r{a.rank}_{i}"): break

if __name__ == "__main__":
    main()
