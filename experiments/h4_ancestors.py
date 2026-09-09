"""H4: fingerprints of the 47's ancestors (schemes one/two plus-transitions above it) vs our pools."""
import json, random
from collections import Counter
n = 4
def load(p): return [tuple(t) for t in json.load(open(p))["scheme_bitmasks"]]
def mrank(x):
    rows=[(x>>(r*n))&0xF for r in range(n)]; rk=0
    for bit in range(n):
        piv=next((i for i,v in enumerate(rows) if (v>>bit)&1),None)
        if piv is None: continue
        p=rows.pop(piv); rk+=1; rows=[v^p if (v>>bit)&1 else v for v in rows]
    return rk
def sig(S): return tuple(tuple(sorted(Counter(mrank(t[ax]) for t in S).items())) for ax in range(3))
def plus(S, rng):
    """targeted plus transition: split term k on axis ax with term j's factor (rank +1)"""
    S = list(S)
    for _ in range(100):
        k, j, ax = rng.randrange(len(S)), rng.randrange(len(S)), rng.randrange(3)
        fk, fj = S[k][ax], S[j][ax]
        if fj and fj != fk:
            t = list(S[k]); u = list(S[k]); t[ax] = fj; u[ax] = fk ^ fj
            S[k] = tuple(t); S.append(tuple(u)); return S
    return S
rng = random.Random(0)
S47 = load("../results/alphatensor_444_rank47.json")
anc = {48: Counter(), 49: Counter(), 50: Counter()}
for _ in range(6000):
    S = plus(S47, rng); anc[48][sig(S)] += 1
    S = plus(S, rng);   anc[49][sig(S)] += 1
    S = plus(S, rng);   anc[50][sig(S)] += 1
for r in (48, 49, 50): print(f"rank-{r} ancestors of the 47: {len(anc[r])} distinct signatures; top: {anc[r].most_common(1)[0][1]} hits")
SS = load("../results/strassen_sq_rank49.json"); print("Strassen^2 signature among rank-49 ancestors:", anc[49].get(sig(SS), 0))
g50 = [ [tuple(t) for t in s] for s in json.load(open("../results/gated50_pool.json"))["schemes"] ]
g50sigs = Counter(sig(s) for s in g50)
overlap = set(g50sigs) & set(anc[50])
print(f"gated-50 pool: {len(g50sigs)} signatures; overlap with rank-50 ancestors: {len(overlap)} signatures covering {sum(g50sigs[s] for s in overlap)} schemes")
# rank-4 factor presence among ancestors
r4 = sum(c for s, c in anc[50].items() if any(dict(ax).get(4, 0) for ax in s))
print(f"rank-50 ancestors containing any rank-4 factor: {r4}/{sum(anc[50].values())}")
if overlap:
    idx = [i for i, s in enumerate(g50) if sig(s) in overlap]
    json.dump({"format":[4,4,4],"schemes":[[list(t) for t in g50[i]] for i in idx]}, open("../results/gated50_ancestor_matched.json","w"))
    print("saved matched gated-50 schemes:", len(idx))
