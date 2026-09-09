import json, time, itertools, sys
import h1_group_sat as H
from flip_graph import verify
G=H.build_group(); S=H.load("../results/alphatensor_444_rank47.json"); Sset=set(S)
seen=set(); specs=[]; reps=[]; orbs=[]
for t in S:
    if t in seen: continue
    orb={H.apply(g,t) for g in G}; seen|=orb
    specs.append(frozenset(g for g in G if H.apply(g,t)==t)); reps.append(t); orbs.append(orb)
def bits_of(t): a,b,c=t; return [(a>>i)&1 for i in range(16)]+[(b>>i)&1 for i in range(16)]+[(c>>i)&1 for i in range(16)]
pairs=[p for p in itertools.combinations(range(10),2)]
for free in pairs:
    free=set(free); units=[]
    for k,t in enumerate(reps):
        if k in free: continue
        for i,bit in enumerate(bits_of(t)): units.append([48*k+i+1] if bit else [-(48*k+i+1)])
    for k in free:
        for elem in orbs[k]:
            units.append([-(48*k+i+1) if bit else (48*k+i+1) for i,bit in enumerate(bits_of(elem))])
    res,dt=H.encode(specs,G,30000000,verbose=False,units=units)
    if isinstance(res,list):
        ok=verify(res,4,4,4); newterms=len(set(res)-Sset); shared=[sum(1 for u,v in itertools.combinations(res,2) if u[ax]==v[ax]) for ax in range(3)]
        print(f"free {sorted(free)} sizes {[len(orbs[k]) for k in free]}: SAT {dt:.1f}s verified={ok} new_terms={newterms} shared={shared}", flush=True)
        if ok and newterms: json.dump({"format":[4,4,4],"rank":47,"field":"F2","verified_brent_f2":True,"found_by":f"N1 rigidity scan, free orbits {sorted(free)}","scheme_bitmasks":[list(t) for t in res]},open(f"../results/n1_47_variant_{'_'.join(map(str,sorted(free)))}.json","w"),indent=1)
    else:
        print(f"free {sorted(free)} sizes {[len(orbs[k]) for k in free]}: {res} ({dt:.1f}s)", flush=True)
