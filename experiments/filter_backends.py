import json,time,sys,symgroup as SG,fp_filter as FP
from pysat.solvers import Cadical153, Cadical195, Cadical300, Glucose42
from flip_graph import target_tensor
n=4;T=set(target_tensor(n,n,n))
groups=json.load(open('../results/census_n4_r46_cyclic.json'))['worklist']
filt=[json.loads(l) for l in open('../results/B2_filter.jsonl')]
bud=[r for r in filt if r['filter']=='budget' and r['group_index'] in (6,24,31)][:6]
print("filter-budget instances sampled:",len(bud),flush=True)
for r in bud:
    G=[tuple(g) for g in groups[r['group_index']]['elements']]
    cl_=FP.subgroup_classes(G,FP.all_subgroups(G,n)); slots=[cl_[ci][0] for ci in r['class_key']]
    cl,nv,st=FP.build_filter(n,G,slots,T,orbit_eqs=True)
    row=[]
    for name,S in [("cadical153",Cadical153),("cadical195",Cadical195),("glucose42",Glucose42)]:
        s=S(bootstrap_with=cl); s.conf_budget(300000); t0=time.time(); ok=s.solve_limited(); dt=time.time()-t0; s.delete()
        row.append("%s=%s/%.1fs"%(name,'budget' if ok is None else ('sat' if ok else 'unsat'),dt))
    print("g%d %s vars %d:"%(r['group_index'],r['orbit_sizes'],nv)," ".join(row),flush=True)
print("done",flush=True)
