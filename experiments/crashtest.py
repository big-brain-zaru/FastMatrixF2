import sys, json, time
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import target_tensor
from pysat.solvers import Cadical300, Glucose42, Cadical195
n=4; T=set(target_tensor(4,4,4))
groups=json.load(open('../results/census_n4_r46_cyclic.json'))['worklist']; wl=json.load(open('../results/B2_worklist.json'))
G=[tuple(g) for g in groups[7]['elements']]; classes=FP.subgroup_classes(G,FP.all_subgroups(G,n))
key=[w for w in wl['work'] if w['group_index']==7 and w['orbit_sizes']==[12,12,12,6,3,1]][0]['class_key']
cl,_,_,nv,_=EO.encode(n,G,[classes[ci][0] for ci in key],T,True,True)
S={"cadical300":Cadical300,"glucose42":Glucose42,"cadical195":Cadical195}[sys.argv[1]]
print("solving with",sys.argv[1],flush=True); s=S(bootstrap_with=cl); s.conf_budget(200000); t0=time.time(); r=s.solve_limited(); print(sys.argv[1],"->",r,"%.1fs"%(time.time()-t0),flush=True)
