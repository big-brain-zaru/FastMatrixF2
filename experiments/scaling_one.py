import sys, json, time
import symgroup as SG, fp_filter as FP, encode_opt as EO
from flip_graph import target_tensor
from pysat.solvers import Cadical195
n=4; T=set(target_tensor(4,4,4))
groups=json.load(open('../results/census_n4_r46_cyclic.json'))['worklist']; wl=json.load(open('../results/B2_worklist.json'))
G=[tuple(g) for g in groups[33]['elements']]; classes=FP.subgroup_classes(G,FP.all_subgroups(G,n))
sizes=[4,4,4,4,4,4,4,4,4,2,2,2,1,1,1,1]
key=[w for w in wl['work'] if w['group_index']==33 and w['orbit_sizes']==sizes][0]['class_key']
cl,_,_,nv,_=EO.encode(n,G,[classes[ci][0] for ci in key],T,True,True)
s=Cadical195(bootstrap_with=cl); s.conf_budget(2000000); t0=time.time(); r=s.solve_limited(); dt=time.time()-t0
print(json.dumps({"k":sys.argv[1],"verdict":str(r),"t_solve":round(dt,1)}),flush=True)
