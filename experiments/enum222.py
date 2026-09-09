"""All rank-7 schemes for <2,2,2> over F2 as term SETS: SAT enumeration with lexicographically increasing
terms (each set once) and assignment blocking; then classification under GL(2,2)^3 x S3."""
import itertools, json, time
from pysat.solvers import Cadical195
import glsym as GL
from sym_break import lex_leq
from flip_graph import verify, target_tensor
n=2; L=12; T=set(target_tensor(2,2,2)); k=7
st={"nv":0}; cl=[]
def new(): st["nv"]+=1; return st["nv"]
terms=[[new() for _ in range(L)] for _ in range(k)]
for tv in terms: cl.append(tv[0:4]); cl.append(tv[4:8]); cl.append(tv[8:12])
def AND(x,y):
    z=new(); cl.extend([[-z,x],[-z,y],[z,-x,-y]]); return z
for ia in range(4):
    for ib in range(4):
        ab=[AND(tv[ia],tv[4+ib]) for tv in terms]
        for ic in range(4):
            lits=[AND(ab[i],terms[i][8+ic]) for i in range(k)]
            cur=lits[0]
            for l in lits[1:]:
                z=new(); cl.extend([[-z,cur,l],[-z,-cur,-l],[z,-cur,l],[z,cur,-l]]); cur=z
            cl.append([cur] if (ia,ib,ic) in T else [-cur])
# strict lex order term_i < term_{i+1}: lex_leq plus "not equal"
for i in range(k-1):
    lex_leq(cl, new, terms[i][::-1], terms[i+1][::-1])   # MSB-first
    ne=[new() for _ in range(L)]
    for j in range(L):
        x,y,d=terms[i][j],terms[i+1][j],ne[j]
        cl.extend([[-d,x,y],[-d,-x,-y],[d,-x,y],[d,x,-y]])
    cl.append(ne)
s=Cadical195(bootstrap_with=cl); found=[]; t0=time.time()
while s.solve():
    m=set(l for l in s.get_model() if l>0)
    sch=[(sum(((tv[i] in m)<<i) for i in range(4)), sum(((tv[4+i] in m)<<i) for i in range(4)), sum(((tv[8+i] in m)<<i) for i in range(4))) for tv in terms]
    S=frozenset(sch); assert len(S)==7 and verify(list(S),2,2,2); found.append(S)
    s.add_clause([-v if v in m else v for tv in terms for v in tv])
    if len(found)%200==0: print("  found",len(found),"%.0fs"%(time.time()-t0),flush=True)
print("rank-7 schemes for <2,2,2> over F2 (distinct term sets):",len(found),"in %.0fs"%(time.time()-t0),flush=True)
GL2=[M for M in itertools.product(range(1,4),repeat=2) if GL.minv(M,2) is not None]
def cyc(t): return (t[1],t[2],t[0])
def tr(m): return sum((((m>>(2*i+j))&1)<<(2*j+i)) for i in range(2) for j in range(2))
def tau(t): return (tr(t[1]),tr(t[0]),tr(t[2]))
def images(S):
    variants=set()
    V=S
    for _ in range(3):
        variants.add(V); variants.add(frozenset(tau(t) for t in V)); V=frozenset(cyc(t) for t in V)
    out=set()
    for V in variants:
        for X in GL2:
            for Y in GL2:
                for Z in GL2: out.add(frozenset(GL.sandwich_term(t,X,Y,Z,2) for t in V))
    return out
classes=[]; seen=set()
for S in found:
    if S in seen: continue
    orb=images(S); seen|=orb; classes.append(len(orb))
print("equivalence classes under GL(2,2)^3 x S3:",len(classes),"orbit sizes",classes,"| union covers all found:",seen>=set(found),flush=True)
json.dump({"count":len(found),"classes":len(classes),"orbit_sizes":classes,"schemes":[[list(t) for t in sorted(S)] for S in found]},open("../results/all_222_rank7_F2.json","w"))
print("done",flush=True)
