"""H2: canonical Strassen x Strassen over F2 -- tau-symmetry check, structured deletion set,
and export for symmetric LNS refill."""
import json
from flip_graph import verify
n = 4
# canonical Strassen (F2): a,b as 2x2 bitmasks row-major (A11,A12,A21,A22); c in our (k,i) layout.
# M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1 ; M2=(A21+A22)B11: C21+=M2, C22-=M2 ; M3=A11(B12-B22): C12,C22
# M4=A22(B21-B11): C11,C21 ; M5=(A11+A12)B22: C11-=,C12+= ; M6=(A21-A11)(B11+B12): C22 ; M7=(A12-A22)(B21+B22): C11
A11,A12,A21,A22 = 1,2,4,8
def cmask(*entries):   # entries as (k,i) -> our c layout bit k*2+i, where C_{ik} is output entry
    m = 0
    for (i,k) in entries: m |= 1 << (k*2+i)
    return m
ST = [
 (A11^A22, A11^A22, cmask((0,0),(1,1))),
 (A21^A22, A11,     cmask((1,0),(1,1))),
 (A11,     A12^A22, cmask((0,1),(1,1))),
 (A22,     A21^A11, cmask((0,0),(1,0))),
 (A11^A12, A22,     cmask((0,0),(0,1))),
 (A21^A11, A11^A12, cmask((1,1),)),
 (A12^A22, A21^A22, cmask((0,0),)),
]
assert verify(ST, 2, 2, 2), "canonical Strassen failed verification in this layout"
def kron(u, v):
    x = 0
    for I in range(2):
        for J in range(2):
            if (u >> (I*2+J)) & 1:
                for i in range(2):
                    for j in range(2):
                        if (v >> (i*2+j)) & 1: x |= 1 << ((2*I+i)*n + (2*J+j))
    return x
SS = [(kron(a1,a2), kron(b1,b2), kron(c1,c2)) for (a1,b1,c1) in ST for (a2,b2,c2) in ST]
assert verify(SS, 4, 4, 4)
def tr(x):
    y = 0
    for i in range(n):
        for j in range(n):
            if (x >> (i*n+j)) & 1: y |= 1 << (j*n+i)
    return y
sym = all((tr(b), tr(a), tr(c)) in set(SS) for (a,b,c) in SS)
def mrank(x):
    rows=[(x>>(r*n))&0xF for r in range(n)]; rk=0
    for bit in range(n):
        piv=next((i for i,v in enumerate(rows) if (v>>bit)&1),None)
        if piv is None: continue
        p=rows.pop(piv); rk+=1; rows=[v^p if (v>>bit)&1 else v for v in rows]
    return rk
hi = [i for i,(a,b,c) in enumerate(SS) if max(mrank(a),mrank(b),mrank(c)) >= 2]
print("Strassen^2 valid; tau-symmetric:", sym, "; terms with a factor of rank>=2:", len(hi))
fixed = sum(1 for (a,b,c) in SS if b == tr(a) and c == tr(c))
print("fixed points:", fixed, "pairs:", (49-fixed)//2)
json.dump({"format":[4,4,4],"rank":49,"field":"F2","verified_brent_f2":True,"tau_symmetric":sym,
           "scheme_bitmasks":[list(t) for t in SS], "high_rank_term_indices": hi},
          open("../results/strassen_sq_rank49.json","w"), indent=1)
print("saved ../results/strassen_sq_rank49.json")
