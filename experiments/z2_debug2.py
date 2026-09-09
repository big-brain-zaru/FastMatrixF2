import random
from flip_graph import verify, trivial_scheme
n=3
def tr(x):
    y=0
    for i in range(n):
        for j in range(n):
            if (x>>(i*n+j))&1: y|=1<<(j*n+i)
    return y
def tau(t): return (tr(t[1]),tr(t[0]),tr(t[2]))
def partners(S):
    P=[-1]*len(S); ok=True
    for t in range(len(S)):
        if P[t]!=-1: continue
        tt=tau(S[t]); p=next((u for u in range(t,len(S)) if P[u]==-1 and S[u]==tt),None)
        if p is None: ok=False; P[t]=t
        else: P[t]=p; P[p]=t
    return P,ok
rng=random.Random(3)
for trial in range(300):
    S=trivial_scheme(n,n,n); P,sym=partners(S)
    for step in range(3000):
        r=len(S)
        if rng.random()<0.3:   # symmetric plus
            k,jj,ax=rng.randrange(r),rng.randrange(r),rng.randrange(3)
            kb=P[k]; fk=S[k][ax]; fj=S[jj][ax]
            if fj!=fk and fj and kb!=k and sym:
                A=[list(t) for t in S]
                new=list(A[k]); A[k][ax]=fj; new[ax]=fk^fj
                A.append(new)
                A[kb]=list(tau(tuple(A[k]))); A.append(list(tau(tuple(new))))
                P=P+[len(A)-1,len(A)-2]
                S2=[tuple(t) for t in A]
                if not verify(S2,n,n,n):
                    print("INVALID after plus: k",S[k],"kb",S[kb],"tau(k)",tau(S[k]),"P[kb]",P[kb],"k",k,"kb",kb)
                    print(" P consistent?", P[kb]==k, " dup terms:", len(S)-len(set(S)))
                    raise SystemExit
                S=S2
                P2,sym2=partners(S)
                pass
            continue
        x,y=rng.randrange(r),rng.randrange(r)
        if x==y: continue
        sh=[ax for ax in range(3) if S[x][ax]==S[y][ax]]
        if len(sh) in (0,3): continue
        axis=rng.choice(sh); xb,yb=P[x],P[y]; xf,yf=xb==x,yb==y
        A=[list(t) for t in S]; mv=None
        if yb==x:
            if axis==2: A[x][1]^=A[y][1]; A[y][0]^=A[x][0]; mv="intra"
        elif xf or yf:
            if not(xf and yf):
                f,rr=(x,y) if xf else (y,x); rb=P[rr]
                if axis==0: A[f][2]^=A[rr][2]^A[rb][2]; A[rr][1]^=A[f][1]; A[rb][0]^=A[f][0]; mv="fixA"
                elif axis==1: A[f][2]^=A[rr][2]^A[rb][2]; A[rr][0]^=A[f][0]; A[rb][1]^=A[f][1]; mv="fixB"
        elif xb!=y and yb!=x and xb!=yb:
            if axis==0: A[x][2]^=A[y][2]; A[y][1]^=A[x][1]
            elif axis==1: A[x][0]^=A[y][0]; A[y][2]^=A[x][2]
            else: A[x][1]^=A[y][1]; A[y][0]^=A[x][0]
            A[xb]=list(tau(tuple(A[x]))); A[yb]=list(tau(tuple(A[y]))); mv="pair"
        if mv is None: continue
        S2=[tuple(t) for t in A]
        if not verify(S2,n,n,n):
            print("INVALID after",mv,"axis",axis); print(" x",S[x],"y",S[y],"xb",S[xb],"yb",S[yb],"P[xb]",P[xb],"P[yb]",P[yb],"x,y,xb,yb",x,y,xb,yb)
            print(" dup terms:", len(S)-len(set(S))); raise SystemExit
        S=S2
        if any(not all(t) for t in S):
            S=[t for t in S if all(t)]; P,sym=partners(S)
print("no invalid state found")
