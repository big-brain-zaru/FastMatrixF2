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
    P=[]; ok=True
    for t in S:
        tt=tau(t); p=next((u for u,s in enumerate(S) if s==tt),None)
        if p is None: ok=False; p=S.index(t)
        P.append(p)
    return P,ok
def reduce_all(S):
    S=[t for t in S if all(t)]
    ch=True
    while ch:
        ch=False
        for u in range(len(S)):
            for v in range(u+1,len(S)):
                a,b,c=S[u]; d,e,f=S[v]
                sa,sb,sc=a==d,b==e,c==f
                if sa+sb+sc>=2:
                    if sa and sb: S[u]=(a,b,c^f)
                    elif sa and sc: S[u]=(a,b^e,c)
                    else: S[u]=(a^d,b,c)
                    del S[v]; S=[t for t in S if all(t)]; ch=True; break
            if ch: break
    return S
rng=random.Random(1)
for trial in range(200):
    S=trivial_scheme(n,n,n); P,sym=partners(S); assert sym
    for step in range(20000):
        r=len(S); x,y=rng.randrange(r),rng.randrange(r)
        if x==y: continue
        sh=[ax for ax in range(3) if S[x][ax]==S[y][ax]]
        if len(sh) in (0,3): continue
        axis=rng.choice(sh); xb,yb=P[x],P[y]; xf,yf=xb==x,yb==y
        S0=list(S); mv=None
        A=[list(t) for t in S]
        if yb==x:
            if axis==2: A[x][1]^=A[y][1]; A[y][0]^=A[x][0]; mv="intra"
        elif xf or yf:
            if xf and yf: pass
            else:
                f,rr=(x,y) if xf else (y,x); rb=P[rr]
                if axis==0: A[f][2]^=A[rr][2]^A[rb][2]; A[rr][1]^=A[f][1]; A[rb][0]^=A[f][0]; mv="fixA"
                elif axis==1: A[f][2]^=A[rr][2]^A[rb][2]; A[rr][0]^=A[f][0]; A[rb][1]^=A[f][1]; mv="fixB"
        elif xb!=y and yb!=x and xb!=yb:
            if rng.random()<0.5: x,y,xb,yb=y,x,yb,xb
            if axis==0: A[x][2]^=A[y][2]; A[y][1]^=A[x][1]
            elif axis==1: A[x][0]^=A[y][0]; A[y][2]^=A[x][2]
            else: A[x][1]^=A[y][1]; A[y][0]^=A[x][0]
            A[xb]=list(tau(tuple(A[x]))); A[yb]=list(tau(tuple(A[y]))); mv="pair"
        if mv is None: continue
        S=[tuple(t) for t in A]
        if not verify(S,n,n,n):
            print("INVALID after move",mv,"axis",axis,"step",step,"trial",trial)
            print(" x",S0[x],"y",S0[y],"xb",S0[xb],"yb",S0[yb], "xf",xf,"yf",yf)
            raise SystemExit
        P2,sym2=partners(S)
        if not sym2:
            print("SYM BROKEN after move",mv,"axis",axis,"step",step); print(" x",S0[x],"y",S0[y],"xb",S0[xb],"yb",S0[yb],"xf",xf,"yf",yf); raise SystemExit
        if any(not all(t) for t in S) or rng.random()<0.1:
            S2=reduce_all(S)
            if len(S2)<len(S):
                S=S2
                if not verify(S,n,n,n): print("INVALID after reduce"); raise SystemExit
                P,sym=partners(S)
                if not sym: print("SYM BROKEN after reduce, rank",len(S)); raise SystemExit
            else: P,sym=partners(S)
print("all moves valid & symmetric over", trial+1, "trials")
