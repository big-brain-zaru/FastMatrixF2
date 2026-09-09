"""Import fast-matrix-multiplication schemes from the Lille FMM database (.mpl tensor files).

Format: Tensor:=TriadSet([Triad([Matrix(n,n,[[..]]), Matrix(n,n,[[..]]), Matrix(n,n,[[..]])]), ...])
Entries are integers (typically -1/0/1). We reduce mod 2 to get an F2 scheme and search the index
convention (which of the 8 natural (axis, transpose) layouts) that satisfies the exact Brent
equations, rather than assuming one. Nothing is saved unless it verifies.
"""
import re, json, sys, itertools
from flip_graph import verify

def parse_matrices(text, n):
    """Return list of n x n integer matrices in file order."""
    out = []
    # Matrix(n, n, [[a,b,...],[...]])  -- tolerate whitespace
    pat = re.compile(r"Matrix\(\s*%d\s*,\s*%d\s*,\s*(\[\[.*?\]\])\s*\)" % (n, n), re.S)
    for m in pat.finditer(text):
        body = m.group(1)
        rows = re.findall(r"\[([-0-9,\s]+)\]", body)
        rows = [r for r in rows if r.strip()]
        if len(rows) != n: continue
        M = [[int(v) for v in r.split(",")] for r in rows]
        if all(len(r) == n for r in M): out.append(M)
    return out

def to_mask(M, n, order):
    """order: 'rc' -> bit index i*n+j ; 'cr' -> j*n+i (transpose)."""
    x = 0
    for i in range(n):
        for j in range(n):
            if M[i][j] % 2:
                x |= 1 << (i*n + j if order == 'rc' else j*n + i)
    return x

def main(path, n, out_json, label):
    text = open(path).read()
    # only the TriadSet region
    k = text.find("TriadSet")
    mats = parse_matrices(text[k:] if k >= 0 else text, n)
    if len(mats) % 3: mats = mats[:len(mats) - len(mats) % 3]
    triads = [tuple(mats[i:i+3]) for i in range(0, len(mats), 3)]
    print(f"{label}: parsed {len(mats)} matrices -> {len(triads)} triads (rank {len(triads)})")
    if not triads: sys.exit("no triads parsed")
    # search layout: permutation of the three slots x transpose flags
    for perm in itertools.permutations(range(3)):
        for flags in itertools.product(['rc','cr'], repeat=3):
            sch = []
            for T in triads:
                a = to_mask(T[perm[0]], n, flags[0])
                b = to_mask(T[perm[1]], n, flags[1])
                c = to_mask(T[perm[2]], n, flags[2])
                sch.append((a, b, c))
            if any(not (x and y and z) for x, y, z in sch): continue
            if verify(sch, n, n, n):
                print(f"  VERIFIED over F2 with slot perm {perm}, transpose flags {flags}")
                json.dump({"format": [n, n, n], "rank": len(sch), "field": "F2",
                           "verified_brent_f2": True, "source": path, "label": label,
                           "layout": {"slot_perm": list(perm), "flags": list(flags)},
                           "scheme_bitmasks": [list(t) for t in sch]}, open(out_json, "w"), indent=1)
                print("  saved", out_json)
                return 0
    print("  NO layout satisfied the Brent equations over F2 -- not saved")
    return 2

if __name__ == "__main__":
    p, n, o, lab = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    sys.exit(main(p, n, o, lab))
