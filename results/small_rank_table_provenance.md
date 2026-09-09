# small_rank_table.json provenance (2026-09-05)
Computed by `fp_filter.small_rank_unsat` (plain Brent encoding over F2, CaDiCaL):
- <2,2,2> rank <= 6: UNSAT in 35.9 s  =>  rank >= 7;  rank <= 7: SAT in 0.1 s  =>  rank = 7 over F2 (independently re-derived Winograd/Hopcroft-Kerr).
- <2,2,4> rank <= 13: undecided after 5,000,000 conflicts (402.9 s); rank <= 14: SAT in 34.3 s. Not in the table; the <2,2,4> count bound stays at the flattening value 8 (the <2,2,2> sub-restriction bound 7 also applies and is weaker).
- <2,4,4>: not attempted; flattening bound 16.
