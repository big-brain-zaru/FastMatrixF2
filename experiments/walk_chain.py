import subprocess, sys
subprocess.run([sys.executable, "-u", "walk_seeds.py", "90", "../results/ON6bfs*_rank153_*.json", "../results/ON_walks6.jsonl", "6"])
subprocess.run([sys.executable, "-u", "walk_seeds.py", "45", "../results/ON5bfs*_rank93_*.json", "../results/ON_walks5b.jsonl", "5"])
print("walk chain done", flush=True)
