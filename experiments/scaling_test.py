import subprocess, time, json, sys
out=open("../results/scaling_test.jsonl","a")
for k in (16,24):
    t0=time.time()
    procs=[subprocess.Popen([sys.executable,"-u","scaling_one.py",str(k)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True) for _ in range(k)]
    res=[json.loads([l for l in p.communicate()[0].splitlines() if l.startswith("{")][-1]) for p in procs]
    wall=time.time()-t0; ts=[r["t_solve"] for r in res]
    rec={"concurrent":k,"wall":round(wall,1),"solve_min":min(ts),"solve_median":sorted(ts)[len(ts)//2],"solve_max":max(ts),"throughput_per_min":round(60*k/wall,2),"verdicts":sorted(set(r["verdict"] for r in res))}
    out.write(json.dumps(rec)+"\n"); out.flush(); print(rec,flush=True)
print("done",flush=True)
