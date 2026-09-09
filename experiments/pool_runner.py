"""Work-queue runner for SAT instance sweeps (Phase 0 item 1).

Replaces static striding. The master keeps K instances in flight, each in its OWN process, hands
out work from a cost-sorted queue (expensive first, so the tail is short), enforces a wall-time cap
per instance by killing that process only (CaDiCaL ignores interrupts and a conflict budget does not
bound wall time), and appends one JSON record per instance to a single log keyed by a faithful
instance key. Per-process start-up (imports) costs ~1 s, negligible for instances of >= 10 s; tiny
instances should be batched by the task module (an instance may be a chunk).

The task module must expose
    parse_args(argv) -> task_args            (optional)
    instances(task_args) -> list[dict]       each with unique 'key' and optional 'cost'
    run_instance(inst, task_args) -> dict    the result record (outcome, t, ...)

Usage:
  python pool_runner.py --task census_task --workers 6 --wall 600 --log ../results/X.jsonl [-- task args]
"""
import argparse, importlib, json, multiprocessing as mp, os, queue, time


def _run_one(task_name, task_args, inst, q_out):
    mod = importlib.import_module(task_name)
    t0 = time.time()
    try:
        res = mod.run_instance(inst, task_args)
    except Exception as e:  # noqa
        res = {"outcome": "error", "error": repr(e)[:300]}
    res["t_wall"] = round(time.time() - t0, 1)
    q_out.put((inst["key"], res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wall", type=float, default=600, help="seconds per instance before its process is killed")
    ap.add_argument("--log", required=True)
    ap.add_argument("--resume", action="store_true", help="skip keys already present in --log")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    mod = importlib.import_module(a.task)
    task_args = mod.parse_args(rest) if hasattr(mod, "parse_args") else rest
    insts = mod.instances(task_args)
    done = set()
    if a.resume and os.path.exists(a.log):
        for l in open(a.log):
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    pending = [i for i in insts if i["key"] not in done]
    pending.sort(key=lambda i: -i.get("cost", 0))
    print("[pool] %d instances (%d already done), %d workers, wall cap %.0fs" % (len(pending), len(done), a.workers, a.wall), flush=True)
    log = open(a.log, "a")
    q_out = mp.Queue()
    in_flight = {}   # key -> (proc, t_start, inst)
    stats = {"done": 0, "wall": 0, "error": 0}
    t_report = time.time()
    while pending or in_flight:
        while pending and len(in_flight) < a.workers:
            inst = pending.pop(0)
            p = mp.Process(target=_run_one, args=(a.task, task_args, inst, q_out), daemon=True); p.start()
            in_flight[inst["key"]] = (p, time.time(), inst)
        try:
            key, res = q_out.get(timeout=1)
            p, t0, inst = in_flight.pop(key); p.join(5)
            rec = {"key": key, **{k: v for k, v in inst.items() if k != "key"}, **res}
            log.write(json.dumps(rec) + "\n"); log.flush(); stats["done"] += 1
            if res.get("outcome") == "error": stats["error"] += 1
            print("[pool] %s -> %s (%.0fs) | done %d, in flight %d, pending %d" % (
                key, res.get("outcome"), res.get("t_wall", 0), stats["done"], len(in_flight), len(pending)), flush=True)
        except queue.Empty:
            pass
        now = time.time()
        for key, (p, t0, inst) in list(in_flight.items()):
            if now - t0 > a.wall:
                p.terminate(); p.join(5)
                in_flight.pop(key)
                rec = {"key": key, **{k: v for k, v in inst.items() if k != "key"}, "outcome": "wall", "t_wall": round(now - t0, 1)}
                log.write(json.dumps(rec) + "\n"); log.flush(); stats["wall"] += 1
                print("[pool] %s -> WALL cap (%.0fs), killed" % (key, now - t0), flush=True)
            elif not p.is_alive() and p.exitcode not in (0, None):
                in_flight.pop(key)
                rec = {"key": key, **{k: v for k, v in inst.items() if k != "key"}, "outcome": "crash", "exitcode": p.exitcode, "t_wall": round(now - t0, 1)}
                log.write(json.dumps(rec) + "\n"); log.flush(); stats["error"] += 1
                print("[pool] %s -> CRASH exit %s" % (key, p.exitcode), flush=True)
        if time.time() - t_report > 300:
            t_report = time.time()
            print("[pool] status: done %d, wall-killed %d, errors %d, in flight %d, pending %d" % (
                stats["done"], stats["wall"], stats["error"], len(in_flight), len(pending)), flush=True)
    print("[pool] finished: %s" % stats, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
