"""Job runner (PROTOKOL_v5 §5, §8).

Usage:  python -m ffexp.runner PLAN.json --out WYNIKI --workers K --device cuda

A plan is a JSON list of jobs. Each finished job writes (atomically):
  OUT/<stage>/runs/<id>.json          run record (no test metrics, ever)
  OUT/<stage>/ckpt/<id>.pt            weights + training info
  OUT/<stage>/preds/<id>.pt           validation predictions (non-blind stages)
For blind stages (C, E2, E9) all validation metrics, curves and predictions go to
  OUT/<stage>/zaslepione/<id>.json / .pt
which must not be opened before the final test evaluation (§5.5).
Finished jobs are skipped on restart; failures write <id>.error.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

BLIND_STAGES = {"C", "E2", "E9"}


def job_id(job) -> str:
    parts = [job["stage"], job["dataset"], job["arm"], f"w{job.get('w', 250)}", f"s{job['seed']}"]
    if "point" in job:
        parts.append(f"p{job['point']}")
    return "__".join(parts)


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions(torch):
    v = {"python": platform.python_version(), "torch": torch.__version__, "host": platform.node()}
    if torch.cuda.is_available():
        v["cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0)
        try:
            import subprocess
            v["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    v["code_commit"] = os.environ.get("FFEXP_COMMIT", "unknown")
    return v


def run_job(job, datasets, out, device, torch):
    from . import data, train

    stage = job["stage"]
    jid = job_id(job)
    blind = stage in BLIND_STAGES
    dirs = {k: os.path.join(out, stage, k) for k in ("runs", "ckpt", "preds", "zaslepione")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    run_path = os.path.join(dirs["runs"], jid + ".json")
    if os.path.exists(run_path):
        return "skip"

    x, y = datasets[job["dataset"]]
    sp = data.make_split(job["seed"])
    tr, es, va = sp.train_idx.to(device), sp.es_idx.to(device), sp.val_idx.to(device)
    task = data.ImageTask
    w = job.get("w", 250)
    t0 = time.time()
    if job["kind"] == "ff":
        l1, l2, info = train.train_ff(job["arm"], task, x[tr], y[tr], x[es], y[es], job["cfg"], job["seed"],
                                      job["emax"], w=w, device=device)
        val_pred = train.predict_ff(l1, l2, task, x[va], info)
        state = {"layer1": l1.state_dict(), "layer2": l2.state_dict()}
    else:
        net, info = train.train_bp(job["arm"], task, x[tr], y[tr], x[es], y[es], job["cfg"], job["seed"],
                                   job["emax"], w=w, device=device)
        val_pred = train.predict_bp(net, x[va], info)
        state = {"net": net.state_dict()}
    val_acc = (val_pred == y[va]).float().mean().item() * 100.0

    ckpt_path = os.path.join(dirs["ckpt"], jid + ".pt")
    torch.save({"job": job, "state": state, "info": info}, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)

    layers = ("layer1", "layer2") if job["kind"] == "ff" else ("bp",)
    summary = {
        "best_epoch": {k: info[k]["best_epoch"] for k in layers},
        "epochs_run": {k: info[k]["epochs_run"] for k in layers},
        "nan": {k: info[k]["nan"] for k in layers},
        "ok": {k: info[k]["ok"] for k in layers},
    }
    val_record = {
        "val_acc": val_acc,
        "es_acc": {k: info[k]["best_es_acc"] for k in layers},
        "curves": {k: info[k]["curve"] for k in layers},
    }
    pred_dir = dirs["zaslepione"] if blind else dirs["preds"]
    pred_path = os.path.join(pred_dir, jid + ".pt")
    torch.save({"val_idx": sp.val_idx, "val_pred": val_pred.cpu().to(torch.uint8)}, pred_path + ".tmp")
    os.replace(pred_path + ".tmp", pred_path)

    record = {
        "id": jid, "job": job, "n_params": info["n_params"], "train_time_sec": info["train_time_sec"],
        "wall_time_sec": time.time() - t0, "versions": _versions(torch), "ckpt_sha256": _sha256(ckpt_path),
        **summary,
    }
    if blind:
        _atomic_write_json(os.path.join(dirs["zaslepione"], jid + ".json"), {"id": jid, **val_record})
    else:
        record.update(val_record)
    _atomic_write_json(run_path, record)
    return "done"


def _lock_dir(out, stage):
    d = os.path.join(out, stage, "locks")
    os.makedirs(d, exist_ok=True)
    return d


def _claim(out, stage, jid) -> bool:
    """Dynamic work queue: a job is taken by the worker that creates its lock file first."""
    if os.path.exists(os.path.join(out, stage, "runs", jid + ".json")):
        return False
    try:
        fd = os.open(os.path.join(_lock_dir(out, stage), jid), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def clear_locks(plan_path, out):
    with open(plan_path) as f:
        stages = {j["stage"] for j in json.load(f)}
    for stage in stages:
        d = _lock_dir(out, stage)
        for name in os.listdir(d):
            os.remove(os.path.join(d, name))


def worker(rank, n_workers, plan_path, out, device_name):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from . import data

    device = torch.device(device_name)
    with open(plan_path) as f:
        jobs = json.load(f)
    datasets = {}
    for ds in sorted({j["dataset"] for j in jobs}):
        x, y = data.load(ds, "train")
        datasets[ds] = (x.to(device), y.to(device))
    for job in jobs:
        jid = job_id(job)
        if not _claim(out, job["stage"], jid):
            continue
        err_path = os.path.join(out, job["stage"], "runs", jid + ".error")
        try:
            status = run_job(job, datasets, out, device, torch)
            if os.path.exists(err_path):
                os.remove(err_path)
            print(f"[w{rank}] {status} {jid}", flush=True)
        except Exception:
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
            print(f"[w{rank}] ERROR {jid}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    clear_locks(args.plan, args.out)  # locks of an interrupted run are stale; finished runs are skipped
    if args.workers == 1:
        worker(0, 1, args.plan, args.out, args.device)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, args.workers, args.plan, args.out, args.device))
             for r in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode for p in procs))


if __name__ == "__main__":
    main()
