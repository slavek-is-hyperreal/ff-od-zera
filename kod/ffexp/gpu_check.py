"""On-instance checks (PROTOKOL_v5 §8, §9.1.6, §9.1.9, §9.3).

Usage:  python -m ffexp.gpu_check --out LOG.json [--k-list 1,2,4,6,8]

1. test 9.1.6 (bitwise reproducibility) and 9.1.9 (buffer tolerance) on the target device;
2. throughput on random data with the layer-1 buffer for k parallel processes; the k with the
   highest runs/hour is reported (to be fixed before the mini-pilot).
"""
import argparse
import json
import multiprocessing as mp
import os
import time


def _setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set in this process
        pass
    torch.use_deterministic_algorithms(True)
    return torch


def checks(device):
    torch = _setup()
    from . import data, models, train

    x, y = data.load("mnist", "train")
    sp = data.make_split(40001)
    tr, es = sp.train_idx[:5000], sp.es_idx[:1000]
    args = (x[tr].to(device), y[tr].to(device), x[es].to(device), y[es].to(device))
    cfg = {"lr": 1e-3, "wd": 1e-4, "c": 0.25}
    a = train.train_ff("K-LGT", data.ImageTask, *args, cfg, 40001, emax=2, device=device)
    b = train.train_ff("K-LGT", data.ImageTask, *args, cfg, 40001, emax=2, device=device)
    bitwise = all(torch.equal(va, vb) for la, lb in zip(a[:2], b[:2])
                  for va, vb in zip(la.state_dict().values(), lb.state_dict().values()))
    c = train.train_ff("K-LGT", data.ImageTask, *args, cfg, 40001, emax=1, use_buffer=True, device=device)
    d = train.train_ff("K-LGT", data.ImageTask, *args, cfg, 40001, emax=1, use_buffer=False, device=device)
    xes = args[2]
    with torch.no_grad():
        hc = c[1](models.normalize(c[0](data.ImageTask.embed(xes, 3))))
        hd = d[1](models.normalize(d[0](data.ImageTask.embed(xes, 3))))
    maxdiff = (hc - hd).abs().max().item()
    same_pred = torch.equal(train.predict_ff(c[0], c[1], data.ImageTask, xes, c[2]),
                            train.predict_ff(d[0], d[1], data.ImageTask, xes, d[2]))
    return {"9.1.6_bitwise": bitwise, "9.1.9_maxdiff": maxdiff, "9.1.9_same_pred": same_pred,
            "9.1.9_pass": maxdiff < 1e-5 and same_pred}


def _one(args):
    device, seed = args
    torch = _setup()
    from . import data, train

    g = torch.Generator().manual_seed(seed)
    x = torch.rand(60000, 784, generator=g).to(device)
    y = torch.randint(0, 10, (60000,), generator=g).to(device)
    t = time.time()
    train.train_ff("K-LGT", data.ImageTask, x[:50000], y[:50000], x[50000:55000], y[50000:55000],
                   {"lr": 1e-3, "wd": 1e-4, "c": 0.25}, seed, emax=1, device=device)
    return time.time() - t


def throughput(device, k_list):
    out = {}
    ctx = mp.get_context("spawn")
    for k in k_list:
        t = time.time()
        with ctx.Pool(k) as pool:
            per = pool.map(_one, [(device, i) for i in range(k)])
        wall = time.time() - t
        out[k] = {"wall_sec": wall, "per_job_sec": per, "layer_epochs_per_hour": 2 * k / wall * 3600}
    best = max(out, key=lambda k: out[k]["layer_epochs_per_hour"])
    return out, best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-list", default="1,2,4,6,8")
    a = ap.parse_args(argv)
    torch = _setup()
    res = {"device": torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
           "vcpu": os.cpu_count()}
    res["checks"] = checks(a.device)
    ks = [k for k in map(int, a.k_list.split(",")) if k <= os.cpu_count()]
    res["throughput"], res["k_best"] = throughput(a.device, ks)
    from .runner import _versions
    res["versions"] = _versions(torch)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"checks": res["checks"], "k_best": res["k_best"],
                      "throughput": {k: round(v["layer_epochs_per_hour"]) for k, v in res["throughput"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
