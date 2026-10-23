"""Forward-Forward and BP training (PROTOKOL_v5 §5.1, §5.2, §5.7, §7.6)."""
import copy
import math
import time

import torch
import torch.nn.functional as F

from . import models, streams

BATCH = 128
CLIP = 10.0
PATIENCE = 5
CHUNK = 1000


def _chunks(n, size=CHUNK):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


@torch.no_grad()
def layer1_scores(layer1, task, x):
    """G1 for every class: [n, C]."""
    out = torch.empty(len(x), task.n_classes, device=x.device)
    for c in range(task.n_classes):
        for s in _chunks(len(x)):
            out[s, c] = models.goodness(layer1(task.embed(x[s], c)))
    return out


@torch.no_grad()
def layer1_buffer(layer1, task, x):
    """Normalized layer-1 outputs for every class: [C, n, w1], computed in chunks of 1000."""
    buf = torch.empty(task.n_classes, len(x), layer1.out_dim, device=x.device)
    for c in range(task.n_classes):
        for s in _chunks(len(x)):
            buf[c, s] = models.normalize(layer1(task.embed(x[s], c)))
    return buf


@torch.no_grad()
def ff_scores(layer1, layer2, task, x, layer2_ok=True):
    """Classification scores sum_l G_l for every class (§5.1); G1 only if layer 2 failed (§7.6)."""
    scores = layer1_scores(layer1, task, x)
    if layer2_ok:
        for c in range(task.n_classes):
            for s in _chunks(len(x)):
                h = models.normalize(layer1(task.embed(x[s], c)))
                scores[s, c] += models.goodness(layer2(h))
    return scores


def _accuracy(scores, y):
    return (scores.argmax(1) == y).float().mean().item() * 100.0


def _neg_labels(y, n_classes, gen):
    offset = torch.randint(1, n_classes, (len(y),), generator=gen)
    return (y.cpu() + offset) % n_classes


def _train_ff_layer(layer, layer_no, inputs_pos, inputs_neg_fn, n, theta, cfg, seed, emax, eval_fn, full_batch,
                    patience=PATIENCE, callback=None):
    """Generic greedy FF layer training with early stopping.

    inputs_pos(idx) / inputs_neg_fn(idx, yneg) return the layer input for a batch.
    eval_fn() returns early-stopping accuracy for the current layer state.
    """
    device = next(layer.parameters()).device
    opt = torch.optim.AdamW(layer.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.999), eps=1e-8)
    best_acc, best_epoch, best_state = -math.inf, None, None
    curve, nan = [], False
    since_best = 0
    for epoch in range(emax):
        order = torch.randperm(n, generator=streams.order_gen(seed, layer_no, epoch))
        yneg_all = inputs_neg_fn.labels(streams.neg_gen(seed, layer_no, epoch))
        batch = n if full_batch else BATCH
        losses = []
        for start in range(0, n - batch + 1, batch):
            idx = order[start : start + batch]
            idx_d = idx.to(device)
            a_pos = layer(inputs_pos(idx_d))
            a_neg = layer(inputs_neg_fn(idx_d, yneg_all[idx].to(device)))
            loss = F.softplus(theta - models.goodness(a_pos)).mean() + F.softplus(models.goodness(a_neg) - theta).mean()
            if not torch.isfinite(loss):
                nan = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), CLIP)
            opt.step()
            losses.append(loss.item())
        if nan:
            break
        acc = eval_fn()
        curve.append({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "es_acc": acc})
        if callback is not None:
            callback(layer_no, epoch, acc)
        if acc > best_acc:
            best_acc, best_epoch, best_state = acc, epoch, copy.deepcopy(layer.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    if best_state is not None:
        layer.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_es_acc": best_acc if best_state else None, "nan": nan,
            "ok": best_state is not None, "epochs_run": len(curve), "curve": curve}


class _NegInputs:
    def __init__(self, fn, y, n_classes):
        self.fn, self.y, self.n_classes = fn, y, n_classes

    def labels(self, gen):
        return _neg_labels(self.y, self.n_classes, gen)

    def __call__(self, idx, yneg):
        return self.fn(idx, yneg)


def train_ff(arm, task, xtr, ytr, xes, yes, cfg, seed, emax, w=250, spiral=False, use_buffer=True,
             full_batch=False, device="cpu", patience=PATIENCE, callback=None):
    """Train one FF arm (§5.1–5.2). Returns (layer1, layer2, info).

    callback(layer_no, epoch, es_acc, layer1, layer2) is called after every epoch (diagnostics only)."""
    torch.manual_seed(streams.init_seed(seed))
    layer1, layer2 = models.build_ff(arm, w=w, in_dim=task.in_dim, spiral=spiral)
    layer1.to(device), layer2.to(device)
    cb = None if callback is None else (lambda ln, ep, acc: callback(ln, ep, acc, layer1, layer2))
    theta1 = cfg["c"] * layer1.out_dim
    theta2 = cfg["c"] * layer2.out_dim
    t0 = time.time()
    n = len(xtr)

    # Layer 1
    neg1 = _NegInputs(lambda idx, yneg: task.embed(xtr[idx], yneg), ytr, task.n_classes)
    info1 = _train_ff_layer(
        layer1, 1, lambda idx: task.embed(xtr[idx], ytr[idx]), neg1, n, theta1, cfg, seed, emax,
        lambda: _accuracy(layer1_scores(layer1, task, xes), yes), full_batch, patience, cb)
    for p in layer1.parameters():
        p.requires_grad_(False)
    layer1_ok = info1["ok"]

    info2 = {"best_epoch": None, "best_es_acc": None, "nan": False, "ok": False, "epochs_run": 0, "curve": []}
    if layer1_ok:
        g1_es = layer1_scores(layer1, task, xes)
        es_buf = layer1_buffer(layer1, task, xes)
        if use_buffer:
            buf = layer1_buffer(layer1, task, xtr)
            pos_fn = lambda idx: buf[ytr[idx], idx]
            neg2 = _NegInputs(lambda idx, yneg: buf[yneg, idx], ytr, task.n_classes)
        else:
            pos_fn = lambda idx: models.normalize(layer1(task.embed(xtr[idx], ytr[idx])))
            neg2 = _NegInputs(lambda idx, yneg: models.normalize(layer1(task.embed(xtr[idx], yneg))), ytr,
                              task.n_classes)

        @torch.no_grad()
        def eval2():
            scores = g1_es.clone()
            for c in range(task.n_classes):
                scores[:, c] += models.goodness(layer2(es_buf[c]))
            return _accuracy(scores, yes)

        info2 = _train_ff_layer(layer2, 2, pos_fn, neg2, n, theta2, cfg, seed, emax, eval2, full_batch, patience, cb)
        if use_buffer:
            del buf
    info = {
        "layer1": info1, "layer2": info2,
        "layer1_ok": layer1_ok, "layer2_ok": info2["ok"],
        "train_time_sec": time.time() - t0,
        "n_params": models.n_params(layer1, layer2),
        "theta": [theta1, theta2],
        "seed": seed,
    }
    return layer1, layer2, info


def predict_ff(layer1, layer2, task, x, info):
    """Predictions with the NaN rules of §7.6."""
    if not info["layer1_ok"]:
        gen = streams.generator(info["seed"], streams.INIT, "fallback")
        return torch.randint(0, task.n_classes, (len(x),), generator=gen).to(x.device)
    return ff_scores(layer1, layer2, task, x, layer2_ok=info["layer2_ok"]).argmax(1)


def train_bp(arm, task, xtr, ytr, xes, yes, cfg, seed, emax, w=250, spiral=False, full_batch=False, device="cpu",
             patience=PATIENCE, callback=None):
    """BP arm (§5.7): cross-entropy, AdamW, clipping, early stopping on es accuracy."""
    torch.manual_seed(streams.init_seed(seed))
    net = models.BPNet(models.BP_ARMS[arm], n_classes=task.n_classes, w=w,
                       in_dim=2 if spiral else 784, spiral=spiral).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.999), eps=1e-8)
    n, t0 = len(xtr), time.time()
    best_acc, best_epoch, best_state, since_best, curve, nan = -math.inf, None, None, 0, [], False
    for epoch in range(emax):
        order = torch.randperm(n, generator=streams.order_gen(seed, 0, epoch))
        batch = n if full_batch else BATCH
        losses = []
        for start in range(0, n - batch + 1, batch):
            idx = order[start : start + batch].to(device)
            loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                nan = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            losses.append(loss.item())
        if nan:
            break
        with torch.no_grad():
            acc = _accuracy(net(xes), yes)
        curve.append({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "es_acc": acc})
        if callback is not None:
            callback(0, epoch, acc, net)
        if acc > best_acc:
            best_acc, best_epoch, best_state, since_best = acc, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    info = {"bp": {"best_epoch": best_epoch, "best_es_acc": best_acc if best_state else None, "nan": nan,
                   "ok": best_state is not None, "epochs_run": len(curve), "curve": curve},
            "train_time_sec": time.time() - t0, "n_params": models.n_params(net), "seed": seed}
    return net, info


def predict_bp(net, x, info):
    if not info["bp"]["ok"]:
        gen = streams.generator(info["seed"], streams.INIT, "fallback")
        return torch.randint(0, net.readout.out_features, (len(x),), generator=gen).to(x.device)
    with torch.no_grad():
        return net(x).argmax(1)
