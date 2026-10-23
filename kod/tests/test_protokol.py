"""Unit tests required by PROTOKOL_v5 §9.1 (numbers in test names refer to §9.1.x)."""
import inspect
import os

import pytest
import torch

from ffexp import data, models, streams, train

CFG = {"lr": 1e-3, "wd": 1e-4, "c": 0.25}


@pytest.fixture(scope="module")
def small():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    x, y = data.load("mnist", "train")
    sp = data.make_split(40001)
    tr, es = sp.train_idx[:2000], sp.es_idx[:500]
    return x[tr], y[tr], x[es], y[es]


def test_9_1_1_param_counts():
    for arm in models.P_ARMS:
        assert models.n_params(*models.build_ff(arm)) == 261500
    for arm in models.K_ARMS:
        assert models.n_params(*models.build_ff(arm)) == 261255
    expected = {32: (26496, 26831), 64: (55040, 55480), 128: (118272, 118016)}
    for w, (p, k) in expected.items():
        assert models.n_params(*models.build_ff("P-R", w=w)) == p
        assert models.n_params(*models.build_ff("K-LGT", w=w)) == k
    assert models.n_params(models.BPNet("P-R")) == 261510
    assert models.n_params(models.BPNet("K-LGT")) == 262125
    spiral_p = {8: 112, 16: 352, 32: 1216, 64: 4480}
    spiral_k = {8: 120, 16: 315, 32: 1275, 64: 4495}
    for w in models.E10_WIDTHS:
        assert models.n_params(*models.build_ff("P-R", w=w, in_dim=4, spiral=True)) == spiral_p[w]
        assert models.n_params(*models.build_ff("K-LGT", w=w, in_dim=4, spiral=True)) == spiral_k[w]
    assert models.bank_sizes(250) == (84, 83, 83)
    assert models.bank_sizes(164) == (55, 55, 54)
    assert models.bank_sizes(103) == (35, 34, 34)
    assert len(models.FF_ARMS) == 18


def test_9_1_2_every_linear_followed_by_nonlinearity():
    for arm in models.FF_ARMS:
        for layer in models.build_ff(arm):
            assert len(layer.acts) == len(layer.linears)
            assert all(a in "RLGTB" for a in layer.acts)


def test_9_1_3_negative_label_differs():
    y = torch.randint(0, 10, (100000,))
    yneg = train._neg_labels(y, 10, streams.neg_gen(1, 1, 0))
    assert bool((yneg != y).all())
    assert set(yneg.unique().tolist()) == set(range(10))


def test_9_1_4_layer2_training_does_not_change_layer1(small, monkeypatch):
    snapshots = []
    original = train._train_ff_layer

    def spy(layer, layer_no, *args, **kwargs):
        out = original(layer, layer_no, *args, **kwargs)
        snapshots.append((layer_no, {k: v.clone() for k, v in layer.state_dict().items()}))
        return out

    monkeypatch.setattr(train, "_train_ff_layer", spy)
    l1, l2, _ = train.train_ff("K-LGT", data.ImageTask, *small, CFG, 40001, emax=2)
    after_l1 = dict(snapshots)[1]
    for k, v in l1.state_dict().items():
        assert torch.equal(v, after_l1[k])


def test_9_1_5_no_test_access_in_training_mode():
    assert not data.test_access_enabled()
    for part in (("mnist", "test"), ("fmnist", "test"), ("qmnist", "test50k")):
        with pytest.raises(data.TestAccessError):
            data.load(*part)
    from ffexp import minipilot, plans, runner, spiral

    for mod in (runner, plans, minipilot, train, spiral):
        assert "enable_test_access" not in inspect.getsource(mod), mod.__name__


def test_9_1_6_bitwise_reproducibility(small):
    a = train.train_ff("K-LGT", data.ImageTask, *small, CFG, 40001, emax=2)
    b = train.train_ff("K-LGT", data.ImageTask, *small, CFG, 40001, emax=2)
    for la, lb in zip(a[:2], b[:2]):
        for (ka, va), (kb, vb) in zip(la.state_dict().items(), lb.state_dict().items()):
            assert torch.equal(va, vb), ka


def test_9_1_7_streams_identical_across_arms(small, monkeypatch):
    logs = {}

    def logging_gen(fn, tag):
        def wrapped(seed, layer, epoch):
            g = fn(seed, layer, epoch)
            logs.setdefault(tag, {}).setdefault((layer, epoch), []).append(
                torch.randint(0, 2**31, (8,), generator=fn(seed, layer, epoch)))
            return g
        return wrapped

    monkeypatch.setattr(streams, "order_gen", logging_gen(streams.order_gen, "order"))
    monkeypatch.setattr(streams, "neg_gen", logging_gen(streams.neg_gen, "neg"))
    # different arms stop layer 1 at different epochs; layer-2 streams must not depend on it
    train.train_ff("P-R", data.ImageTask, *small, {**CFG, "lr": 1e-2}, 40001, emax=3)
    train.train_ff("K-T", data.ImageTask, *small, {**CFG, "lr": 1e-4}, 40001, emax=4)
    for tag in ("order", "neg"):
        for key, draws in logs[tag].items():
            assert all(torch.equal(draws[0], d) for d in draws), (tag, key)
    assert data.make_split(5).train_idx.equal(data.make_split(5).train_idx)


def test_9_1_8_label_pixels(capsys):
    fractions = {}
    for ds in ("mnist", "fmnist"):
        x, _ = data.load(ds, "train")
        fractions[ds] = (x[:, :10] > 0).any(1).float().mean().item()
    print("odsetek obrazów z niezerowymi pikselami 0-9:", fractions)
    assert fractions["mnist"] == 0.0
    assert fractions["fmnist"] > 0.01  # -> decision D4: label concatenated instead of overwriting pixels
    x, y = data.load("fmnist", "train")
    e = data.ImageTask.embed(x[:100], y[:100])
    assert e.shape[1] == 794 and torch.equal(e[:, :784], x[:100])  # image content untouched
    assert torch.equal(e[:, 784:].argmax(1), y[:100])


def test_9_1_9_buffer_matches_recompute(small):
    a = train.train_ff("K-LGT", data.ImageTask, *small, CFG, 40001, emax=1, use_buffer=True)
    b = train.train_ff("K-LGT", data.ImageTask, *small, CFG, 40001, emax=1, use_buffer=False)
    xes = small[2]
    with torch.no_grad():
        ha = a[1](models.normalize(a[0](data.ImageTask.embed(xes, 3))))
        hb = b[1](models.normalize(b[0](data.ImageTask.embed(xes, 3))))
    assert (ha - hb).abs().max().item() < 1e-5
    pa = train.predict_ff(a[0], a[1], data.ImageTask, xes, a[2])
    pb = train.predict_ff(b[0], b[1], data.ImageTask, xes, b[2])
    assert torch.equal(pa, pb)


def test_9_1_10_data_hashes():
    sums = data.load_sums()
    assert len(sums) == 10
    for rel in sums:
        data.verify_file(rel)


def test_min_epochs_before_early_stop(small):
    """Decision D5a: early stopping never ends a layer before MIN_EPOCHS."""
    cfg = {"lr": 1e-6, "wd": 0.0, "c": 0.25}  # practically no learning -> patience would trigger early
    _, _, info = train.train_ff("P-R", data.ImageTask, *small, cfg, 40001, emax=12)
    assert info["layer1"]["epochs_run"] >= train.MIN_EPOCHS
    assert info["layer2"]["epochs_run"] >= train.MIN_EPOCHS
