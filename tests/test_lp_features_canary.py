"""Canary tests for the ATLAS-style lag-safe LP features (round 5b).

Each leak test has two halves: the REAL implementation must be invariant, and a deliberately LEAKY
variant must be caught by the same check (a canary that cannot fail proves nothing).
Needs torch + torch_geometric; the real-sampler test also needs pyg-lib and skips itself without it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData  # noqa: E402

from src.data.loader import SECONDS_PER_DAY  # noqa: E402
from src.features.lp_features import (  # noqa: E402
    LP_FEATURE_NAMES, N_LP_FEATURES, LPFeatureTransform, compute_lp_features, lp_transform_from_config,
)
from src.graph.builder import TXN  # noqa: E402
from src.graph.sampler import ComposeTransforms, DropAttributes, build_neighbor_loader, has_pyg_lib  # noqa: E402

D = float(SECONDS_PER_DAY)
W = 10.0  # maturity window in days
T_CUT = int(110 * D)  # a "label cut" later than every train-like seed time (snapshot-leak variant)


# --------------------------------------------------------------------------- #
# Toy data with a hand-computed answer
# --------------------------------------------------------------------------- #
def toy_arrays() -> dict:
    """4 seeds (rows 0..3), then history nodes. Times in DAYS (converted to seconds below).

    seed row : t_seed  own y  |  history (t_j, det_j, y_j)
    0        : 100     1      |  a (60, 62, 1)  b (95, 97, 1)  c (98, 105, 1)  d (50, 52, 0)
                              |  e (97, 99, 0)  f (70, 120, 1) g (99, 99.5, 0)
    1        : 50      0      |  h (20, 25, 0)  i (45, 48, 1)  j (49, 51, 1)
    2        : 30      1      |  k (29, 31, 1)
    3        : 20      1      |  (no history)
    """
    seeds_t = [100, 50, 30, 20]
    seeds_y = [1, 0, 1, 1]
    hist = [
        (0, 60, 62, 1), (0, 95, 97, 1), (0, 98, 105, 1), (0, 50, 52, 0), (0, 97, 99, 0), (0, 70, 120, 1),
        (0, 99, 99.5, 0),
        (1, 20, 25, 0), (1, 45, 48, 1), (1, 49, 51, 1),
        (2, 29, 31, 1),
    ]
    time = torch.tensor([round(t * D) for t in seeds_t] + [round(t * D) for _, t, _, _ in hist], dtype=torch.long)
    det = torch.tensor([t * D + D for t in seeds_t] + [d * D for _, _, d, _ in hist], dtype=torch.float64)
    y = torch.tensor(seeds_y + [yy for *_, yy in hist], dtype=torch.long)
    sub = torch.tensor(list(range(4)) + [r for r, *_ in hist], dtype=torch.long)
    return dict(time=time, detection_time=det, y_true=y, subgraph=sub, batch_size=4,
                maturity_window_days=W, node_id=torch.arange(time.numel()))


def to_batch(arrs: dict, x_dim: int = 5) -> HeteroData:
    n = arrs["time"].numel()
    b = HeteroData()
    st = b[TXN]
    st.x = torch.arange(n * x_dim, dtype=torch.float32).view(n, x_dim) / 10.0
    st.time = arrs["time"].clone()
    st.detection_time = arrs["detection_time"].clone()
    st.y_true = arrs["y_true"].clone()
    st.batch = arrs["subgraph"].clone()
    st.batch_size = arrs["batch_size"]
    st.n_id = arrs["node_id"].clone()
    return b


# --------------------------------------------------------------------------- #
# Larger random batch for the invariance tests
# --------------------------------------------------------------------------- #
def random_batch(bs: int = 60, seed: int = 0, p_fraud: float = 0.3) -> dict:
    g = torch.Generator().manual_seed(seed)
    t_seed = ((60.0 + 40.0 * torch.rand(bs, generator=g, dtype=torch.float64)) * D).long()
    n_hist = torch.randint(0, 13, (bs,), generator=g)
    owner = torch.repeat_interleave(torch.arange(bs), n_hist)
    h = owner.numel()
    age = (torch.rand(h, generator=g, dtype=torch.float64) * 70.0 * D).long().clamp(min=1)
    time = torch.cat([t_seed, t_seed[owner] - age])
    delay = (1.0 / 24 + (30.0 - 1.0 / 24) * torch.rand(bs + h, generator=g, dtype=torch.float64)) * D
    det = time.to(torch.float64) + delay
    y = (torch.rand(bs + h, generator=g) < p_fraud).long()
    return dict(time=time, detection_time=det, y_true=y, subgraph=torch.cat([torch.arange(bs), owner]),
                batch_size=bs, maturity_window_days=W)


def hist_masks(kw: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(unreported, reportable) masks over ALL nodes: history nodes with detection_time > / <= t_seed."""
    bs = kw["batch_size"]
    row = kw["subgraph"][bs:]
    t_seed = kw["time"][:bs][row].to(torch.float64)
    late = kw["detection_time"][bs:] > t_seed
    pad = torch.zeros(bs, dtype=torch.bool)
    return torch.cat([pad, late]), torch.cat([pad, ~late])


def output_changes(fn, kw: dict, mask: torch.Tensor, how: str = "flip", seed: int = 0, **extra) -> bool:
    """Alter y_true on ``mask`` (flip or shuffle) and report whether ``fn``'s output changed."""
    base = fn(**kw, **extra)
    y = kw["y_true"].clone()
    idx = mask.nonzero().view(-1)
    assert idx.numel() > 0, "vacuous: empty mask"
    if how == "flip":
        y[idx] = 1 - y[idx]
    else:
        g = torch.Generator().manual_seed(seed)
        y[idx] = kw["y_true"][idx[torch.randperm(idx.numel(), generator=g)]]
    assert not torch.equal(y, kw["y_true"]), "vacuous: the labels did not change"
    return not torch.equal(fn(**{**kw, "y_true": y}, **extra), base)


# --- deliberately leaky variants (negative controls) ---------------------------- #
def leaky_ignore_gate(**kw) -> torch.Tensor:
    """Reads the label of every earlier neighbour, reported or not (the classic future leak)."""
    return compute_lp_features(**{**kw, "detection_time": torch.full_like(kw["detection_time"], -1.0)})


def leaky_snapshot_y_obs(*, T_cut: int, **kw) -> torch.Tensor:
    """Uses y_obs, a snapshot at T_cut, instead of the labels known at t_seed."""
    y_obs = ((kw["y_true"] == 1) & (kw["detection_time"] <= T_cut)).long()
    return compute_lp_features(**{**kw, "y_true": y_obs, "detection_time": torch.full_like(kw["detection_time"], -1.0)})


def leaky_self_label(**kw) -> torch.Tensor:
    """Adds the seed's own label as if it were a labelled fraud neighbour."""
    out = compute_lp_features(**kw).clone()
    own = kw["y_true"][: kw["batch_size"]].to(torch.float32)
    out[:, 0] += own
    out[:, 1] += own
    return out


# --------------------------------------------------------------------------- #
# Test 1: labels of neighbours reported AFTER t_seed must not be read
# --------------------------------------------------------------------------- #
def test_1_real_impl_ignores_labels_of_neighbours_not_yet_reported():
    kw = random_batch()
    late, reportable = hist_masks(kw)
    assert int(late.sum()) > 50 and int(reportable.sum()) > 50, "vacuous batch"
    assert not output_changes(compute_lp_features, kw, late, how="flip")
    # not trivially label-blind: flipping labels that WERE reported does change the features
    assert output_changes(compute_lp_features, kw, reportable, how="flip")


def test_1_injected_leaks_are_caught():
    kw = random_batch()
    late, _ = hist_masks(kw)
    assert output_changes(leaky_ignore_gate, kw, late, how="flip"), "gate removed: leak not detected"
    assert output_changes(leaky_snapshot_y_obs, kw, late, how="flip", T_cut=T_CUT), (
        "y_obs snapshot (t_train) leaks labels that arrive after t_seed: leak not detected"
    )


# --------------------------------------------------------------------------- #
# Test 2: the seed's own label is never read
# --------------------------------------------------------------------------- #
def test_2_real_impl_ignores_the_seeds_own_label():
    kw = random_batch()
    seeds = torch.arange(kw["time"].numel()) < kw["batch_size"]
    assert not output_changes(compute_lp_features, kw, seeds, how="flip")


def test_2_injected_self_label_is_caught():
    kw = random_batch()
    seeds = torch.arange(kw["time"].numel()) < kw["batch_size"]
    assert output_changes(leaky_self_label, kw, seeds, how="flip")


def test_2_equal_time_and_same_id_neighbours_are_ignored():
    t = round(50 * D)
    base = dict(
        detection_time=torch.tensor([t + 3600.0, float(t - 4 * D), float(t - 4 * D)], dtype=torch.float64),
        y_true=torch.tensor([0, 1, 1]), subgraph=torch.tensor([0, 0, 0]), batch_size=1,
        maturity_window_days=W,
    )
    # same time as the seed -> not strictly earlier -> ignored even though its report "arrived"
    out = compute_lp_features(time=torch.tensor([t, t, t]), node_id=torch.tensor([7, 8, 9]), **base)
    assert out.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    # earlier nodes: the one carrying the SEED'S OWN id (7) is dropped, the other (8) counts
    time = torch.tensor([t, t - round(5 * D), t - round(5 * D)])
    with_ids = compute_lp_features(time=time, node_id=torch.tensor([7, 7, 8]), **base)
    assert with_ids.tolist() == [[1.0, 1.0, 1.0, 1.0]]
    without_ids = compute_lp_features(time=time, **base)  # no id guard: both count (shows the guard works)
    assert without_ids.tolist() == [[2.0, 2.0, 1.0, 1.0]]


# --------------------------------------------------------------------------- #
# Test 3: shuffling the labels of not-yet-reported neighbours changes nothing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_3_shuffle_of_unreported_neighbour_labels_is_invariant(seed):
    kw = random_batch(seed=seed)
    late, _ = hist_masks(kw)
    assert not output_changes(compute_lp_features, kw, late, how="shuffle", seed=seed)


@pytest.mark.parametrize("rule", ["maturity_window", "detection_time"])
def test_3_pipeline_level_invariance_and_labels_survive_until_dropped(rule):
    kw = random_batch(seed=5)
    late, _ = hist_masks(kw)
    lp = LPFeatureTransform(W, negative_rule=rule)

    def run(y):
        arrs = {**kw, "y_true": y, "node_id": torch.arange(kw["time"].numel())}
        return ComposeTransforms([lp, DropAttributes(("y_true",))])(to_batch(arrs))

    y2 = kw["y_true"].clone()
    idx = late.nonzero().view(-1)
    y2[idx] = kw["y_true"][idx[torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(1))]]
    assert not torch.equal(y2, kw["y_true"])
    a, b = run(kw["y_true"]), run(y2)
    assert torch.equal(a[TXN].x, b[TXN].x)
    assert "y_true" not in a[TXN]  # the transform ran first, then the labels were dropped


# --------------------------------------------------------------------------- #
# Test 4: hand-computed features on a small graph
# --------------------------------------------------------------------------- #
def test_4_hand_computed_default_rule():
    """Seed 0 (t=100): P = {a, b}; N = {d, f} (f is a fraud reported after t_seed -> it is only
    'mature', counted as a negative: label noise, not a leak); U = {c, e, g}.
    => n_labeled=4, n_fraud=2, rate=0.5, has_fraud=1.
    Seed 1 (t=50): P={i}; N={h}; U={j (reported at 51)} -> 2, 1, 0.5, 1.
    Seed 2 (t=30): k is reported at 31 > 30 -> unknown -> 0, 0, 0, 0.  Seed 3: no neighbour -> 0, 0, 0, 0."""
    out = compute_lp_features(**toy_arrays())
    expected = torch.tensor([[4, 2, 0.5, 1], [2, 1, 0.5, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_4_hand_computed_detection_time_rule():
    """Oracle-like reading: a non-reported neighbour with detection_time <= t_seed also counts as negative.
    Seed 0: P={a,b}; N={d,e,g} (f and c are reported after t_seed -> unknown) -> 5, 2, 0.4, 1."""
    out = compute_lp_features(**toy_arrays(), negative_rule="detection_time")
    expected = torch.tensor([[5, 2, 0.4, 1], [2, 1, 0.5, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_4_transform_appends_columns_and_zero_fills_history():
    arrs = toy_arrays()
    batch = to_batch(arrs, x_dim=5)
    x0 = batch[TXN].x.clone()
    out = LPFeatureTransform(W, log1p_counts=False)(batch)
    x = out[TXN].x
    assert x.shape == (x0.size(0), 5 + N_LP_FEATURES) and int(out[TXN].lp_dim) == N_LP_FEATURES
    assert torch.equal(x[:, :5], x0)  # original columns untouched
    torch.testing.assert_close(x[:4, 5:], compute_lp_features(**arrs))
    assert bool((x[4:, 5:] == 0).all())  # history rows: zeros

    logged = LPFeatureTransform(W, log1p_counts=True)(to_batch(arrs))[TXN].x[:4, 5:]
    raw = compute_lp_features(**arrs)
    torch.testing.assert_close(logged[:, :2], torch.log1p(raw[:, :2]))
    torch.testing.assert_close(logged[:, 2:], raw[:, 2:])  # rate and flag are not transformed


# --------------------------------------------------------------------------- #
# Test 5: no labelled neighbour -> zeros, never NaN
# --------------------------------------------------------------------------- #
def test_5_no_labelled_neighbour_gives_zeros_not_nan():
    out = compute_lp_features(**toy_arrays())
    for r in (2, 3):  # seed 2: only an unreported neighbour; seed 3: no neighbour
        assert out[r].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert bool(torch.isfinite(out).all())


def test_5_batch_without_any_history_and_all_unknown_batch():
    t = torch.tensor([round(40 * D), round(41 * D)])
    only_seeds = compute_lp_features(time=t, detection_time=t.double() + D, y_true=torch.tensor([1, 1]),
                                     subgraph=torch.tensor([0, 1]), batch_size=2, maturity_window_days=W)
    assert only_seeds.shape == (2, 4) and bool((only_seeds == 0).all())

    kw = random_batch()
    kw["maturity_window_days"] = 1e6  # nothing is ever mature
    kw["detection_time"] = kw["time"].double() + 1e9  # nothing is ever reported
    out = compute_lp_features(**kw)
    assert bool((out == 0).all()) and bool(torch.isfinite(out).all())


# --------------------------------------------------------------------------- #
# Transform guards, config, sampler wiring
# --------------------------------------------------------------------------- #
def test_transform_must_run_before_y_true_is_dropped_and_only_once():
    lp = LPFeatureTransform(W)
    with pytest.raises(KeyError):
        ComposeTransforms([DropAttributes(("y_true",)), lp])(to_batch(toy_arrays()))
    out = lp(to_batch(toy_arrays()))
    with pytest.raises(RuntimeError):
        lp(out)


def test_config_default_is_off_and_window_comes_from_meta():
    assert lp_transform_from_config(None) is None
    assert lp_transform_from_config({"lp_features": {"enabled": False}}) is None
    meta = {"config": {"delay": {"maturity_window_days": 30}}}
    lp = lp_transform_from_config({"lp_features": {"enabled": True}}, meta)
    assert lp is not None and lp.maturity_window_days == 30.0 and lp.n_features == 4
    assert lp.names == LP_FEATURE_NAMES
    with pytest.raises(ValueError):
        lp_transform_from_config({"lp_features": {"enabled": True}})  # no window anywhere
    with pytest.raises(ValueError):
        lp_transform_from_config({"lp_features": {"enabled": True, "negative_rule": "bogus"}}, meta)


def tiny_graph(n: int = 40, n_ent: int = 8, n_seeds: int = 15, seed: int = 0) -> HeteroData:
    """Phase-graph-like toy (node `time`, edge `timestamp`, one entity type) for the sampler tests."""
    g = torch.Generator().manual_seed(seed)
    t = (torch.arange(n) * 6 * 3600 + 86_400).long()
    delay = (3600.0 + 29.0 * D * torch.rand(n, generator=g, dtype=torch.float64))
    ent = torch.randint(0, n_ent, (n,), generator=g)
    ent[:n_ent] = torch.arange(n_ent)  # every entity id occurs
    graph = HeteroData()
    st = graph[TXN]
    st.x = torch.randn(n, 3, generator=g)
    st.time = t
    st.detection_time = t.double() + delay
    st.y_true = (torch.rand(n, generator=g) < 0.3).long()
    st.seed_mask = torch.zeros(n, dtype=torch.bool)
    st.seed_mask[-n_seeds:] = True
    first = torch.stack([t[ent == e].min() for e in range(n_ent)])
    graph["card"].node_id = torch.arange(n_ent)
    graph["card"].time = first
    graph["card"].num_nodes = n_ent
    src = torch.arange(n)
    graph[(TXN, "uses_card", "card")].edge_index = torch.stack([src, ent])
    graph[(TXN, "uses_card", "card")].timestamp = t
    graph[("card", "rev_uses_card", TXN)].edge_index = torch.stack([ent, src])
    graph[("card", "rev_uses_card", TXN)].timestamp = t
    return graph


def test_sampler_composes_lp_before_drop_and_keeps_the_old_behaviour():
    g = tiny_graph()
    lp = LPFeatureTransform(W)
    train = build_neighbor_loader(g, [5, 3], 8, batch_transforms=[lp], check_backend=False)
    assert isinstance(train.transform, ComposeTransforms)
    assert train.transform.steps[0] is lp and isinstance(train.transform.steps[1], DropAttributes)
    ev = build_neighbor_loader(g, [5, 3], 8, batch_transforms=[lp], expose_true_labels=True, check_backend=False)
    assert ev.transform is lp
    plain = build_neighbor_loader(g, [5, 3], 8, check_backend=False)
    assert isinstance(plain.transform, DropAttributes)  # unchanged default
    assert build_neighbor_loader(g, [5, 3], 8, expose_true_labels=True, check_backend=False).transform is None


@pytest.mark.skipif(not has_pyg_lib(), reason="temporal NeighborLoader needs pyg-lib")
def test_real_sampler_batches_match_an_independent_python_computation():
    """End to end: sampler -> LP transform -> DropAttributes. Checks the layout assumptions (seeds first,
    subgraph id per node, history = non-seed transaction nodes) against a pure-python reference."""
    g = tiny_graph()
    loader = build_neighbor_loader(g, [10_000, 10_000], 15, batch_transforms=[LPFeatureTransform(W, log1p_counts=False)])
    batch = next(iter(loader))
    st = batch[TXN]
    bs = int(st.batch_size)
    assert st.x.size(1) == 3 + N_LP_FEATURES and "y_true" not in st

    t, det, y = g[TXN].time, g[TXN].detection_time, g[TXN].y_true
    src, dst = g[(TXN, "uses_card", "card")].edge_index
    for r, s in enumerate(st.n_id[:bs].tolist()):
        ts = int(t[s])
        ents = {int(e) for e in dst[src == s] if int(g["card"].time[e]) <= ts - 1}
        nbrs = {int(j) for j, e in zip(src.tolist(), dst.tolist()) if e in ents and int(t[j]) <= ts - 1}
        n_lab = n_fr = 0
        for j in nbrs:
            reported = int(y[j]) == 1 and float(det[j]) <= ts
            if reported:
                n_lab += 1
                n_fr += 1
            elif ts - int(t[j]) >= W * SECONDS_PER_DAY:
                n_lab += 1
        got = st.x[r, 3:].tolist()
        want = [float(n_lab), float(n_fr), n_fr / max(n_lab, 1), float(n_fr > 0)]
        assert got == pytest.approx(want), f"seed {s}: {got} != {want}"