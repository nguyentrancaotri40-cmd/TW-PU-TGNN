"""Throwaway diagnostic: causal sampler + value-based feature-leak scan."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.dataset import load_phase_graph
from src.graph.builder import TXN
from src.graph.sampler import build_neighbor_loader, has_pyg_lib, verify_batch_causality

ROOT = Path("data/processed/delay_high")
assert has_pyg_lib(), "pyg-lib missing: temporal sampling impossible (do NOT fall back)"
tr, va = load_phase_graph(ROOT / "train.pt"), load_phase_graph(ROOT / "val.pt")

# 1) feature leakage by VALUE: correlation of every feature with y_true on train
x, y = tr[TXN].x.numpy(), tr[TXN].y_true.numpy().astype(np.float32)
corr = np.array([np.corrcoef(x[:, j], y)[0, 1] if x[:, j].std() > 0 else 0.0
                 for j in range(x.shape[1])])
top = np.argsort(-np.abs(corr))[:8]
print("top |corr|:", [(tr.feature_names[i], round(float(corr[i]), 3)) for i in top])
assert np.abs(corr).max() < 0.9, "a feature is almost a copy of the label"
bad = [n for n in tr.feature_names
       if n.lower().startswith(("detection", "y_", "pnu", "in_train", "delay"))]
assert not bad, bad

# 2) "delete the future": same train seeds, same history in train.pt and val.pt
rng = np.random.default_rng(0)
cand = torch.nonzero(tr[TXN].seed_mask).view(-1)[-5000:]        # late-train seeds: longest histories
seeds = cand[torch.from_numpy(rng.choice(len(cand), 50, replace=False))].sort().values


def history(data):
    m = torch.zeros(data[TXN].num_nodes, dtype=torch.bool)
    m[seeds] = True
    loader = build_neighbor_loader(data, [15, 10], batch_size=len(seeds), seed_mask=m,
                                   temporal_strategy="last")
    b = next(iter(loader))
    verify_batch_causality(b)
    st, bs = b[TXN], int(b[TXN].batch_size)
    return st.tid[:bs], [sorted(st.tid[st.batch == i].tolist()) for i in range(bs)]


s_tr, h_tr = history(tr)
s_va, h_va = history(va)
assert torch.equal(s_tr, s_va)
assert h_tr == h_va, "history of a train seed changed when LATER rows were added -> future leaks"
print("OK: causal sampler, delete-the-future and value-based leak scan passed")