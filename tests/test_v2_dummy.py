import torch
import yaml
from src.losses.tw_nndf import HybridTWnnDFLoss
from src.graph.builder import PNU_CODES
from scripts.run_benchmark import build_plan

PNU_P, PNU_U, PNU_N = PNU_CODES["P"], PNU_CODES["U"], PNU_CODES["N"]


def make_batch(n_p=2, n_n=2, n_u=6, u_logit=0.0):
    """Tạo batch giả với số lượng P, N, U tùy chỉnh."""
    n = n_p + n_n + n_u
    pnu = torch.cat([
        torch.full((n_p,), PNU_P),
        torch.full((n_n,), PNU_N),
        torch.full((n_u,), PNU_U),
    ]).long()
    y = (pnu == PNU_P).long()
    age = torch.rand(n) * 10
    view = torch.ones(n, dtype=torch.bool)

    logits = torch.randn(n)
    logits[pnu == PNU_U] = u_logit
    return logits, y, pnu, age, view


def call(loss_fn, batch):
    logits, y, pnu, age, view = batch
    return loss_fn(logits, y, pnu, age_days=age, in_train_view=view)


print("Bắt đầu chạy V2 Dummy Smoke Test...")

# 1. Defaults must preserve V1.5 behavior.
batch = make_batch()
a = call(HybridTWnnDFLoss(0.05, lambda_pu=0.3), batch)
b = call(HybridTWnnDFLoss(0.05, lambda_pu=0.3, bce_pos_weight=1.0, u_conf_filter=False), batch)
assert torch.equal(a, b)
print("  [PASS] 1. Default parameters preserve V1.5 behavior.")

# 2. pos_weight must match BCEWithLogitsLoss on the BCE population.
logits, y, pnu, age, view = make_batch()
loss_fn = HybridTWnnDFLoss(0.05, lambda_pu=0.0, bce_pos_weight=3.0)
actual = call(loss_fn, (logits, y, pnu, age, view))
keep = pnu != PNU_U
expected = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))(logits[keep], y[keep].float())
assert torch.allclose(actual, expected, atol=1e-6)
print("  [PASS] 2. pos_weight matches BCEWithLogitsLoss.")

# 3. All U in the grey zone: PU contribution is zero, but BCE still has a valid gradient.
batch = make_batch(u_logit=0.0)
logits, y, pnu, age, view = batch
logits = logits.clone().requires_grad_(True)
loss_fn = HybridTWnnDFLoss(0.05, lambda_pu=0.5, u_conf_filter=True)
out = call(loss_fn, (logits, y, pnu, age, view))
assert torch.isfinite(out)
out.backward()
assert torch.isfinite(logits.grad).all()
assert torch.all(logits.grad[pnu == PNU_U] == 0)
assert float(loss_fn.last_risks["u_kept_frac"]) == 0.0
assert float(loss_fn.last_risks["pu_loss"]) == 0.0
print("  [PASS] 3. Grey-zone U rows produce zero PU contribution.")

# 4. Only high-confidence U rows should be retained.
batch = make_batch(u_logit=0.0)
logits, y, pnu, age, view = batch
logits = logits.clone()
u_idx = (pnu == PNU_U).nonzero().squeeze(1)
logits[u_idx[:2]] = 3.0
logits[u_idx[2:4]] = -3.0
loss_fn = HybridTWnnDFLoss(0.05, lambda_pu=0.5, u_conf_filter=True)
_ = call(loss_fn, (logits, y, pnu, age, view))
assert abs(float(loss_fn.last_risks["u_kept_frac"]) - 4 / 6) < 1e-6
print("  [PASS] 4. Only confident U rows are retained.")

# 5. Benchmark config must expand to 15 unique runs (6 baselines + 9 V2 variants).
with open("configs/benchmark_v2_probe.yaml", encoding="utf-8") as f:
    bench = yaml.safe_load(f)["benchmark"]
plan = build_plan(bench)
assert len(plan) == 15
assert len({s.run_id for s in plan}) == 15

baseline = [s for s in plan if s.variant is None]
variants = [s for s in plan if s.variant is not None]

assert len(baseline) == 6 and len(variants) == 9
assert {s.loss for s in baseline} == {"naive_bce", "maturity_mask"}
assert {s.variant for s in variants} == {"v2_light", "v2_imbalance", "v2_conf"}
assert {s.seed for s in plan} == {42, 43, 44}
assert {s.delay for s in plan} == {"delay_high"}
assert {s.backbone for s in plan} == {"sage"}
assert {s.use_temporal for s in plan} == {False}
print("  [PASS] 5. Config expands to 15 unique runs (6 baselines + 9 variants).")

print("\\n=> V2 DUMMY SMOKE TEST: PASS TOÀN BỘ!")