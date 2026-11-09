"""Simulated compute islands: eight workers, one GPU, and a logical clock.

One process holds eight copies of a small GPT and eight inner optimizer states and steps them in turn. There is no DDP
and no real network. A worker's step costs 1/v_i simulated seconds, with v_i measured on four GPU types, and a round
costs max_i(H_i / v_i) + c. Each worker returns Delta_i = theta_t - theta_i, a rule combines the Delta_i, and one
Nesterov outer step applies the result. Inner optimizer state is never synchronized.

This file holds the same model, optimizers, rules and run loop as our Exp01 and Exp02, with the cloud launcher removed
and the device made a parameter. See docs/exp01-results.md and docs/exp02-results.md for what those runs found.

    python islands.py selftest
    python islands.py bridge --opt adamw --data data/fineweb
    python islands.py run --opt adamw --rule R2 --seed 1 --tokens 60000000 --data data/fineweb --out results/mine

The rules. H_i is the number of local steps and c_i the weight on Delta_i.

    R1   equal steps (H_i = 30), token weights, which are uniform here
    R2   equal time, H_i = floor(30 v_i / v_max) and at least 1, token weights c_i = H_i / sum H_j
    R3   equal time, inverse-noise weights u_i on per-step rates, scaled by H_eff = sum H_j^2 / sum H_j
    R3a  R3 with u_i fixed at 1/n: the normalization alone
    R3b  equal time, c_i proportional to H_i u_i: the inverse-noise weights alone, with no normalization
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODEL = dict(n_layer=6, n_head=6, n_embd=384, block=256, vocab=50257)
N_WORKERS, BATCH = 8, 32
H_FAST = 30
FLEET = ["L4", "L4", "A10", "A10", "L40S", "L40S", "H100", "H100"]
PRICES = {"L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "H100": 0.001097}  # US dollars per second, Modal list, 4 Nov 2026

EVAL_EVERY = 2_500_000
WARMUP_TOKENS = 4_000_000
WEIGHT_DECAY, BETAS, CLIP = 0.01, (0.9, 0.95), 1.0
MUON_MOMENTUM = 0.95
OUTERS = ((0.7, 0.9), (1.0, 0.0), (0.4, 0.8))
OPTS = ("adamw", "muon")


def tok_per_step(batch=None):
    return (batch or BATCH) * MODEL["block"]


# ------------------------------------------------------------------ schedules and weights

def schedule(rule, v):
    """Local steps per worker. R1 is equal steps; the others are equal time, floor(30 v_i / v_max), at least 1."""
    if rule == "R1":
        return [H_FAST] * len(v)
    vmax = max(v)
    return [max(1, int(math.floor(H_FAST * x / vmax + 1e-9))) for x in v]


def clip_uniform(u, n, factor=4.0, iters=50):
    """Project weights so that each lies within a factor of 4 of 1/n and they sum to 1."""
    lo, hi = 1.0 / (factor * n), factor / n
    u = list(u)
    for _ in range(iters):
        c = [min(hi, max(lo, x)) for x in u]
        s = sum(c)
        u = [x / s for x in c]
        if all(lo - 1e-12 <= x <= hi + 1e-12 for x in u):
            break
    return u


# ------------------------------------------------------------------ model and inner optimizers

def build_model(torch, seed):
    nn, F = torch.nn, torch.nn.functional
    c = MODEL

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(c["n_embd"], 3 * c["n_embd"], bias=False)
            self.proj = nn.Linear(c["n_embd"], c["n_embd"], bias=False)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            shape = (B, T, c["n_head"], C // c["n_head"])
            q, k, v = [t.view(shape).transpose(1, 2) for t in (q, k, v)]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).reshape(B, T, C))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(c["n_embd"], bias=False)
            self.attn = Attn()
            self.ln2 = nn.LayerNorm(c["n_embd"], bias=False)
            self.fc = nn.Linear(c["n_embd"], 4 * c["n_embd"], bias=False)
            self.fc_out = nn.Linear(4 * c["n_embd"], c["n_embd"], bias=False)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(c["vocab"], c["n_embd"])
            self.pos = nn.Embedding(c["block"], c["n_embd"])
            self.blocks = nn.ModuleList(Block() for _ in range(c["n_layer"]))
            self.ln = nn.LayerNorm(c["n_embd"], bias=False)
            self.register_buffer("idx", torch.arange(c["block"]), persistent=False)

        def forward(self, x, y):
            h = self.tok(x) + self.pos(self.idx[: x.shape[1]])
            for b in self.blocks:
                h = b(h)
            logits = self.ln(h) @ self.tok.weight.T      # tied head
            # The loss is computed in float32: bf16 has a spacing of 0.03 near 5, too coarse to compare runs.
            with torch.amp.autocast(x.device.type, enabled=False):
                return F.cross_entropy(logits.float().reshape(-1, c["vocab"]), y.reshape(-1))

    m = GPT()
    g = torch.Generator().manual_seed(seed)
    for n, p in m.named_parameters():
        if p.ndim == 2:
            std = 0.02 / math.sqrt(2 * c["n_layer"]) if n.endswith((".proj.weight", ".fc_out.weight")) else 0.02
            torch.nn.init.normal_(p, 0.0, std, generator=g)
    return m


def make_inner(torch, model, opt, lr, aux_lr, fused=True):
    """AdamW on everything, or Muon on hidden matrices with AdamW on embeddings and norms (the head is tied)."""
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
    emb = [p for n, p in model.named_parameters() if not n.startswith("blocks.") and p.ndim == 2]
    norms = [p for p in model.parameters() if p.ndim != 2]
    if opt == "adamw":
        groups = [dict(params=hidden + emb, weight_decay=WEIGHT_DECAY), dict(params=norms, weight_decay=0.0)]
        o = torch.optim.AdamW(groups, lr=lr, betas=BETAS, fused=fused)
        return [o], [lr]
    m = torch.optim.Muon(hidden, lr=lr, weight_decay=WEIGHT_DECAY, momentum=MUON_MOMENTUM, nesterov=True)
    groups = [dict(params=emb, weight_decay=WEIGHT_DECAY), dict(params=norms, weight_decay=0.0)]
    a = torch.optim.AdamW(groups, lr=aux_lr, betas=BETAS, fused=fused)
    return [m, a], [lr, aux_lr]


# ------------------------------------------------------------------ data

class Data:
    """Worker i draws random windows from shard i of one token array. Validation is one fixed token array."""

    def __init__(self, torch, device, train, val):
        self.torch, self.device = torch, device
        self.train = torch.from_numpy(train).to(device)
        self.val = torch.from_numpy(val).to(device)
        self.shard = len(train) // N_WORKERS
        self.off = torch.arange(MODEL["block"] + 1, device=device)

    @classmethod
    def from_dir(cls, torch, device, folder):
        import numpy as np
        train = np.fromfile(f"{folder}/train.u16.bin", dtype=np.uint16).astype(np.int32)
        val = np.fromfile(f"{folder}/val.u16.bin", dtype=np.uint16).astype(np.int32)
        return cls(torch, device, train, val)

    def gens(self, seed):
        t = self.torch
        return [t.Generator(device=self.device).manual_seed(seed * 1000 + i) for i in range(N_WORKERS)]

    def batch(self, worker, gen, n=None):
        t = self.torch
        n = n or BATCH
        start = t.randint(0, self.shard - MODEL["block"] - 1, (n,), generator=gen, device=self.device)
        seq = self.train[(start + worker * self.shard)[:, None] + self.off].long()
        return seq[:, :-1], seq[:, 1:]

    def val_loss(self, torch, model, bs=128):
        return loss_on(torch, model, self.val, self.off, bs)


def loss_on(torch, model, tokens, off, bs=128):
    """Mean loss over consecutive windows of a fixed token array. Use it once per language for per-language validation."""
    model.eval()
    T = MODEL["block"]
    n = (len(tokens) - 1) // T
    total, count = 0.0, 0
    device = tokens.device
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        for i in range(0, n, bs):
            k = min(bs, n - i)
            idx = torch.arange(i, i + k, device=device)[:, None] * T + off
            seq = tokens[idx].long()
            total += float(model(seq[:, :-1], seq[:, 1:])) * k
            count += k
    model.train()
    return total / count


# ------------------------------------------------------------------ helpers

def setup_torch():
    import torch
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")
    return torch


def gpu_info(torch, device):
    if device != "cuda":
        return dict(torch=torch.__version__, cuda_name=device, driver="")
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return dict(torch=torch.__version__, cuda_name=torch.cuda.get_device_name(0), driver=drv)


def warm(tokens):
    return min(1.0, max(0.1, tokens / WARMUP_TOKENS))


def set_lr(opts, base, mult):
    for o, b in zip(opts, base):
        for g in o.param_groups:
            g["lr"] = b * mult


def step_once(torch, model, opts, x, y):
    with torch.amp.autocast(x.device.type, dtype=torch.bfloat16):
        loss = model(x, y)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    for o in opts:
        o.step()
    return loss.detach()


def theta_sha(theta):
    return hashlib.sha256(theta.detach().cpu().numpy().tobytes()).hexdigest()


class RunLog:
    def __init__(self, key, folder, header):
        os.makedirs(folder, exist_ok=True)
        self.path = f"{folder}/{key}.jsonl"
        self.f = open(self.path, "w")
        self.write(dict(t="header", key=key, **header))

    def write(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self, end):
        self.write(dict(t="end", **end))
        self.f.close()


# ------------------------------------------------------------------ the rules

class RuleState:
    def __init__(self):
        self.prev_rate, self.s2 = None, None


def rule_step(torch, rule, D, Ht, st):
    """One round's aggregate and its log fields. The noise estimator runs under every rule. Only R3 and R3b let it set
    weights; under R1, R2 and R3a it is a shadow against the uniform mean rate, so its values can be read there too."""
    n = D.shape[0]
    rec = {}
    d = D / Ht[:, None]
    if st.prev_rate is None:
        u = [1.0 / n] * n
    else:
        dist2 = ((d - st.prev_rate) ** 2).sum(1).double()
        st.s2 = dist2 if st.s2 is None else 0.9 * st.s2 + 0.1 * dist2
        inv = (1.0 / st.s2.clamp(min=1e-30)).tolist()
        u = clip_uniform([x / sum(inv) for x in inv], n)
        rec["s2"] = st.s2.tolist()
    ref = u if rule in ("R3", "R3b") else [1.0 / n] * n
    st.prev_rate = torch.tensor(ref, dtype=torch.float32, device=D.device) @ d
    if rule in ("R1", "R2"):
        w = Ht / Ht.sum()
        agg = w @ D
        rec["w"] = w.tolist()
        rec["u_shadow"] = u
    elif rule == "R3":
        h_eff = float((Ht ** 2).sum() / Ht.sum())
        agg = h_eff * st.prev_rate
        rec["u"], rec["h_eff"] = u, h_eff
    elif rule == "R3a":
        h_eff = float((Ht ** 2).sum() / Ht.sum())
        c = [h_eff / (n * h) for h in Ht.tolist()]
        agg = torch.tensor(c, dtype=torch.float32, device=D.device) @ D
        rec["c"], rec["h_eff"], rec["u_shadow"] = c, h_eff, u
    elif rule == "R3b":
        q = [h * x for h, x in zip(Ht.tolist(), u)]
        total = sum(q)
        c = [x / total for x in q]
        agg = torch.tensor(c, dtype=torch.float32, device=D.device) @ D
        rec["c"], rec["u"] = c, u
    else:
        raise ValueError(rule)
    return agg, rec


def check_round(G, H, rule, rec, round_no):
    """The control our first smoke attempt lacked. G is D D^T. A failed assert stops the run."""
    n = len(H)
    diag = [float(G[i][i]) for i in range(n)]
    assert min(diag) > 0, "zero Delta: a worker returned Delta_i = 0"
    for i in range(n):
        for j in range(i + 1, n):
            assert diag[i] + diag[j] - 2 * float(G[i][j]) > 0, f"identical Delta: workers {i} and {j}"
    if rule == "R3b":
        token = [h / sum(H) for h in H]
        assert abs(sum(rec["c"]) - 1) < 1e-9, "R3b weights must sum to 1"
        if "s2" in rec and round_no >= 3 and max(rec["s2"]) / min(rec["s2"]) > 1.001:
            assert max(abs(n * x - 1) for x in rec["u"]) > 1e-6, "uniform weights: s2 differ but u did not move"
            assert max(abs(c / t - 1) for c, t in zip(rec["c"], token)) > 1e-6, "token weights: s2 differ but c did not move"


# ------------------------------------------------------------------ one simulated run

def run_islands(torch, data, key, opt, rule, H, lr, aux_lr, eta, beta, seed, budget, eval_every, out, info,
                speeds=None, device="cuda", inject_view_bug=False, val_fn=None, max_rounds=None):
    """Eight workers stepped in turn. val_fn(model) may replace the validation measure; it returns a float or a dict whose
    "val_loss" field is the headline number, so per-language losses go into the log next to it."""
    n = N_WORKERS
    log = RunLog(key, out, dict(kind="islands", opt=opt, rule=rule, H=H, lr=lr, aux_lr=aux_lr, eta=eta, beta=beta, seed=seed,
                                budget=budget, eval_every=eval_every, speeds=speeds, fleet=FLEET, model=MODEL,
                                batch_per_worker=BATCH, n_workers=n, betas=BETAS, weight_decay=WEIGHT_DECAY, clip=CLIP,
                                warmup_tokens=WARMUP_TOKENS, **info))
    p2v = torch.nn.utils.parameters_to_vector

    def v2p(vec, params):
        if inject_view_bug:                  # our first smoke bug, kept for the negative control only
            torch.nn.utils.vector_to_parameters(vec, params)
            return
        # Copy. vector_to_parameters makes the parameters views of vec, which made every Delta_i zero.
        with torch.no_grad():
            i = 0
            for q in params:
                q.copy_(vec[i:i + q.numel()].view_as(q))
                i += q.numel()

    def measure(model):
        r = val_fn(model) if val_fn else data.val_loss(torch, model)
        return r if isinstance(r, dict) else dict(val_loss=r)

    init = build_model(torch, seed).to(device)
    theta = p2v(init.parameters()).detach().clone()
    N = theta.numel()
    eval_model = init
    models, inner = [], []
    fused = device == "cuda"
    for _ in range(n):
        m = build_model(torch, seed).to(device)
        models.append(m)
        inner.append(make_inner(torch, m, opt, lr, aux_lr, fused=fused))
    gens = data.gens(seed)
    Ht = torch.tensor(H, dtype=torch.float32, device=device)
    D = torch.zeros((n, N), dtype=torch.float32, device=device)
    mom = None
    st = RuleState()
    tokens, rnd, next_eval, status = 0, 0, eval_every, "ok"
    per_round = sum(H) * tok_per_step()
    round_c0 = max(h / v for h, v in zip(H, speeds)) if speeds else None
    t0 = time.time()
    train_s = 0.0
    log.write(dict(t="val", tokens=0, round=0, sim_c0=0.0, wall=0.0, **measure(eval_model)))
    while tokens < budget and (max_rounds is None or rnd < max_rounds):
        ts = time.time()
        mult = warm(tokens)
        losses = []
        for i in range(n):
            v2p(theta, models[i].parameters())
            opts, base = inner[i]
            set_lr(opts, base, mult)
            for _ in range(H[i]):
                x, y = data.batch(i, gens[i])
                loss = step_once(torch, models[i], opts, x, y)
            losses.append(float(loss))
            D[i] = theta - p2v(models[i].parameters()).detach()
        train_s += time.time() - ts
        G = (D @ D.T).double()
        norms = G.diag().clamp(min=0).sqrt()
        off = (G / (norms[:, None] * norms[None, :]).clamp(min=1e-30))[~torch.eye(n, dtype=torch.bool, device=device)]
        rec = dict(t="round", round=rnd + 1, H=H, tokens=tokens + per_round, norm=norms.tolist(),
                   rate=(norms / Ht.double()).tolist(), cos=float(off.mean()), train_loss=losses, mult=mult,
                   G=G.tolist())
        if not (math.isfinite(sum(losses)) and bool(torch.isfinite(G).all())):
            status = "diverged"
            log.write(rec | dict(t="diverged"))
            tokens += per_round
            rnd += 1
            break
        agg, extra = rule_step(torch, rule, D, Ht, st)
        rec.update(extra)
        check_round(rec["G"], H, rule, rec, rnd + 1)
        if beta > 0:
            mom = agg.clone() if mom is None else beta * mom + agg
            theta = theta - eta * (agg + beta * mom)
        else:
            theta = theta - eta * agg
        tokens += per_round
        rnd += 1
        rec["sim_c0"] = rnd * round_c0 if round_c0 else None
        log.write(rec)
        if tokens >= next_eval or tokens >= budget:
            v2p(theta, eval_model.parameters())
            m = measure(eval_model)
            log.write(dict(t="val", tokens=tokens, round=rnd, sim_c0=rec["sim_c0"],
                           wall=round(time.time() - t0, 1), **m))
            if not math.isfinite(m["val_loss"]):
                status = "diverged"
                break
            while next_eval <= tokens:
                next_eval += eval_every
    end = dict(status=status, tokens=tokens, rounds=rnd, wall=round(time.time() - t0, 1), train_seconds=round(train_s, 1),
               tokens_per_s=round(tokens / max(train_s, 1e-9)), theta_sha256=theta_sha(theta))
    log.close(end)
    print(f"{key} {status} tokens={tokens} rounds={rnd} wall={end['wall']}s train_tok/s={end['tokens_per_s']}", flush=True)
    return end


# ------------------------------------------------------------------ settings from our runs

def speeds_for(opt):
    prof = {g: json.load(open(HERE / f"results/exp01/profile/{g}.json")) for g in set(FLEET)}
    return [prof[g]["opts"][opt]["steps_per_s"] for g in FLEET]


def tuned(opt, rule):
    """Inner learning rates from Exp01, outer settings from the tuning runs of Exp01 and Exp02."""
    ch = json.load(open(HERE / "results/exp01/choices/inner.json"))[opt]
    folder = "exp02" if rule in ("R3a", "R3b") else "exp01"
    o = json.load(open(HERE / f"results/{folder}/choices/outer_{opt}_{rule}.json"))
    return ch["lr"], ch["aux_lr"], o["eta"], o["beta"]


# ------------------------------------------------------------------ self tests (no GPU needed)

def selftest(torch, device="cpu"):
    """Arithmetic checks on the rules and the controls, including that each control fires on the fault it is there for."""
    assert schedule("R2", [1, 2, 3, 6]) == [5, 10, 15, 30]
    assert clip_uniform([0.25] * 4, 4) == [0.25] * 4
    n, N = 8, 20000
    H = [4, 4, 6, 6, 13, 13, 30, 30]
    Ht = torch.tensor(H, dtype=torch.float32, device=device)
    token = [h / sum(H) for h in H]
    h_eff = sum(h * h for h in H) / sum(H)
    g = torch.Generator(device=device).manual_seed(5)
    r = torch.randn(N, generator=g, device=device) / math.sqrt(N)

    def deltas(sigma):
        d = r + torch.randn(n, N, generator=g, device=device) * torch.tensor(sigma, device=device)[:, None] / math.sqrt(N)
        return d * Ht[:, None]

    same = r[None, :].repeat(n, 1) * Ht[:, None]
    a2, _ = rule_step(torch, "R2", same, Ht, RuleState())
    a3a, ra = rule_step(torch, "R3a", same, Ht, RuleState())
    a3b, rb = rule_step(torch, "R3b", same, Ht, RuleState())
    assert torch.allclose(a2, a3a, atol=1e-5) and torch.allclose(a2, a3b, atol=1e-5), "rules must agree at equal rates"
    assert all(abs(c - t) < 1e-12 for c, t in zip(rb["c"], token)), "R3b at uniform u must be token weights"
    assert abs(sum(ra["c"]) - h_eff * sum(1 / h for h in H) / n) < 1e-5, "R3a weight sum"

    D = deltas([1.0] * n)
    a3a, _ = rule_step(torch, "R3a", D, Ht, RuleState())
    r3 = h_eff * (torch.full((n,), 1 / n, device=device) @ (D / Ht[:, None]))
    assert torch.allclose(a3a, r3, atol=1e-4), "R3a must equal R3's normalization with uniform weights"

    Heq = torch.full((n,), 10.0, device=device)
    assert rule_step(torch, "R3a", D, Heq, RuleState())[1]["c"] == [0.125] * n

    st, rec = RuleState(), None
    for _ in range(8):
        _, rec = rule_step(torch, "R3b", deltas([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), Ht, st)
    assert rec["u"][0] < rec["u"][2] and rec["u"][1] < rec["u"][3], "a noisy worker must get a smaller u"
    assert max(abs(c / t - 1) for c, t in zip(rec["c"], token)) > 0.05, "R3b must move away from token weights"
    st = RuleState()
    for _ in range(8):
        dd = (r + torch.randn(n, N, generator=g, device=device) / math.sqrt(N)) * 10.0
        _, rec = rule_step(torch, "R3b", dd, Heq, st)
    assert max(abs(n * x - 1) for x in rec["u"]) < 0.25, "equal noise must leave u near uniform"

    def fires(fn):
        try:
            fn()
        except AssertionError:
            return True
        return False
    good = (D @ D.T).double().tolist()
    okrec = dict(c=[x / sum(token) for x in token], u=[1 / n] * n, s2=[1.0] * n)
    check_round(good, H, "R3b", okrec, 3)
    zeros = [[0.0] * n for _ in range(n)]
    assert fires(lambda: check_round(zeros, H, "R2", {}, 1)), "zero Delta must fire"
    twin = D.clone()
    twin[1] = twin[0]
    assert fires(lambda: check_round((twin @ twin.T).double().tolist(), H, "R2", {}, 1)), "identical workers must fire"
    stuck = dict(c=token, u=[1 / n] * n, s2=[1.0, 2.0] * 4)
    assert fires(lambda: check_round(good, H, "R3b", stuck, 3)), "uniform u with differing s2 must fire"
    stuck_c = dict(c=token, u=[0.2, 0.05] + [0.125] * 6, s2=[1.0, 2.0] * 4)
    assert fires(lambda: check_round(good, H, "R3b", stuck_c, 3)), "token c with differing s2 must fire"
    print("selftest ok", flush=True)


def bridge(torch, opt, folder, device, rounds=4):
    """Rerun R2 at seed 1 for a few rounds and compare per-worker norms with our logged run. On the same GPU type and
    PyTorch build they should be equal; elsewhere they should be close, and the output shows how close."""
    ref = [json.loads(x) for x in open(HERE / f"results/exp02/runs/main_{opt}_R2_s1.jsonl")]
    ref = [r for r in ref if r["t"] == "round"][:rounds]
    v = speeds_for(opt)
    lr, aux, eta, beta = tuned(opt, "R2")
    data = Data.from_dir(torch, device, folder)
    key = f"bridge_{opt}_R2_s1"
    out = HERE / "results/mine"
    run_islands(torch, data, key, opt, "R2", schedule("R2", v), lr, aux, eta, beta, 1, 10**12, 10**12, out,
                gpu_info(torch, device), speeds=v, device=device, max_rounds=rounds)
    mine = [json.loads(x) for x in open(out / f"{key}.jsonl")]
    mine = [r for r in mine if r["t"] == "round"]
    worst = 0.0
    for a, b in zip(mine, ref):
        worst = max(worst, max(abs(x - y) / y for x, y in zip(a["norm"], b["norm"])))
    print(f"bridge {opt}: largest relative difference in per-worker norms over {rounds} rounds: {worst:.2e}")
    print("logged GPU:", json.loads(open(HERE / f'results/exp02/runs/main_{opt}_R2_s1.jsonl').readline())["cuda_name"])
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("selftest", "bridge", "run"))
    ap.add_argument("--opt", choices=OPTS, default="adamw")
    ap.add_argument("--rule", choices=("R1", "R2", "R3", "R3a", "R3b"), default="R2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=60_000_000)
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--out", default="results/mine")
    ap.add_argument("--device", default=None)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    a = ap.parse_args()
    torch = setup_torch()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.command == "selftest":
        selftest(torch)
    elif a.command == "bridge":
        bridge(torch, a.opt, a.data, device)
    else:
        v = speeds_for(a.opt)
        lr, aux, eta, beta = tuned(a.opt, a.rule)
        eta = eta if a.eta is None else a.eta
        beta = beta if a.beta is None else a.beta
        data = Data.from_dir(torch, device, a.data)
        run_islands(torch, data, f"main_{a.opt}_{a.rule}_s{a.seed}", a.opt, a.rule, schedule(a.rule, v), lr, aux, eta, beta,
                    a.seed, a.tokens, EVAL_EVERY, a.out, gpu_info(torch, device), speeds=v, device=device)


if __name__ == "__main__":
    main()
