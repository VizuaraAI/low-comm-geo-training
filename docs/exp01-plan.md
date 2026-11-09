# Exp01 plan, topic 09: does the inner optimizer change which rule wins?

Written on 4 November 2026 by the Vizuara research team. Nothing here has been run and no GPU has been used. All speeds below are assumptions until the smoke test replaces them.

**Question.** On eight simulated islands with GPU speeds measured on Modal, does the ranking of three aggregation rules differ between AdamW and Muon?

**Why this size.** [1] (arXiv 2610.19064) argues that whitened updates make simple averaging close to optimal, so inverse-noise weighting should help AdamW more than Muon. We can test the direction of that effect on one GPU for under $15, before spending on the main study. A small model cannot settle the question at scale, and it can show whether the interaction is worth chasing.

## Design

**Simulation.** One process on one L40S holds eight copies of the model and eight inner optimizer states, and it steps them in turn. There is no DDP and no real network. Time is a logical clock: a worker's step costs 1/v_i seconds and a round costs max_i(H_i / v_i) + c. Inner optimizer state is never synchronized, as in [1]. Compression, streaming, overlap and asynchrony are all left out.

**Fleet.** Two workers each of L4, A10, L40S and H100. The speeds v_i come from a short profile of this exact model on each GPU type on Modal. Our guess is about 5x to 7x between H100 and L4; the profile replaces it.

**Model and data.** A GPT with 6 layers, 6 heads, 384 dimensions and context 256. With the GPT-2 tokenizer this is about 30M parameters, of which about 19M are the tied embedding. FineWeb-Edu, first shards only, about 150M tokens for training and a fixed held-out split (revision pinned and licence confirmed on the dataset card at download). Each worker reads its own i.i.d. shard. Per-worker batch is 32 sequences. The budget is 120M training tokens per run, so loss will still be falling at the end. That is a stated limit: we are after a ranking.

**Inner optimizers.** AdamW, and Muon on hidden 2-D matrices with AdamW on embeddings, norms and head, as in MuLoCo. Muon uses `torch.optim.Muon` from one pinned PyTorch release if it matches this split, otherwise about 40 lines of our own following Keller Jordan's writeup. No other new dependency. Every package is pinned by hash in the image, and nothing is installed at run time.

**Rules.** Each worker returns Delta_i = theta_t - theta_i. The aggregate goes to one Nesterov outer step (eta, beta). H = 30 for the fastest worker, the default in [1] and MuLoCo.

| Rule | Schedule | Weights | Source |
|---|---|---|---|
| R1, equal steps | H_i = 30 for all; fast workers wait | Uniform | DiLoCo |
| R2, equal time, token weights | H_i = floor(30 v_i / v_max), at least 1 | w_i = H_i / sum H_j, applied to Delta_i | DyLU in 2401.09135; token weights in 2610.03457 |
| R3, equal time, inverse noise | As R2 | Per-step rate d_i = Delta_i / H_i. u_i proportional to 1 / s_i^2, clipped to within a factor of 4 of uniform and normalized. Aggregate = H_eff sum u_i d_i, with H_eff = sum H_i^2 / sum H_i | Ours, from the Aitken argument in [1] and the normalization in FedNova |

s_i^2 is an exponential moving average (decay 0.9) of the squared distance between d_i and the previous round's aggregate rate. It costs one scalar per worker per round. R3 is our own proposal, and we found no earlier DiLoCo version.

**Baselines and control.** A data-parallel run for each optimizer with the same global batch and tokens, as [1] compares each optimizer with its own data-parallel run. Control: with equal speeds, R2 must equal R1 exactly, and R3's weights must stay close to uniform. R1's loss curve does not depend on speeds, so one set of R1 runs can be re-priced for any fleet.

**Runs.**

| Stage | Runs | Tokens each | L40S time (assumed 300k tokens per second, plus 10% for evaluation) |
|---|---|---|---|
| Smoke test, equal-speed control, profile | 8 short | 1M | 0.25 h, plus 5 minutes on each of four GPU types |
| Tune inner learning rate on data-parallel | 2 x 4 = 8 | 30M | 0.25 h |
| Tune outer (eta, beta) in {(0.7, 0.9), (1.0, 0.0), (0.4, 0.8)} per cell | 2 x 3 x 3 = 18 | 30M | 0.5 h |
| Main: 2 x 3 x 3 seeds, plus 2 data-parallel x 3 seeds | 24 | 120M | 2.9 h |
| Reserve for a failed or repeated run | | | 0.3 h |

The 300k tokens per second is our discount on nanoGPT's reported 5,000 iterations in about 3 minutes on one A100 at batch 64, which is about 455k tokens per second. It has not been measured on an L40S at batch 32. Tuning at a quarter of the budget may favour fast starters, and we report that limit.

**Expected runtime.** About 4.2 L40S hours in total. That is about 5 hours of wall clock on one container with start-ups, or about 3 hours on two containers at the same cost.

**Logged, per round and per worker.** H_i, weights, ||Delta_i||, ||Delta_i|| / H_i, and the mean pairwise cosine of the Delta_i. Validation loss every 5M tokens with the simulated time and tokens at that point. Seed, configuration and code hash for every run, in JSONL. Because the clock is logical, cost c can be re-priced afterwards without any new run: we report c at 0, 0.1 and 1 times the fastest worker's round compute. The cross-facility paper's 32% overhead at H = 100 falls inside this range.

**Metrics.** Primary: simulated wall-clock time to a common target loss, set from the AdamW data-parallel curve before the DiLoCo runs are read. Secondary: validation loss at 120M tokens; time to each optimizer's own target; and dollars to target, computed as the sum of list prices over all eight workers times wall clock, since every rented GPU is paid for while it waits.

## Predictions and gates

Written down before the run. To be fixed in `decision_rule.json` before any result is opened.

- **P1.** R2 and R3 reach the target sooner than R1 in wall clock for both optimizers, and they lose some loss per token.
- **P2 (the main one).** Let g(opt) = log(time R2 / time R3). Then g(AdamW) > g(Muon).
- **P3.** ||Delta_i|| / H_i varies less with H_i under Muon than under AdamW.
- **G0, sound measurement.** Muon and AdamW data-parallel runs train stably, and the equal-speed control passes. Failing this stops the experiment, and any cell that fails is reported as unmeasured.
- **G1, speed matters.** R2 or R3 is at least 20% faster than R1 for each optimizer. If this fails, the fleet is not hard enough for the question, and we add a severe profile within the cap before drawing any conclusion.
- **G2, interaction.** I = g(AdamW) - g(Muon). Three seeds can only show direction, so the rule is: signal if I has the same sign in all three paired seeds and |I| is more than twice the seed standard deviation of g. If there is no signal, add two seeds on R2 and R3 only (about $2, within the cap).
- **Reading the outcome.** Signal: go to the main study with a whitening arm. No signal after five seeds: the claim becomes "the rule and the optimizer can be chosen separately at this scale", and we test it again at 124M before believing it. R3 worse than R2 under both: drop R3 and compare schedules and existing weights, which is still an open comparison.

## Cost

Modal list prices, read on 4 November 2026 from modal.com/pricing.

| Item | Amount | Cost |
|---|---|---|
| L40S, $0.000542 per second ($1.95 per hour) | 4.2 h planned | $8.19 |
| 4 CPU cores with the GPU, $0.0000131 per core-second | same | $0.79 |
| 16 GiB memory with the GPU, $0.00000222 per GiB-second | same | $0.54 |
| Speed profile: L4, A10, L40S, H100, 5 minutes each, with CPU and memory | 20 minutes | $0.76 |
| CPU container for download and tokenizing (2 cores, 8 GiB) | about 2 h | $0.32 |
| **Planned total** | | **about $10.60** |
| **Hard stop** at 5.5 L40S hours | | **about $13.55** |

Every GPU function has a timeout. A guard script sums GPU seconds in `costlog.jsonl` and refuses to start a function once 5.5 hours are used, and we check the Modal billing summary at the end of each day. No paid API is used. Credentials stay out of the code and the logs.

## What this leaves out, for the ablations and the main study

- R4, equal steps with speed-proportional batches: what Core Automation and the cross-facility paper do, and the first addition we would make.
- SPlus or SOAP as the third inner optimizer.
- Plain averaging against Nesterov, because SNOO warns that momentum alone explains much of DiLoCo's benefit.
- Non-IID shards, with one language or domain per worker, as in HeLoCo. This is the setting of ablation A.
- 4-bit pseudo-gradients, and real measured links replacing c.
- One downstream benchmark, given the nanochat warning in arXiv 2511.13761.
