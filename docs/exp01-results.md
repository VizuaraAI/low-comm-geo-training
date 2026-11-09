# Exp01 islands, topic 09: lab note

Run by the Vizuara research team on 4 November 2026.

## In short

We asked whether the ranking of three aggregation rules differs between AdamW and Muon on eight simulated islands. The pre-registered interaction test (G2) did not fire, because it could not be computed: all three AdamW R3 runs missed even the fallback target within their token budget, and the rule forbids extrapolation. So this experiment gives neither a signal nor a no-signal result.

Three things did come out clearly. First, equal-time schedules are much faster than equal steps on this fleet: R2 reached the target in 22% of R1's simulated time with AdamW and 29% with Muon. Second, our own rule, R3, was slower than R2 under both optimizers, so the pre-registered "drop R3" outcome applies. For AdamW this reading uses a lower bound, as explained below. Third, R2 also had lower loss per token than R1, which is the opposite of what we predicted.

The evidence outside the verdict points against prediction P2: R3 hurt AdamW more than it hurt Muon. We do not read this as a test of the whitening mechanism. R3's weights stayed close to uniform, so what we measured is mostly the effect of R3's per-step normalization.

## What we ran

**Simulation.** One process on one L40S held eight copies of the model and eight inner optimizer states and stepped them in turn. There was no real network. A worker's step costs 1/v_i simulated seconds and a round costs max_i(H_i / v_i) + c, with c = k x 30 / v_max. We report k at 0, 0.1 and 1.

**Model and data.** A GPT with 6 layers, 6 heads, 384 dimensions and context 256, which has 30,018,816 parameters. Of these, 19,298,688 are the tied embedding and 10,616,832 are the hidden matrices that Muon updates. The data is FineWeb-Edu at revision 87f09149 (licence odc-by on the card at that revision), tokenized with GPT-2. We read the first 146,000 documents of the first parquet file, shuffled them, and took 150M training tokens in eight equal shards and 2M held-out validation tokens from separate documents.

**Inner optimizers.** AdamW, and `torch.optim.Muon` from PyTorch 2.10.0 paired with AdamW on embeddings and norms. All 41 packages are pinned by hash in `requirements.lock.txt`, and each release was at least 14 days old on the run date.

**Measured speeds**, in steps per second at batch 32 (`results/profile/`):

| Optimizer | L4 | A10 | L40S | H100 | H100 over L4 |
|-|-|-|-|-|-|
| AdamW | 7.423 | 10.573 | 21.566 | 46.972 | 6.3x |
| Muon | 7.145 | 9.971 | 20.330 | 38.579 | 5.4x |

**Runs finished.** 3 seeds of every cell: 18 DiLoCo main runs and 6 data-parallel main runs at 60M tokens, 8 inner tuning runs and 18 outer tuning runs at 15M tokens, and 9 smoke runs.

## Why the budget was cut to 60M tokens

The plan assumed 300,000 tokens per second on the L40S. The smoke test measured 176,903 for AdamW and 167,409 for Muon. At these speeds, 120M tokens per run projected to 9.01 L40S hours, well past the 5.5 hour hard stop. The rule in `decision_rule.json` takes the largest budget whose projection, including the two-seed extension, is at most 5.4 hours. That gave 60M tokens per main run and 15M per tuning run, projected at 5.24 hours (`results/budget.json`). This is also the lowest budget the rule allows.

The plan's speed came from a nanoGPT figure. Our guess is that the full 50,257-token head explains much of the gap, since nanoGPT's benchmark may use a far smaller vocabulary. We did not test this.

## Main table

The target is 5.7323, the AdamW data-parallel loss at 25% of the budget. This is the pre-registered fallback. The primary target of 5.0567 was missed by AdamW R1 and AdamW R3 in all seeds and by Muon R1 in seed 1. Times are simulated seconds. Means and standard deviations are over seeds 1, 2 and 3. H is listed for L4, A10, L40S and H100.

| Optimizer | Rule | H | Outer (eta, beta) | Loss at 60M | Tokens to target | Time, k = 0 | Time, k = 0.1 | Time, k = 1 | Dollars, k = 0.1 |
|-|-|-|-|-|-|-|-|-|-|
| AdamW | R1 | 30, 30, 30, 30 | 1.0, 0.0 | 5.370 (0.007) | 37.7M | 77.45 | 78.68 | 89.69 | 0.34 |
| AdamW | R2 | 4, 6, 13, 30 | 1.0, 0.0 | 5.048 (0.014) | 21.5M | 15.82 | 17.41 | 31.65 | 0.08 |
| AdamW | R3 | 4, 6, 13, 30 | 0.4, 0.8 | 6.178 (0.119) | not reached | not reached | above 49.18 | not reached | not reached |
| Muon | R1 | 30, 30, 30, 30 | 0.7, 0.9 | 5.069 (0.016) | 19.2M | 41.01 | 41.76 | 48.60 | 0.18 |
| Muon | R2 | 5, 7, 15, 30 | 1.0, 0.0 | 4.731 (0.003) | 13.1M | 10.89 | 11.97 | 21.77 | 0.05 |
| Muon | R3 | 5, 7, 15, 30 | 0.4, 0.8 | 5.027 (0.008) | 18.9M | 15.77 | 17.35 | 31.54 | 0.08 |

For reference, the data-parallel runs reached a mean loss of 4.564 with AdamW and 4.449 with Muon at 60M tokens. They crossed the target at about 15M and 8M tokens. Their inner learning rates were 0.0024 for AdamW and 0.04 for Muon.

## The decision rule

The rule was written before the data was downloaded, and its SHA-256 (4fc05c38...) was logged at the time. Our analysis script refused to run if the file had changed.

| Item | Outcome |
|-|-|
| G0, sound measurement | Passed. Data-parallel runs are stable. R2's final weights are bit-identical to R1's at equal speeds for both optimizers. R3's mean maximum deviation from uniform is 0.011 for AdamW and 0.004 for Muon, against a limit of 0.25. |
| G1, speed matters | Passed. At k = 0.1 the best rule takes 0.221 of R1's time with AdamW (R2 only, since R3 is unmeasured) and 0.287 with Muon. |
| P1, R2 and R3 sooner than R1, with higher loss | Mostly fails. R2 is sooner under both, and so is Muon R3. AdamW R3 never reached the target. R2 and Muon R3 have lower loss than R1 at 60M, not higher. Only AdamW R3 has higher loss. |
| P2 and G2, the interaction | **Did not fire: unmeasured.** g(AdamW) is undefined because AdamW R3 missed the target in all three seeds. |
| P3, rate varies less with H under Muon | Holds. The slope of log(rate) on log H is -0.156 for Muon and -0.201 for AdamW, and the three seeds agree closely. |
| R3 worse than R2 under both | Applies, with a qualification. Muon: 17.35 against 11.97 seconds. AdamW: above 49.18 against 17.41 seconds. |

The qualification concerns AdamW. The value of 49.18 is R3's simulated time at the end of its budget, which it had used up without reaching the target. It is an observed lower bound and not an extrapolation, but the rule does not say whether a lower bound may be used for this outcome. the project lead should decide whether to accept this reading.

The rule adds seeds 4 and 5 only when there is no signal. Here G2 was unmeasured, and more seeds of the same runs would not make g(AdamW) defined, so we did not run them. The severe profile in G1 was not needed either.

The plan says the rule is to be fixed with the project lead. He has not reviewed it. The run agent wrote it, and `decision_rule.json` lists each choice the plan left open.

## What the logs say about R3

We first checked for a bug. The weights in all 133 logged rounds of seed 1 (69 for AdamW, 64 for Muon) recompute exactly from the logged s_i^2, and H_eff matches its formula. The readouts in this section were added after the results were opened, and none of them enters the verdict.

**The weights hardly moved.** Mean u_i over the second half of seed 1 ranges from 0.109 to 0.151 across the two optimizers, and uniform is 0.125. The largest weight went to the H100 workers. The L4 workers received more weight than the A10 and L40S workers under AdamW, so s_i^2 did not separate slow workers from fast ones in any simple way.

**The normalization did change the aggregate.** With nearly uniform u_i, R3 gives each worker's Delta_i a weight of u_i H_eff / H_i, so a short run is scaled up. Over the second half of seed 1 (each pair of workers shares a value, to within 0.01):

| Weight on Delta_i | L4 | A10 | L40S | H100 | Sum over 8 workers |
|-|-|-|-|-|-|
| AdamW, R2 | 0.038 | 0.057 | 0.123 | 0.283 | 1.00 |
| AdamW, R3 | 0.68 | 0.38 | 0.18 | 0.11 | 2.70 |
| Muon, R2 | 0.044 | 0.061 | 0.132 | 0.263 | 1.00 |
| Muon, R3 | 0.46 | 0.34 | 0.18 | 0.11 | 2.17 |

So R3 gave an L4 worker 18 times R2's weight under AdamW and 10 times under Muon. Our reading is that this is the main reason for R3's loss, but we did not isolate it. Running R3's normalization with uniform weights would test it directly.

**The outer choice does not explain it.** R2 and R3 chose different outer settings, so g mixes the rule with the outer step. The tuning runs compare them at matched settings (seed 100, 15M tokens, one run each). R3 was worse in all six comparisons, and the gap was larger for AdamW at every setting:

| Outer (eta, beta) | AdamW, R3 minus R2 | Muon, R3 minus R2 |
|-|-|-|
| (0.7, 0.9) | 1.336 | 0.955 |
| (1.0, 0.0) | 1.122 | 0.502 |
| (0.4, 0.8) | 0.691 | 0.251 |

**The direction is against P2.** At the fallback target, g(Muon) is -0.369, -0.361 and -0.382 in seeds 1, 2 and 3. For AdamW we have only upper bounds, because R3's tokens to target are at least its full budget: -1.044, -1.046 and -1.026. That makes I at most -0.675, -0.685 and -0.644. The loss gap at 60M agrees. R2 is lower than R3 by 1.130 with AdamW and 0.296 with Muon, with the same sign in all seeds. P2 predicted that I would be positive.

**R2's advantage per token was not predicted.** R2 ran 70 rounds (65 for Muon) against R1's 31 in the same tokens, and more of its weight sits on the long runs of the fast workers. Either could explain the lower loss, and we tested neither.

## Problems during the run

**A serious bug in the first smoke attempt.** `torch.nn.utils.vector_to_parameters` makes parameters views of the input vector, so every worker trained the same shared weights and every Delta_i was zero. Training loss still fell, and R1, R2 and R3 gave identical final hashes. We caught it because R3's weights were exactly uniform. The logs are in `results/smoke_first_attempt_zero_delta_bug/`. The fix copies the values and asserts that no Delta_i is zero. Note that the pre-registered control would have passed on this broken code, so it is weaker than it looks. In the repaired run, the smallest norm of Delta_i in the R1 control is 5.70 for AdamW and 7.11 for Muon, and R3's final weights differ from R1's.

**A memory failure.** The same attempt ran out of memory on the data-parallel step, because full-vocabulary logits for 256 sequences need a 12.27 GiB allocation for their gradient alone. We now accumulate the mean gradient over eight batches of 32, which keeps the same global batch.

**Lost log streams.** Two local log streams ended with `Connection lost`. The detached jobs were not affected, and `logs/` holds the app logs recovered afterwards. 

**Code versions.** Data preparation and the speed profile ran on code hash d55ef0a9, before the bug fix. Every smoke result we report, all tuning and all main runs ran on 77dec35f. The fix did not touch the timed step apart from detaching its returned loss, and we did not repeat the profile. The analysis script changed after the results were opened: G1 now accepts one measured rule because the gate reads "R2 or R3", P1 is reported per cell, and the lower bound and the post hoc readouts were added.

## Changes from the plan

- 60M and 15M tokens per run, in place of 120M and 30M, for the reason above.
- The smoke control ran six full rounds and not 1M tokens, because one R1 round is 1.97M tokens and R3's weights cannot leave uniform before round two.
- Each speed profile took about 2.7 minutes, not 5.
- Stage b ran on two containers at once. The plan allows this at the same cost.
- The 41-package lock and the choice of PyTorch 2.10.0 (CUDA 12) were ours. We chose this release because it has `torch.optim.Muon` and stays on the CUDA 12 family used by our earlier experiments.

## Limits

- **Three seeds, a small model, an early stage of training.** Loss was still falling in every run at 60M tokens. AdamW R3 might reach the target with more tokens, and we do not know how the ranking would change.
- **AdamW's learning rate is at the top of its grid.** The value 0.0024 was the largest we tried. It was tuned on data-parallel batches of 256 over 15M tokens and then used at batch 32 for local steps. This may have inflated AdamW's R3 penalty, and we did not test it.
- **Tuning at a quarter of the budget** may favour fast starters, as the plan warned, and warmup takes 4M of those 15M tokens.
- **The common target falls at different stages.** Muon reaches it earlier in training than AdamW, so any comparison of g across optimizers also compares different phases.
- **R2 leans on two workers.** The H100 pair holds 57% of the weight on Delta_i under AdamW and 53% under Muon, so R2 uses the slower GPUs lightly.
- **The equal-speed control cannot see R3's scaling.** With equal H, H_eff / H_i is 1 for every worker, so the behaviour that matters here does not occur in the control.
- **Determinism is lightly tested.** One repeat of R1 on one GPU type gave bit-identical weights, while PyTorch warned that Flash Attention is non-deterministic by default.
- **The clock is logical.** Compression, overlap, asynchrony and real links are left out, and speeds were profiled once. The dollar figures are list prices on this toy model and only their ratios mean anything.

## What we suggest next

These are proposals and have not been run. The rule's outcome points to dropping R3 as written and comparing schedules and existing weights. Before dropping the idea of inverse-noise weights, a small ablation could separate the two things R3 changes: R3's normalization with uniform weights, and inverse-noise weights applied on top of R2's token weights. Extending AdamW R3 to 120M tokens would cost about $1.50 at the rates above, and it could make g(AdamW) defined at the fallback target, but the rule does not provide for it. That decision belongs to the project lead.

## Cost

Modal billed **$9.78** for this experiment on 4 November across 7 app runs, as read after the last run (`results/billing.json`). Our own estimate from runtime and list prices is $9.58 (`results/costlog.jsonl`). The L40S ran for 4.05 hours, against 4.2 planned and the 5.5 hour hard stop. The plan was about $10.60, with a hard stop at $13.55.

| Step | Billed |
|-|-|
| Data preparation | $0.01 |
| Speed profile on four GPU types | $0.44 |
| Smoke test, two attempts | $0.55 |
| Stage a: inner tuning, data-parallel runs | $2.20 |
| Stage b, AdamW | $3.19 |
| Stage b, Muon | $3.38 |

No paid API was used, and no key appears in the code or the logs. All our apps have stopped. We kept our cloud volume, which holds about 0.3 GB of tokens plus the logs, and this storage is not in the figures above. We touched no other Modal resource.

## Files

Everything needed to check these numbers is in `results/exp01/`: `decision_rule.json`, `summary.json`, `runs/` (one JSONL per tuning and main run, with per-round and per-worker records, the seed, the configuration and the code hash), `profile/`, `choices/`, `target.json`, `budget.json` and `manifest.json`. The code is now `islands.py`. The logs carry the hash of the original cloud launcher, which is not in this repository.

The rule file here differs from the one hashed at the time in one way: a name and a cloud profile name were replaced. `results/decision_rule_original_sha256.json` holds the hash of the original.
