# Exp02 R3 split, topic 09: lab note

Run by the Vizuara research team on 4 November 2026.

## In short

Exp01's R3 lost to R2, but R3 changed two things at once. We separated them. R3a keeps R3's normalization and drops the inverse-noise weights. R3b keeps the inverse-noise weights on top of token weights and drops the normalization.

The normalization caused R3's loss. R3a alone is worse than R2 by 1.54 nats with AdamW and 0.42 with Muon, which is more than all of R3's gap (137% and 141% of it).

The weights themselves did almost nothing at 60M tokens. R3b minus R2 in loss is -0.005 with AdamW and +0.011 with Muon, and the rule calls both NO_EFFECT_DETECTED. The Muon difference is below our 0.02 threshold, but it has the same sign in all three seeds, so it looks real and small.

The interaction is defined this time, because every run has a loss at the budget. For R3b, I = +0.016, which the rule calls NO_SIGNAL. This call depends on the 0.02 minimum size we added. Without that minimum the same test would fire, since all seeds share a sign and |I| is 2.13 sigma. On Exp01's own measure, g at the fallback target, it does not fire (1.74 sigma).

On these i.i.d. shards, the noise estimate carries no information about a worker beyond its H_i. Two workers on the same GPU type agree to within 1.5%. So R3b is a fixed tilt of the token weights toward long runs and not an adaptive noise estimate. This experiment did not test inverse-noise weighting where noise truly differs between workers, because in this setup it does not.

## What we ran

Everything is Exp01's unless stated here: the fleet, data, model, inner learning rates, 60M token budget, 15M tuning budget, seeds and clock. All runs carry code hash 1b86790a.

| Arm | Weight c_i on Delta_i | Note |
|-|-|-|
| R2 | H_i / sum H_j | Control, same lines as Exp01 |
| R3a | (1/8) H_eff / H_i | R3 with u_i fixed at 1/8. The weights sum to 2.79 (AdamW) and 2.33 (Muon) |
| R3b | H_i u_i / sum H_j u_j | u_i from R3's estimator without change. The weights sum to 1 |

**Runs finished.** 26 in total: 12 outer tuning runs at 15M tokens (seed 100), 12 main runs at 60M (R3a and R3b, seeds 1, 2, 3, both optimizers) and 2 R2 control runs at 60M (seed 1). There were also 12 short smoke runs and the injected bug run. No run diverged and none was stopped.

**Outer settings.** Tuning picked (1.0, 0.0) for R3b, the same as R2, and (0.4, 0.8) for R3a, the same as R3, under both optimizers. Each was the outright best in its table, so the anchor rule made no difference. This means R3b against R2, and R3a against R3, each differ in one thing only. R3a against R2 also differs in the outer step.

**Cut rule.** The smoke test projected $5.49 against the $7.50 limit, so nothing was cut. All arms ran at 60M tokens and three seeds.

## The control and the bridge

The code had to change, so Exp01's logs could not be reused on their hash alone. We reran R2 at seed 1 for both optimizers. The final weights are bit-identical to Exp01's seed 1 under both optimizers (`bridge` in `summary.json`). Per-worker norms are equal in all 70 rounds for AdamW and all 65 for Muon, and no setting differs. Losses are 5.0327 and 4.7271, the same as Exp01's seed 1.

The bridge rule compares seed 1 with seed 1 and not with the three-seed mean. Exp01's seed 1 sits 1.06 SD (AdamW) and 1.10 SD (Muon) from its own mean, so an exact repeat would fail a test against the mean. The contrasts below use Exp01's R2 for seeds 1, 2 and 3.

## Main table

Loss is validation loss at 60M tokens, with mean and SD over seeds 1, 2, 3. Gain is R2 minus the arm in the same seed, so a positive gain means the arm is better than R2. The target for the last two columns is Exp01's fallback, 5.7323. R1, R2 and R3 come from Exp01's logs. Time is simulated seconds at k = 0.1.

| Optimizer | Rule | Outer (eta, beta) | Loss at 60M | Gain over R2, seeds 1, 2, 3 | Tokens to target | Time |
|-|-|-|-|-|-|-|
| AdamW | R1 | 1.0, 0.0 | 5.370 (0.007) | reference only | 37.7M | 78.68 |
| AdamW | R2 | 1.0, 0.0 | 5.048 (0.014) | control | 21.5M | 17.41 |
| AdamW | R3 | 0.4, 0.8 | 6.178 (0.119) | reference only | not reached | above 49.18 |
| AdamW | R3a | 0.4, 0.8 | 6.591 (0.013) | -1.554, -1.531, -1.545 | not reached | above 49.18 |
| AdamW | R3b | 1.0, 0.0 | 5.043 (0.013) | +0.004, -0.005, +0.016 | 20.8M | 16.84 |
| Muon | R1 | 0.7, 0.9 | 5.069 (0.016) | reference only | 19.2M | 41.76 |
| Muon | R2 | 1.0, 0.0 | 4.731 (0.003) | control | 13.1M | 11.97 |
| Muon | R3 | 0.4, 0.8 | 5.027 (0.008) | reference only | 18.9M | 17.35 |
| Muon | R3a | 0.4, 0.8 | 5.149 (0.005) | -0.416, -0.418, -0.420 | 20.2M | 18.46 |
| Muon | R3b | 1.0, 0.0 | 4.742 (0.004) | -0.010, -0.013, -0.010 | 12.9M | 11.79 |

## The decision rule

The rule was written before any Exp02 run, and its SHA-256 (db49227a, logged at 10:46 IST) was fixed before the smoke test. Our analysis script refused to run if the file had changed. the project lead has not reviewed the rule.

| Item | Outcome |
|-|-|
| Smoke gate | Passed. Details are under Controls below |
| Bridge | Passed under both optimizers, by bit-identical weights |
| R3b effect, AdamW | NO_EFFECT_DETECTED. Mean gain +0.005, SD 0.010 |
| R3b effect, Muon | NO_EFFECT_DETECTED. Mean gain -0.011, SD 0.002. All seeds share a sign and the mean exceeds 2 SD, but it is below 0.02 |
| Interaction, R3b (primary) | NO_SIGNAL. I = +0.016 (seeds +0.014, +0.008, +0.026), sigma_g 0.007, in the direction of P2. All seeds share a sign and abs(I) > 2 sigma_g, but abs(I) < 0.02 |
| Interaction, R3a (secondary) | SIGNAL. I = -1.125 (seeds -1.138, -1.113, -1.125), sigma_g 0.008, opposite of P2. AdamW loses more than Muon |
| R3a effect | HURTS under both optimizers |
| NORMALIZATION_EXPLAINS_R3 | Fires under both. Mean share 1.37 (AdamW), 1.41 (Muon), against the bound of 0.75 |
| H_ONLY | Holds under both |
| HARDLY_DIFFER | Holds for Muon. Narrowly fails for AdamW (A 0.268 against 0.25; E 0.105 against 0.10) |
| Extension to seeds 4 and 5 | Not triggered. The rule asks for abs(I) or abs(mean gain) of at least 0.02 |

The outcome text fixed in advance applies as follows. R3b_NO_EFFECT_both: on i.i.d. shards at this scale, inverse-noise weights add nothing detectable to token weights, and the paper may not claim a benefit. This does not reject the idea where noise truly differs between workers. NORMALIZATION_EXPLAINS_R3: Exp01's R3_WORSE outcome applies to R3 as written and does not reach the weighting idea.

**The minimum size decides the interaction call.** We added delta = 0.02 before the run, because paired seeds can give a very small sigma_g, and a trivial gap could otherwise count as a signal. Without it, the same test on loss would read SIGNAL here, while the test on time to target would not. In loss terms I is about 1.4M AdamW tokens or 2.8M Muon tokens at the R2 slopes near the budget. A t interval on the three I_s runs from -0.007 to +0.038 (post hoc), so it contains both zero and 0.02. We would read this as a small effect that three seeds cannot settle, in the direction the plan predicted.

**Time to target.** All R3b runs reached the fallback target, so g is defined without any bound: +0.033 (AdamW) and +0.015 (Muon), with I = +0.018 at 1.74 sigma. This is NO_SIGNAL as well. R3b crossed the target about 3% and 1.5% sooner than R2, and that is a statement about early training, as the next section shows.

## Can inverse-noise weights matter on i.i.d. shards?

Before the run we argued that they have no worker-specific information to use. Every worker draws from the same distribution with the same batch, so a worker's noise can depend only on its H_i. The logs support this. These readouts cover the second half of rounds of R3b, as means over seeds.

| Readout | AdamW | Muon |
|-|-|-|
| s^2 between GPU types, largest over smallest (B) | 1.49 | 1.40 |
| s^2 within a same-GPU pair, largest over smallest, worst pair and seed (C) | 1.015 | 1.005 |
| Noise from same-GPU pairs, squared distance of d_a and d_b over 2, L4 over H100 (D) | 2.06 | 1.72 |
| Mean of max abs(8 u_i - 1) (A) | 0.268 | 0.206 |
| Mean u_i for L4, A10, L40S, H100 (uniform is 0.125) | 0.106, 0.111, 0.125, 0.158 | 0.107, 0.114, 0.128, 0.151 |
| H100 share of the weight, R2 then R3b | 56.6%, then 63.5% | 52.6%, then 58.1% |
| n_eff = 1 / sum c_i^2, R2 then R3b | 5.01, then 4.32 | 5.42, then 4.83 |
| Step change against R2 on the same Deltas (E) | 0.105 | 0.095 |
| Step norm over R2's step norm | 1.09 | 1.08 |
| Cosine with R2's step | 0.999 | 0.999 |

Three things follow.

**The estimator encodes H and nothing else.** Workers on the same GPU type have s^2 within 1.5% of each other, and the between-type ratio is at most 1.49. The same held in Exp01's R3 (at most 1.013 and 1.005). s^2 is ordered by H, from L4 down to H100, and it tracks the pair-based noise, though in a compressed form. So inverse-noise weighting here amounts to a fixed function of H. A true adaptive benefit cannot appear in this design, whatever the result.

**The tilt pushes weight toward long runs.** A short run has a noisier per-step rate, so u_i rises with H_i, and the H100 pair gains weight. n_eff falls under both optimizers. We predicted this direction. We also predicted that 8 u_i would reach about 1.6 and E about 0.2, and the measured tilt is smaller than that: 8 u_i reaches about 1.26 and E is about 0.10.

**The resulting step is nearly R2's step, slightly longer.** The cosine with R2's step is 0.999 and the norm ratio is 1.08 to 1.09. A change this small in the outer step is consistent with a small effect on loss. Weights in general can matter a lot here, as R3a shows. It is this rule at this operating point that moves them too little to matter.

**The gain changes over the run (post hoc).** R3b leads R2 early and then loses the lead. Mean gain by tokens:

| Tokens | 5M | 10M | 15M | 20M | 30M | 40M | 50M | 60M |
|-|-|-|-|-|-|-|-|-|
| AdamW | +0.027 | +0.024 | +0.027 | +0.024 | +0.017 | +0.011 | +0.009 | +0.005 |
| Muon | +0.018 | +0.015 | +0.011 | +0.001 | -0.007 | -0.009 | -0.010 | -0.011 |
| I | +0.010 | +0.009 | +0.016 | +0.023 | +0.024 | +0.020 | +0.019 | +0.016 |

The interaction stays positive at every checkpoint and reaches 0.02 only at 20M and 30M. These checkpoints come from the same runs, so they are not independent evidence. We did not fix this curve in advance, and the verdict uses only the value at 60M. Since Muon's gain changes sign between 20M and 30M, the answer at 60M may not hold at a longer budget.

## What the logs say about R3a

R3a is far from R2's weights. An L4 worker gets 0.661 against 0.038 under AdamW (17.5 times) and 0.526 against 0.044 under Muon (12 times). The H100 share falls to about 18%.

The outer step does not explain the loss. At R2's own setting (1.0, 0.0), seed 100 and 15M tokens, R3a reaches 6.940 against R2's 5.960 with AdamW, and 6.147 against 5.557 with Muon.

We expected noise to be the mechanism, and the logs do not support that simple reading. On the same Deltas, before the outer step, R3a's step has 0.94 (AdamW) and 0.87 (Muon) of the norm of the token-weighted step. Its off-diagonal share is higher, 0.71 against 0.47 and 0.49 against 0.30 (post hoc). So the workers agree more inside R3a's step, and yet the loss is far worse. Our guess is that short runs agree because they replay stale inner momentum, since inner state is never synchronized. We did not test this.

R3 itself is better than R3a in every seed, by 0.41 nats with AdamW (0.52, 0.42, 0.30) and 0.12 with Muon (0.12, 0.12, 0.12). Both use (0.4, 0.8), and R3's u_i stayed close to uniform, so a modest shift of weight inside the normalization changes the loss a good deal. We do not read this as support for P2. It looks more like the tilt undoing a part of the damage the normalization does.

## What this means for the paper

- R2, equal time with token weights, remains the best rule we have measured. R3b did not beat it.
- R3's normalization should be dropped. Exp01's result was mostly a statement about it.
- The paper may not claim that inverse-noise weights help on i.i.d. shards. It also may not claim that they fail in general, since this setup has no differences in noise between workers to exploit.
- For the weighting rule, we found no evidence at the pre-registered size that the optimizer and the rule must be chosen together. Three seeds can miss an interaction of this size, and this one is close to the line. The schedule contrast from Exp01's logs agrees (R2 over R1, I = -0.015, sigma_g 0.019, NO_SIGNAL).
- For the normalization, the interaction is large (I = -1.125). It says that a bad aggregation rule hurts AdamW more than Muon at this stage, which is opposite to P2. The two optimizers are at different stages of training here, so we would not make much of the size.

## Controls

**The zero-Delta bug.** Smoke injected Exp01's actual bug into a short run, and the new assert fired with "zero Delta: a worker returned Delta_i = 0". Exp01's own control had passed on broken code, so this time we tested the control against the fault.

**Weights respond to noise.** The unit tests feed the assert uniform u_i with differing s_i^2, and token weights with differing s_i^2, and it fires on both. They also confirm that a noisy synthetic worker gets a smaller u_i. On the real fleet, R3b's max abs(8 u_i - 1) was 0.60 to 0.66 (AdamW) and 0.29 to 0.36 (Muon) over smoke rounds 2 to 4.

**Equal speeds.** With H = 10 for all, R3a's final weights are bit-identical to R2's under both optimizers. R3b's differ from R2's, and its mean deviation from uniform is 0.025 and 0.022, against the limit of 0.25.

**Main runs.** The asserts held in all 26 runs. The smallest norm of any Delta_i in the R3a and R3b main runs is 3.25 (AdamW) and 4.42 (Muon), and the smallest distance between two workers' Deltas is 2.14 and 2.86. Across 405 R3b rounds, u_i and c_i recompute exactly from the logged s_i^2 and H_i, and no round reached the clip (`results/exp02/posthoc.json`).

**Data and settings.** The data hashes match Exp01's manifest. The schedules, speeds and all header fields match Exp01's seed 1 run. Early agreement was visible in smoke, where R2's norms over four rounds equalled Exp01's exactly.

## Problems during the run

**A launch that blocked.** `modal run` in detached mode keeps streaming logs, so our chained launch sat on the AdamW stage and had not started Muon. We stopped the local client, confirmed that the detached app kept running, and launched Muon separately about five minutes later. No run was affected. The local AdamW log ends early, and `logs/stage_main_adamw_app_logs.log` holds the full record. The same file for Muon is there as well.

**A self test tolerance.** One self test failed locally because h_eff is computed in float32, as it was in Exp01. We loosened that test's tolerance from 1e-9 to 1e-5 before hashing and before any Modal spend.

**Timestamp.** The rule's text says about 10:50 IST. The logged hash says 10:46 IST, and that is the correct time.

## Limits

- **Three seeds and six tests.** Two optimizers, two arms and two interactions were read together, with no correction. The t interval on the AdamW gain is -0.021 to +0.031, so an effect of 0.02 either way is not excluded there. Under Muon it is -0.015 to -0.007.
- **The 0.02 threshold is ours.** the project lead has not reviewed it, and it decides the R3b interaction call.
- **I.i.d. shards with equal batches.** The design has no noise difference between workers, except through H_i.
- **R3b's estimator is R3's.** Its reference rate is the u-weighted mean. A reference based on token weights, or a noise estimate from same-GPU pairs, might behave differently, and we tried neither.
- **R3a mixes two things.** It changes the split between workers, and with (0.4, 0.8) it also changes the outer step. The outer grid has no setting that scales the step down to match its weight sum of 2.79, so we cannot say whether a much smaller eta would rescue it.
- **R3 was not rerun.** R3 values come from Exp01. Only the R2 path was shown to repeat exactly.
- **An early stage of training.** Loss was still falling in every run at 60M tokens. The losses also compare different stages between optimizers, most of all for R3a, where AdamW is still at 6.59.
- **Limits carried from Exp01.** AdamW's learning rate is at the top of its grid, the model is small, the clock is logical and speeds were profiled once.

## What we suggest next

These are proposals and have not been run.

- Non-i.i.d. shards or unequal batches, which give noise that truly differs between workers. This is the only setting in which adaptive inverse-noise weighting can be tested, and it is already on the plan's list.
- Seeds 4 and 5 on R2 and R3b would tighten I. They would cost about $2.47 and fit the $8 limit, but the rule does not call for them. That decision belongs to the project lead, along with whether 0.02 is the right minimum size.

## Cost

Modal billed **$4.93** for this experiment across 3 app runs, as read after the last run (`results/billing.json`). Our own estimate from runtime and list prices is $4.86 (`results/costlog.jsonl`). The L40S ran for 2.14 hours. The plan was about $5.50, with the limit at $8 and the hard stop at $9. With Exp01's $9.78, this student's total is $14.71.

| Step | Billed |
|-|-|
| Smoke test | $0.24 |
| Tuning and main runs, AdamW | $2.49 |
| Tuning and main runs, Muon | $2.20 |

No paid API was used, and no key appears in the code or the logs. All our apps have stopped. We reused our cloud volume and added the folder `exp02/` to it, and this storage is not in the figures above. We touched no other Modal resource.

## Files

Everything needed to check these numbers is in `results/exp02/`: `decision_rule.json`, `summary.json` (the verdict and every pre-registered number), `posthoc.json` (readouts added after the results were opened, none of which enters the verdict), `runs/` (one JSONL per tuning and main run, with per-round and per-worker records, the matrix of inner products, the seed, the configuration and the code hash), `choices/` and `plan.json`. The code is now `islands.py`.
