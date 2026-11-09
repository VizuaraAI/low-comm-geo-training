# Low-communication training on mixed GPU fleets

A research project in the Vizuara Inference Research Bootcamp (November 2026 to January 2027). The target is a workshop
paper. Candidates are an ICLR 2027 workshop (the list appears on 29 November), EuroMLSys 2027 and NAIC at SIGCOMM 2027.

## The question

DiLoCo-style training lets each worker take many local steps and exchange only the change in its weights. When the
workers run at different speeds, someone has to decide how many steps each one takes and how much say each one gets. We
ask which rule works best on a mixed fleet, whether the answer changes with the inner optimizer, and what it costs to
reach a fixed loss. Papers so far have varied the rule or the optimizer, and we found none that crosses a set of rules
with Muon or a whitening optimizer on mixed workers. `docs/literature-review.md` has the search and its date.

## What we know so far

Two experiments, each on eight simulated workers on one L40S: two each of L4, A10, L40S and H100 at their measured
speeds, a 30M-parameter GPT, FineWeb-Edu split into i.i.d. shards, three seeds, 60M tokens per run. The clock is logical
and there is no real network. Each experiment had a decision rule written before its runs, and the rule file is in
`results/`. Nobody outside the run reviewed either rule. The notes are `docs/exp01-results.md` and
`docs/exp02-results.md`.

What the rules decided in advance:

- Equal time with token weights (R2) reached the common target in 22% of the simulated time of equal steps (R1) under
  AdamW and 29% under Muon, at the link cost we call k = 0.1. We had predicted that R2 would cost more loss per token.
  It cost less, and we do not know why.
- Our first proposal, R3, lost to R2 under both optimizers. For AdamW this rests on a lower bound, because R3 never
  reached the target in its budget. Exp01's main test, whether the optimizer changes the gain from R3, could not be
  computed for the same reason.
- Exp02 split R3 into its two changes. Its normalization alone (R3a) hurt under both optimizers and accounts for all of
  R3's loss (137% and 141% of the gap).
- Its inverse-noise weights alone (R3b) showed no detectable effect against R2: gains of +0.004, -0.005, +0.016 nats for
  AdamW and -0.010, -0.013, -0.010 for Muon over the three seeds.
- The interaction between the optimizer and those weights was I = +0.016, which the rule calls no signal. This is close.
  The rule requires |I| of at least 0.02, a floor we set before the run. Without that floor the same test on loss would
  fire (2.13 sigma), while the same test on time to target would not (1.74 sigma). Three seeds cannot settle a gap of this
  size.
- On these shards, two workers on the same GPU type agree on the noise estimate to within 1.5%. The estimator therefore
  encodes H_i and nothing else about a worker, as we expected before the run. So Exp02 did not test inverse-noise
  weighting where noise really differs between workers, since in this setup it does not.

Our readings after the fact, which are post hoc and carry less weight: that R3b leads R2 early and loses the lead by
60M tokens, and that the tilt moves weight toward long runs. The reason R2 beats R1 per token is also open.

So the paper can say, at this small scale and early stage of training, that equal time beats equal steps, and that R3's
normalization should go. It cannot say that inverse-noise weights help or fail in general. The next test needs workers
whose data really differ. Limits that apply throughout: three seeds, a small model, loss still falling at 60M tokens,
AdamW's learning rate at the top of its grid, a logical clock, and speeds profiled once.

## How we work: agentic-first

Coding is done by an agent, Claude Code or Codex. Your job is to understand each experiment, run it through the agent,
check what comes back and think about what it means.

1. Clone this repository and open a terminal in it.
2. Start the agent here: `claude` for Claude Code, or `codex` for Codex.
3. Paste `prompts/00-setup.md` first. It sets up the environment and checks that our numbers reproduce.
4. Then paste `prompts/A-skewed-shards.md` and `prompts/D-price-of-a-link.md`, one at a time. Before each one, read its
   explanation below and make sure you could defend the design to a colleague. Each prompt asks for a decision rule
   before the runs. Write your own expectations into it, and keep it fixed once the runs start.
5. After each one, read the agent's `NOTES.md` and table. If a number looks wrong, ask the agent to check it against the
   logs.
6. Send us the tables, the figures and a few lines per ablation on what surprised you.

You can start with D, which needs only a laptop. A is the only GPU ablation for now, and it needs one GPU with at least
24 GB (peak use on our L40S was about 21 GB). A 60M-token run took about 8 minutes on an L40S. Our cost measure below is
Modal's list price for one L40S with 4 CPU cores and 16 GB of memory, which came to $4.50 to $5 per billion tokens in our
runs.

## The ablations, and why each matters for the paper

**A. Shards that really differ ($11 to $12).** This is the test the paper now needs. The pool is the same throughout: 75M
English and 75M Hindi Wikipedia tokens. We change only how it is split, from half and half on every worker to one
language per worker, with each GPU type holding one worker of each language. If the noise estimate now separates two
workers on the same GPU type, inverse-noise weights have something to use for the first time. One caution matters more
than the others. The estimator measures distance from the aggregate, so a language that is merely different looks like a
noisy one. The primary measure is therefore a language-balanced loss, with each language's loss beside it. The
nearest published setup is HeLoCo, which runs five languages on five workers asynchronously, with language tied to speed;
it does not vary the rule across inner optimizers.

**D. The price of a link (no GPU).** Because the clock is logical, every logged run can be repriced at any link cost
without training. Early arithmetic suggests that equal steps can overtake equal time once links are slow enough, because
R2 needs more rounds. You will work out where that happens and what link speed it corresponds to at this scale, which
tells us what the live measurements should cover. All of it is post hoc analysis of runs designed for another purpose.

Two more GPU ablations, on language tied to speed and on speed-proportional batches, are held for now. After A and D, we
fix the paper's scope together, choose the third inner optimizer (SOAP or SPlus), and move to the larger
model and real links.

## Repository

- `islands.py`: the simulator, the rules and the controls. The model, optimizers, rules and run loop are those of Exp01
  and Exp02 with the cloud launcher removed. On a tiny CPU model it gives the same final weights as the original code.
- `prepare_data.py`: builds the token files at pinned revisions and checks their hashes. Text is not in this repository.
- `data/`: the English and Hindi tokenizer used by A, and the expected hashes of their token files.
- `results/exp01/`, `results/exp02/`: decision rules, summaries, tuning choices, GPU speed profiles and every run log, one
  JSONL per run with per-round and per-worker records.
- `tests/test_core.py`: checks that need no GPU, including that each control fires on the fault it is there to catch.
- `prompts/`: the prompts for the agent.
- `docs/`: the literature review, the first experiment's plan and the two lab notes.

## Data and licence

FineWeb-Edu is under ODC-BY at the revision we pin, and the GPT-2 tokenizer is under MIT. For Wikipedia the dataset card
lists CC BY-SA 3.0 and GFDL, and the tokenizer in `data/` is trained from it. The code is released under the MIT licence.
