Read README.md, islands.py and docs/exp02-results.md first.

Create a Python virtual environment in .venv. On a Linux GPU machine with Python 3.11, install requirements.lock.txt with `pip install --require-hashes -r requirements.lock.txt`. Add matplotlib==3.9.2 for the figures. On a laptop, install requirements.txt, which holds the same top-level versions without hashes and is enough for the tests and for ablation D. Tell me which of these you did.

Run `python tests/test_core.py`. It uses a tiny model on random tokens and needs no GPU. Tell me whether it passes.

If this machine has a GPU with at least 24 GB, run `python prepare_data.py fineweb`, which downloads about 2.4 GB and must report that the hashes match our runs, and then `python islands.py bridge --opt adamw` and `python islands.py bridge --opt muon`. Report the largest relative difference in per-worker norms for each. On an L40S with the locked PyTorch build we expect exact agreement. On another GPU type, report the number and do not treat a small difference as a failure. Report the peak GPU memory.

Then explain to me, in six plain sentences: what H_i and Delta_i are; how R2 and R3b differ; what the logical clock charges for a round, and what k means; why the code asserts that no Delta_i is zero; what the s_i^2 estimator measures; and why Exp02 found that on these shards it can only encode H_i.

Do not change islands.py, prepare_data.py or anything in results/exp01 or results/exp02.
