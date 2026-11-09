"""Build the token files the simulator reads. Runs on a CPU, in about ten minutes for each dataset.

    python prepare_data.py fineweb      FineWeb-Edu with the GPT-2 tokenizer, the data of Exp01 and Exp02
    python prepare_data.py wiki-en-hi   English and Hindi Wikipedia with a tokenizer trained on both, for ablation A

Both write a manifest with SHA-256 hashes. `fineweb` stops if its hashes differ from the manifest of our runs in
results/exp01/manifest.json. Text is never committed to this repository: it is downloaded at pinned revisions.

Why wiki-en-hi has its own tokenizer: GPT-2's byte-level vocabulary gives Hindi about 0.67 characters per token and
English about 4.3, so a Hindi worker would see mostly byte fragments. A BPE vocabulary of the same size (50,257, so the
model and its parameter count do not change) trained on equal parts of both gave about 1.5 and 4.4 in our test.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EOT = 50256  # end-of-text id for GPT-2; the trained tokenizer stores its own

FINEWEB = "HuggingFaceFW/fineweb-edu"
FINEWEB_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"   # licence odc-by on the card at this revision
FINEWEB_DIR = "data/CC-MAIN-2013-20"
GPT2 = "openai-community/gpt2"
GPT2_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"       # licence mit
N_TRAIN, N_VAL = 150_000_000, 2_000_000
DATA_SEED = 9009

WIKI = "wikimedia/wikipedia"                                # the dataset card lists CC BY-SA 3.0 and GFDL; dump of 1 November 2023
WIKI_FILES = {"en": ["20231101.en/train-00000-of-00041.parquet"],
              "hi": ["20231101.hi/train-00000-of-00002.parquet", "20231101.hi/train-00001-of-00002.parquet"]}
WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_MIN_CHARS = 1500
WIKI_TRAIN, WIKI_VAL = 75_000_000, 1_000_000                # per language, so the pool has Exp01's 150M training tokens
TOK_TRAIN_CHARS = 60_000_000                                # per language


def sha_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def fineweb(out):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer

    info = HfApi().dataset_info(FINEWEB, revision=FINEWEB_REV)
    assert info.sha == FINEWEB_REV, "dataset revision does not match"
    names = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(FINEWEB_DIR + "/"))
    tk = Tokenizer.from_pretrained(GPT2, revision=GPT2_REV)
    tk.no_truncation()
    tk.no_padding()
    docs, total = [], 0
    need = int((N_TRAIN + N_VAL) * 1.01)
    for name in names:
        path = hf_hub_download(FINEWEB, name, repo_type="dataset", revision=FINEWEB_REV)
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=2000, columns=["text"]):
            for e in tk.encode_batch_fast(b.column("text").to_pylist(), add_special_tokens=False):
                a = np.array(e.ids + [EOT], dtype=np.uint16)
                docs.append(a)
                total += len(a)
            if total >= need:
                break
        print(f"{name}: total tokens {total}", flush=True)
        if total >= need:
            break
    rng = np.random.default_rng(DATA_SEED)
    order = rng.permutation(len(docs))
    val, nv, k = [], 0, 0
    while nv < N_VAL:
        val.append(docs[order[k]])
        nv += len(docs[order[k]])
        k += 1
    train, nt = [], 0
    while nt < N_TRAIN:
        train.append(docs[order[k]])
        nt += len(docs[order[k]])
        k += 1
    train = np.concatenate(train)[:N_TRAIN]
    val = np.concatenate(val)[:N_VAL]
    out.mkdir(parents=True, exist_ok=True)
    train.tofile(out / "train.u16.bin")
    val.tofile(out / "val.u16.bin")
    man = dict(dataset=FINEWEB, revision=FINEWEB_REV, tokenizer=GPT2, tokenizer_revision=GPT2_REV, train_tokens=len(train),
               val_tokens=len(val), train_sha256=sha_array(train), val_sha256=sha_array(val))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    ref = json.load(open(HERE / "results/exp01/manifest.json"))
    same = man["train_sha256"] == ref["train_sha256"] and man["val_sha256"] == ref["val_sha256"]
    print("hashes match our runs" if same else "HASHES DIFFER from our runs: check the package versions in requirements.lock.txt")
    return 0 if same else 1


def wiki_texts(lang, local=None):
    """All articles of at least WIKI_MIN_CHARS characters in file order."""
    import pyarrow.parquet as pq
    texts = []
    for name in WIKI_FILES[lang]:
        if local:
            path = Path(local) / {WIKI_FILES["en"][0]: "en0.parquet", WIKI_FILES["hi"][0]: "hi0.parquet",
                                  WIKI_FILES["hi"][1]: "hi1.parquet"}[name]
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(WIKI, name, repo_type="dataset", revision=WIKI_REV)
        texts += [t.strip() for t in pq.read_table(path, columns=["text"])["text"].to_pylist()]
    return [t for t in texts if len(t) >= WIKI_MIN_CHARS]


def take_chars(texts, n):
    out, c = [], 0
    for t in texts:
        out.append(t)
        c += len(t)
        if c >= n:
            break
    return out


def wiki_en_hi(out, local=None):
    """Writes en.train, en.val, hi.train, hi.val as uint16. A document never appears in both training and validation."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    t0 = time.time()
    rng = np.random.default_rng(DATA_SEED)
    texts = {}
    for lang in ("en", "hi"):
        L = wiki_texts(lang, local)
        order = rng.permutation(len(L))
        texts[lang] = [L[i] for i in order]
        print(f"{lang}: {len(L)} articles of at least {WIKI_MIN_CHARS} characters, {time.time() - t0:.0f} s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = HERE / "data" / "tokenizer_en_hi_50k.json"
    if not tok_path.exists():
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=50257, special_tokens=["<|endoftext|>"], show_progress=False,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(take_chars(texts["en"], TOK_TRAIN_CHARS) + take_chars(texts["hi"], TOK_TRAIN_CHARS), trainer=tr)
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(tok_path))
        print(f"tokenizer trained, {time.time() - t0:.0f} s", flush=True)
    tok = Tokenizer.from_file(str(tok_path))
    eot = tok.token_to_id("<|endoftext|>")
    man = dict(dataset=WIKI, files=WIKI_FILES, revision=WIKI_REV, min_chars=WIKI_MIN_CHARS, split_seed=DATA_SEED,
               tokenizer_sha256=hashlib.sha256(tok_path.read_bytes()).hexdigest(), vocab=tok.get_vocab_size(), languages={})
    for lang in ("en", "hi"):
        val, train, nv, nt, chars = [], [], 0, 0, 0
        for i in range(0, len(texts[lang]), 2000):
            batch = texts[lang][i:i + 2000]
            for s, e in zip(batch, tok.encode_batch_fast(batch, add_special_tokens=False)):
                a = np.array(e.ids + [eot], dtype=np.uint16)
                if nv < WIKI_VAL:
                    val.append(a)
                    nv += len(a)
                elif nt < WIKI_TRAIN:
                    train.append(a)
                    nt += len(a)
                    chars += len(s)
            if nt >= WIKI_TRAIN:
                break
        assert nt >= WIKI_TRAIN and nv >= WIKI_VAL, f"{lang}: only {nt} training tokens available; lower WIKI_TRAIN or the length filter"
        tr_arr = np.concatenate(train)[:WIKI_TRAIN]
        va_arr = np.concatenate(val)[:WIKI_VAL]
        tr_arr.tofile(out / f"{lang}.train.u16.bin")
        va_arr.tofile(out / f"{lang}.val.u16.bin")
        man["languages"][lang] = dict(train_tokens=len(tr_arr), val_tokens=len(va_arr), chars_per_token=round(chars / nt, 3),
                                      train_sha256=sha_array(tr_arr), val_sha256=sha_array(va_arr))
        print(f"{lang}: {man['languages'][lang]}, {time.time() - t0:.0f} s", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("fineweb", "wiki-en-hi"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--local-parquet", default=None, help="folder that already holds en0, hi0 and hi1 .parquet")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if a.dataset == "fineweb":
        raise SystemExit(fineweb(Path(a.out or HERE / "data/fineweb")))
    raise SystemExit(wiki_en_hi(Path(a.out or HERE / "data/wiki_en_hi"), a.local_parquet))


if __name__ == "__main__":
    main()
