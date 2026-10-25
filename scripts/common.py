"""Shared paths, data loaders and model helpers for the audit scripts."""
import os
import json
import glob
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))  # keep all downloads inside the project

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
PRIVATE = DATA / "private"  # git-ignored: book text, hand-written questions
for d in (RESULTS, FIGURES):
    d.mkdir(exist_ok=True)

SEED = 0
N_PASSAGES = 300

MODELS = {
    "qwen3.5-orig": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5-issai": "issai/Qwen3.5-4B-Base-Kazakh",
}
# Tokenizer-only references. Gated: accept the terms on Hugging Face and `hf auth login` first.
REFERENCE_TOKENIZERS = {
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "sherkala": "inception42/Llama-3.1-Sherkala-8B-Chat",
    "sozkz": "stukenov/sozkz-core-gpt2-50k-kk-base-v1",
}
BELEBELE_LANGS = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}

# KazMMLU subjects that exist in both the Kazakh and the Russian track (Math/Informatics
# left out of the text sets because formulas are not running prose).
KAZMMLU_TEXT_SUBJECTS = ["Biology", "Chemistry", "Geography", "Kazakh History",
                         "Law", "Physics", "World History"]


def belebele_passages():
    """300 unique parallel FLORES passages (same row = same passage in every language)."""
    cache = DATA / "belebele_passages.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from datasets import load_dataset
    data = {k: load_dataset("facebook/belebele", cfg, split="test") for k, cfg in BELEBELE_LANGS.items()}
    seen, idx = set(), []
    for i, passage in enumerate(data["kk"]["flores_passage"]):
        if passage not in seen:
            seen.add(passage)
            idx.append(i)
        if len(idx) == N_PASSAGES:
            break
    texts = {k: [data[k][i]["flores_passage"] for i in idx] for k in BELEBELE_LANGS}
    cache.write_text(json.dumps(texts, ensure_ascii=False))
    return texts


def kazmmlu_dir():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("MBZUAI/KazMMLU", repo_type="dataset"))


def kazmmlu(split="test", dtype=None):
    import pandas as pd
    frames = [pd.read_csv(f, dtype=dtype) for f in sorted(glob.glob(str(kazmmlu_dir() / "*" / f"{split}.csv")))]
    return pd.concat(frames, ignore_index=True)


def kazmmlu_texts(lang, n=N_PASSAGES):
    """Native (not translated) text: question + options of n sampled KazMMLU items."""
    df = kazmmlu()
    track = {"kk": "in kaz)", "ru": "in rus)"}[lang]
    keep = df["Subset"].apply(lambda s: s.endswith(track) and s.split(" (")[0] in KAZMMLU_TEXT_SUBJECTS)
    df = df[keep].sample(n=n, random_state=SEED)
    texts = []
    for _, r in df.iterrows():
        opts = [str(r[f"Option {c}"]) for c in "ABCDE" if isinstance(r[f"Option {c}"], str)]
        texts.append("\n".join([str(r["Question"]).strip()] + opts))
    return texts


def private_texts(name):
    """Plain-text files in data/private/<name>/, one document per file. Empty if absent."""
    files = sorted((PRIVATE / name).glob("*.txt"))
    return [f.read_text(encoding="utf-8") for f in files]


def text_sets():
    """All text sets as {set_name: {lang: [texts]}}; private sets appear only if present."""
    bb = belebele_passages()
    sets = {
        "belebele": bb,
        "kazmmlu": {"kk": kazmmlu_texts("kk"), "ru": kazmmlu_texts("ru")},
    }
    alash = private_texts("alash")
    if alash:
        sets["alash"] = {"kk": alash}
    return sets


def bootstrap_ci(values_a, values_b=None, n=2000, seed=SEED):
    """95% percentile bootstrap CI of sum(a)/sum(b) (or mean(a)) resampling items."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=float)
    b = np.ones_like(a) if values_b is None else np.asarray(values_b, dtype=float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    stats = a[idx].sum(1) / b[idx].sum(1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def pick_gpu(min_free_gb=14):
    """Use the GPU with the most free memory; refuse to start on a crowded shared node."""
    import torch
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return 0
    free = [torch.cuda.mem_get_info(i)[0] / 1e9 for i in range(torch.cuda.device_count())]
    best = max(range(len(free)), key=free.__getitem__)
    if free[best] < min_free_gb:
        raise SystemExit(f"No GPU with {min_free_gb} GB free (best: GPU {best}, {free[best]:.1f} GB)")
    print(f"using GPU {best} ({free[best]:.1f} GB free)")
    return best


def load_model(name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    repo = MODELS[name]
    gpu = pick_gpu()
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map={"": gpu})
    model.eval()
    return tok, model


def seed_everything():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
