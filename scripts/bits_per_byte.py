"""Table 4 (section 6.4): bits per byte of each model on every text set.

    python bits_per_byte.py qwen3.5-orig     # one model per process, writes per-document NLL
    python bits_per_byte.py --summarize      # Table 4 with bootstrap CIs and paired differences

Method: each document is scored on its own, batch size 1 (no padding), bfloat16 weights,
log-softmax in float32. The document is prefixed with <|endoftext|> (same id in both models,
neither has a BOS token) so every text token, including the first, is predicted.
Documents longer than CTX tokens are split into non-overlapping windows, each with the prefix.
bits per byte = sum NLL (nats) / ln 2 / UTF-8 bytes, so it does not depend on the tokenizer.
"""
from common import MODELS, RESULTS, text_sets, load_model, bootstrap_ci, seed_everything, SEED
import sys
import math
import time
import numpy as np
import pandas as pd

CTX = 2048


def score(tok, model, text):
    import torch
    ids = tok(text, add_special_tokens=False)["input_ids"]
    nll = 0.0
    for start in range(0, len(ids), CTX):
        window = [tok.eos_token_id] + ids[start:start + CTX]
        x = torch.tensor([window], device=model.device)
        with torch.no_grad():
            logits = model(x).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
    return nll, len(ids)


def run(name):
    seed_everything()
    tok, model = load_model(name)
    rows, t0 = [], time.time()
    for set_name, langs in text_sets().items():
        for lang, texts in langs.items():
            for i, text in enumerate(texts):
                nll, n_tok = score(tok, model, text)
                rows.append({"model": name, "set": set_name, "lang": lang, "doc": i, "nll_nats": nll,
                             "tokens": n_tok, "bytes": len(text.encode("utf-8"))})
            print(f"{name} {set_name}/{lang}: done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"bpb_per_doc_{name}.csv", index=False)


def summarize():
    df = pd.concat([pd.read_csv(RESULTS / f"bpb_per_doc_{m}.csv") for m in MODELS])
    rows = []
    for (set_name, lang), g in df.groupby(["set", "lang"], sort=False):
        per = {}
        for m in MODELS:
            d = g[g.model == m].sort_values("doc")
            bits = d.nll_nats.to_numpy() / math.log(2)
            per[m] = (bits, d.bytes.to_numpy(), d.tokens.to_numpy())
            lo, hi = bootstrap_ci(bits, d.bytes)
            rows.append({"set": set_name, "lang": lang, "model": m, "docs": len(d),
                         "bytes": int(d.bytes.sum()), "tokens": int(d.tokens.sum()),
                         "bits_per_byte": round(bits.sum() / d.bytes.sum(), 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "bits_per_token": round(bits.sum() / d.tokens.sum(), 3)})
        # paired difference (ISSAI minus original) resampling the same documents
        (a, nb, _), (b, _, _) = per["qwen3.5-orig"], per["qwen3.5-issai"]
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(a), size=(2000, len(a)))
        diff = (b[idx].sum(1) - a[idx].sum(1)) / nb[idx].sum(1)
        rows.append({"set": set_name, "lang": lang, "model": "issai_minus_orig", "docs": len(a),
                     "bits_per_byte": round((b.sum() - a.sum()) / nb.sum(), 4),
                     "ci_low": round(float(np.percentile(diff, 2.5)), 4),
                     "ci_high": round(float(np.percentile(diff, 97.5)), 4),
                     "pct_change": round(100 * (b.sum() / a.sum() - 1), 1),
                     "docs_issai_better": int((b < a).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "bits_per_byte.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        summarize()
    else:
        run(sys.argv[1])
