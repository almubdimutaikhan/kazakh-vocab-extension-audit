"""Table 6 (section 6.6): Belebele (kk/ru/en), KazMMLU (kk/ru tracks), English MMLU.

    python benchmarks.py qwen3.5-orig       # one model per process, writes per-item predictions
    python benchmarks.py --summarize        # accuracy, 95% CIs, paired ISSAI-minus-original difference

Zero-shot, identical prompts for both models. The prediction is the answer letter with the
highest next-token logit after the answer cue, restricted to the item's valid letters
(" A".." E" are the same single token ids in both tokenizers). Batch size 1, no padding.
"""
from common import MODELS, RESULTS, load_model, kazmmlu, belebele_passages, seed_everything, SEED  # first: sets HF_HOME
import sys
import time
import numpy as np
import pandas as pd

# language-matched labels for the prompt
CUES = {
    "kk": {"passage": "Мәтін", "question": "Сұрақ", "answer": "Жауап"},
    "ru": {"passage": "Текст", "question": "Вопрос", "answer": "Ответ"},
    "en": {"passage": "Passage", "question": "Question", "answer": "Answer"},
}
LETTERS = "ABCDE"


def prompt(lang, question, options, passage=None):
    c = CUES[lang]
    lines = [f"{c['passage']}: {passage.strip()}"] if passage else []
    lines.append(f"{c['question']}: {question.strip()}")
    lines += [f"{LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options)]
    lines.append(f"{c['answer']}:")
    return "\n".join(lines)


def items():
    """Yield dicts with benchmark, lang, subject, id, prompt, n_options, gold (index)."""
    from datasets import load_dataset
    for lang, cfg in {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}.items():
        for i, r in enumerate(load_dataset("facebook/belebele", cfg, split="test")):
            opts = [r[f"mc_answer{k}"] for k in range(1, 5)]
            yield {"benchmark": "belebele", "lang": lang, "subject": "reading", "id": i,
                   "prompt": prompt(lang, r["question"], opts, r["flores_passage"]),
                   "n_options": 4, "gold": int(r["correct_answer_num"]) - 1}
    km = kazmmlu(dtype=str)  # keep numeric options as written ("3", not "3.0")
    for _, r in km.iterrows():
        lang = {"Kazakh": "kk", "Russian": "ru"}[r["Language"]]
        opts = [r[f"Option {c}"] for c in LETTERS if not pd.isna(r[f"Option {c}"])]
        gold = LETTERS.index(str(r["Answer Key"]).strip())
        if gold >= len(opts):
            continue
        yield {"benchmark": "kazmmlu", "lang": lang, "subject": r["Subset"], "id": int(r["ID"]),
               "prompt": prompt(lang, str(r["Question"]), opts), "n_options": len(opts), "gold": gold}
    for i, r in enumerate(load_dataset("cais/mmlu", "all", split="test")):
        yield {"benchmark": "mmlu", "lang": "en", "subject": r["subject"], "id": i,
               "prompt": prompt("en", r["question"], r["choices"]), "n_options": 4, "gold": r["answer"]}


def run(name):
    import torch
    seed_everything()
    tok, model = load_model(name)
    letter_ids = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in LETTERS]
    assert all(len(i) == 1 for i in letter_ids), letter_ids
    letter_ids = [i[0] for i in letter_ids]
    rows, t0 = [], time.time()
    for k, it in enumerate(items()):
        x = torch.tensor([tok(it["prompt"], add_special_tokens=False)["input_ids"]], device=model.device)
        with torch.no_grad():
            logits = model(x).logits[0, -1].float()
        scores = logits[letter_ids[:it["n_options"]]]
        pred = int(scores.argmax())
        rows.append({**{k2: v for k2, v in it.items() if k2 != "prompt"}, "model": name, "pred": pred,
                     "correct": int(pred == it["gold"]), "prompt_tokens": x.shape[1]})
        if k % 2000 == 0:
            print(f"{name}: {k} items ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"bench_items_{name}.csv", index=False)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize():
    df = pd.concat([pd.read_csv(RESULTS / f"bench_items_{m}.csv") for m in MODELS])
    rows = []
    for (bench, lang), g in df.groupby(["benchmark", "lang"], sort=False):
        acc = {}
        for m in MODELS:
            d = g[g.model == m].sort_values(["subject", "id"])
            acc[m] = d.correct.to_numpy()
            lo, hi = wilson(acc[m].sum(), len(d))
            rows.append({"benchmark": bench, "lang": lang, "model": m, "n": len(d),
                         "accuracy": round(acc[m].mean(), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "chance": round(float((1 / d.n_options).mean()), 3)})
        a, b = acc["qwen3.5-orig"], acc["qwen3.5-issai"]
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(a), size=(2000, len(a)))
        diff = b[idx].mean(1) - a[idx].mean(1)
        rows.append({"benchmark": bench, "lang": lang, "model": "issai_minus_orig", "n": len(a),
                     "accuracy": round(b.mean() - a.mean(), 4),
                     "ci_low": round(float(np.percentile(diff, 2.5)), 4),
                     "ci_high": round(float(np.percentile(diff, 97.5)), 4),
                     "only_orig_correct": int(((a == 1) & (b == 0)).sum()),
                     "only_issai_correct": int(((a == 0) & (b == 1)).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "benchmarks.csv", index=False)
    print(out.to_string(index=False))

    # per-subject view for KazMMLU
    km = df[df.benchmark == "kazmmlu"].groupby(["subject", "model"]).correct.agg(["mean", "size"]).unstack("model")
    km.columns = [f"{a}_{b}" for a, b in km.columns]
    km["diff"] = km["mean_qwen3.5-issai"] - km["mean_qwen3.5-orig"]
    km.round(4).to_csv(RESULTS / "kazmmlu_by_subject.csv")


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        summarize()
    else:
        run(sys.argv[1])
