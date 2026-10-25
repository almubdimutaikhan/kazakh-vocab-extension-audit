"""Tables 1 and 1b: tokens per word / per 1k chars on every text set, with bootstrap CIs.

Gated reference tokenizers (LLaMA-3.1, Sherkala, SozKZ) are skipped with a note if not accessible.
"""
from common import MODELS, REFERENCE_TOKENIZERS, RESULTS, text_sets, bootstrap_ci  # first: sets HF_HOME
import pandas as pd
from transformers import AutoTokenizer

# base -> extended pairs whose relative change we report (Table 1b)
PAIRS = [("qwen3.5-orig", "qwen3.5-issai"), ("llama3.1", "sherkala")]

sets = text_sets()
rows, per_doc, skipped = [], [], []
for name, repo in {**MODELS, **REFERENCE_TOKENIZERS}.items():
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        skipped.append({"tokenizer": name, "repo": repo, "reason": str(e).splitlines()[0][:200]})
        print("SKIP", name, "->", skipped[-1]["reason"])
        continue
    for set_name, langs in sets.items():
        for lang, texts in langs.items():
            n_tok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            n_words = [len(t.split()) for t in texts]
            n_chars = [len(t) for t in texts]
            for i, (a, b, c) in enumerate(zip(n_tok, n_words, n_chars)):
                per_doc.append({"tokenizer": name, "set": set_name, "lang": lang, "doc": i,
                                "tokens": a, "words": b, "chars": c})
            lo, hi = bootstrap_ci(n_tok, n_words)
            rows.append({
                "tokenizer": name, "set": set_name, "lang": lang, "vocab": len(tok),
                "docs": len(texts), "tokens": sum(n_tok), "words": sum(n_words), "chars": sum(n_chars),
                "tokens_per_word": round(sum(n_tok) / sum(n_words), 3),
                "tpw_ci_low": round(lo, 3), "tpw_ci_high": round(hi, 3),
                "tokens_per_1k_chars": round(1000 * sum(n_tok) / sum(n_chars), 1),
            })

df = pd.DataFrame(rows)
# Kazakh-to-English parity only exists on the parallel Belebele set
en = df[(df.set == "belebele") & (df.lang == "en")].set_index("tokenizer")["tokens"]
df["tokens_vs_en"] = df.apply(
    lambda r: round(r.tokens / en[r.tokenizer], 2) if r.set == "belebele" and r.tokenizer in en else None, axis=1)
df.to_csv(RESULTS / "tokenizer_audit.csv", index=False)
pd.DataFrame(per_doc).to_csv(RESULTS / "tokenizer_audit_per_doc.csv", index=False)
pd.DataFrame(skipped, columns=["tokenizer", "repo", "reason"]).to_csv(RESULTS / "tokenizer_skipped.csv", index=False)
print(df.to_string(index=False))

# Table 1b: relative change in token count, extended vs. its own base, same text
pd_doc = pd.DataFrame(per_doc)
change = []
for base, ext in PAIRS:
    if base not in set(df.tokenizer) or ext not in set(df.tokenizer):
        continue
    for (set_name, lang), g in pd_doc.groupby(["set", "lang"]):
        a = g[g.tokenizer == base].sort_values("doc")["tokens"].to_numpy()
        b = g[g.tokenizer == ext].sort_values("doc")["tokens"].to_numpy()
        lo, hi = bootstrap_ci(b, a)
        change.append({"base": base, "extended": ext, "set": set_name, "lang": lang,
                       "base_tokens": int(a.sum()), "extended_tokens": int(b.sum()),
                       "pct_change": round(100 * (b.sum() / a.sum() - 1), 1),
                       "pct_ci_low": round(100 * (lo - 1), 1), "pct_ci_high": round(100 * (hi - 1), 1)})
change = pd.DataFrame(change)
change.to_csv(RESULTS / "extension_change.csv", index=False)
print(change.to_string(index=False))
