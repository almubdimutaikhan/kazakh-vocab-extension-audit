"""Table 2 (section 6.2): why does Russian text get more expensive under the ISSAI tokenizer?

1. Structural diff of the two tokenizer.json files (vocab, merges, added tokens).
2. Word-level comparison of Russian splits; share of words / tokens that involve new tokens.
3. Counterfactual: drop the added tokens that contain no Kazakh-specific letter and re-count.
"""
from common import MODELS, REFERENCE_TOKENIZERS, RESULTS, text_sets, bootstrap_ci  # first: sets HF_HOME
import json
from collections import Counter
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

KAZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
N_SHOW = 20

raw = {k: json.loads(open(hf_hub_download(MODELS[k], "tokenizer.json")).read()) for k in MODELS}
A, B = raw["qwen3.5-orig"], raw["qwen3.5-issai"]


def merges(t):
    return [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in t["model"]["merges"]]


orig_added = {t["content"] for t in A["added_tokens"]}
new_added = [t for t in B["added_tokens"] if t["content"] not in orig_added and not t["special"]]
no_kaz = [t["content"] for t in new_added if not any(c in KAZ_LETTERS for c in t["content"])]
structure = {
    "same_bpe_vocab": A["model"]["vocab"] == B["model"]["vocab"],
    "same_bpe_merges": merges(A) == merges(B),
    "same_pre_tokenizer": A["pre_tokenizer"] == B["pre_tokenizer"],
    "same_normalizer": A["normalizer"] == B["normalizer"],
    "new_added_tokens": len(new_added),
    "new_added_normalized": sum(t["normalized"] for t in new_added),
    "new_added_single_word": sum(t["single_word"] for t in new_added),
    "new_added_without_kazakh_letter": len(no_kaz),
    "new_added_with_leading_space": sum(t["content"].startswith(" ") for t in new_added),
}
print(json.dumps(structure, indent=1, ensure_ascii=False))
pd.Series(structure).to_csv(RESULTS / "tokenizer_structure.csv", header=["value"])

# How each extension was stored, for every tokenizer we can read (Sherkala is gated)
how = []
for name, repo in {**MODELS, **REFERENCE_TOKENIZERS}.items():
    try:
        t = json.loads(open(hf_hub_download(repo, "tokenizer.json")).read())
    except Exception:
        continue
    how.append({"tokenizer": name, "model": t["model"]["type"], "bpe_vocab": len(t["model"]["vocab"]),
                "bpe_merges": len(t["model"]["merges"]),
                "non_special_added_tokens": sum(not a["special"] for a in t["added_tokens"])})
pd.DataFrame(how).to_csv(RESULTS / "extension_structure.csv", index=False)
print(pd.DataFrame(how).to_string(index=False))
pd.Series(sorted(no_kaz, key=len)).to_csv(RESULTS / "added_tokens_without_kazakh_letter.csv",
                                          index=False, header=["token"])

tok = {k: AutoTokenizer.from_pretrained(r) for k, r in MODELS.items()}
first_new_id = min(t["id"] for t in new_added)

# --- 2. word level: each whitespace word tokenized with a leading space, as it appears mid-sentence
sets = text_sets()
words = Counter(w for t in sets["belebele"]["ru"] for w in t.split())
rows = []
for w, freq in words.items():
    pa = tok["qwen3.5-orig"].tokenize(" " + w)
    ib = tok["qwen3.5-issai"](" " + w, add_special_tokens=False)["input_ids"]
    rows.append({"word": w, "freq": freq, "orig_pieces": len(pa), "issai_pieces": len(ib),
                 "uses_new_token": any(i >= first_new_id for i in ib),
                 "orig_split": " | ".join(tok["qwen3.5-orig"].convert_tokens_to_string([p]) for p in pa),
                 "issai_split": " | ".join(tok["qwen3.5-issai"].decode([i]) for i in ib)})
wd = pd.DataFrame(rows)
wd["delta"] = wd.issai_pieces - wd.orig_pieces
summary = {
    "unique_words": len(wd),
    "word_tokens_in_text": int(wd.freq.sum()),
    "share_words_more_pieces_by_freq": round((wd.freq * (wd.delta > 0)).sum() / wd.freq.sum(), 3),
    "share_words_fewer_pieces_by_freq": round((wd.freq * (wd.delta < 0)).sum() / wd.freq.sum(), 3),
    "share_words_using_new_token_by_freq": round((wd.freq * wd.uses_new_token).sum() / wd.freq.sum(), 3),
    "more_pieces_and_uses_new_token": int(((wd.delta > 0) & wd.uses_new_token).sum()),
    "more_pieces_total": int((wd.delta > 0).sum()),
}
# share of running-text tokens that are new ids
ids = [i for t in sets["belebele"]["ru"] for i in tok["qwen3.5-issai"](t, add_special_tokens=False)["input_ids"]]
summary["share_ru_tokens_that_are_new"] = round(sum(i >= first_new_id for i in ids) / len(ids), 3)
print(json.dumps(summary, indent=1))
pd.Series(summary).to_csv(RESULTS / "russian_splits_summary.csv", header=["value"])
wd.sort_values(["delta", "freq"], ascending=False).to_csv(RESULTS / "russian_words_all.csv", index=False)
table2 = wd[wd.delta > 0].sort_values("freq", ascending=False).head(N_SHOW)
table2[["word", "freq", "orig_split", "issai_split", "orig_pieces", "issai_pieces"]].to_csv(
    RESULTS / "russian_splits.csv", index=False)
print(table2[["word", "freq", "orig_split", "issai_split"]].to_string(index=False))

# --- 3. counterfactual tokenizer: keep only added tokens containing a Kazakh-specific letter
def without(drop):
    t = json.loads(json.dumps(B))
    t["added_tokens"] = [a for a in t["added_tokens"] if a["content"] not in drop]
    return Tokenizer.from_str(json.dumps(t))


variants = {
    "qwen3.5-orig": Tokenizer.from_str(json.dumps(A)),
    "qwen3.5-issai": Tokenizer.from_str(json.dumps(B)),
    # drop every added token without a Kazakh-specific letter
    "issai-kazakh-letter-only": without(set(no_kaz)),
    # finer: drop only those that can fire mid-word (no leading space), keep whole words like " болады"
    "issai-no-midword-fragments": without({t for t in no_kaz if not t.startswith(" ")}),
}
out = []
for set_name, langs in sets.items():
    for lang, texts in langs.items():
        base = [len(variants["qwen3.5-orig"].encode(t, add_special_tokens=False).ids) for t in texts]
        for name, t in variants.items():
            n = [len(t.encode(x, add_special_tokens=False).ids) for x in texts]
            lo, hi = bootstrap_ci(n, base)
            out.append({"tokenizer": name, "set": set_name, "lang": lang, "tokens": sum(n),
                        "pct_vs_orig": round(100 * (sum(n) / sum(base) - 1), 1),
                        "ci_low": round(100 * (lo - 1), 1), "ci_high": round(100 * (hi - 1), 1)})
out = pd.DataFrame(out)
out.to_csv(RESULTS / "russian_counterfactual.csv", index=False)
print(out.to_string(index=False))
