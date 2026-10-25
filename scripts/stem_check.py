"""Table 3 (section 6.3): do token boundaries fall on Kazakh morpheme boundaries?

Input: data/kk_segmentation.tsv, one surface segmentation per line (stem+ending+ending).
Each word is tokenized with a leading space (as it appears mid-sentence); the space is ignored.
Recall    = share of true morpheme boundaries that are also token boundaries.
Precision = share of token boundaries (inside the word) that are true morpheme boundaries.
"""
from common import MODELS, REFERENCE_TOKENIZERS, DATA, RESULTS  # first: sets HF_HOME
import pandas as pd
from transformers import AutoTokenizer

seg = pd.read_csv(DATA / "kk_segmentation.tsv", sep="\t")
seg["word"] = seg.segmentation.str.replace("+", "", regex=False)


def gold_boundaries(s):
    cuts, pos = set(), 0
    for m in s.split("+")[:-1]:
        pos += len(m)
        cuts.add(pos)
    return cuts


def token_boundaries(tok, word):
    enc = tok(" " + word, add_special_tokens=False, return_offsets_mapping=True)
    # character offsets inside the word (shift by the leading space); drop the word edges
    ends = {e - 1 for _, e in enc["offset_mapping"]}
    return {c for c in ends if 0 < c < len(word)}, len(enc["input_ids"])


rows, detail = [], []
for name, repo in {**MODELS, **REFERENCE_TOKENIZERS}.items():
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        print("SKIP", name, str(e).splitlines()[0][:100])
        continue
    tp = fp = fn = pieces = exact = stem_whole = 0
    for _, r in seg.iterrows():
        gold = gold_boundaries(r.segmentation)
        pred, n = token_boundaries(tok, r.word)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        pieces += n
        exact += pred == gold
        stem_end = min(gold)  # stem kept as one piece and cut exactly at its end
        stem_whole += stem_end in pred and not any(c < stem_end for c in pred)
        detail.append({"tokenizer": name, "segmentation": r.segmentation, "category": r.category,
                       "tokens": " | ".join(tok.decode([i]).strip() for i in
                                            tok(" " + r.word, add_special_tokens=False)["input_ids"]),
                       "recall_word": len(gold & pred) / len(gold)})
    rows.append({"tokenizer": name, "words": len(seg), "gold_boundaries": tp + fn,
                 "boundary_recall": round(tp / (tp + fn), 3),
                 "boundary_precision": round(tp / (tp + fp), 3) if tp + fp else None,
                 "pieces_per_word": round(pieces / len(seg), 2),
                 "exact_match": round(exact / len(seg), 3),
                 "stem_kept_whole": round(stem_whole / len(seg), 3)})

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "stem_alignment.csv", index=False)
pd.DataFrame(detail).to_csv(RESULTS / "stem_alignment_words.csv", index=False)
print(df.to_string(index=False))
