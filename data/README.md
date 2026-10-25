# Data

Nothing here needs a manual download; the scripts fetch everything into `hf/` (git-ignored)
on first run. Set nothing yourself: `scripts/common.py` points `HF_HOME` at `./hf`.

| Set | Source | License | Size used | Native / translated | Used in |
|---|---|---|---|---|---|
| Belebele passages | `facebook/belebele` (`kaz_Cyrl`, `rus_Cyrl`, `eng_Latn`), test split | CC BY-SA 4.0 | first 300 unique FLORES passages, aligned across languages (`belebele_passages.json`) | translated (FLORES) | tokenizer audit, bits per byte |
| Belebele questions | same | CC BY-SA 4.0 | 900 per language | translated | benchmark |
| KazMMLU text | `MBZUAI/KazMMLU`, test split | CC BY-NC 4.0 | 300 items per language, `random_state=0`, from Biology, Chemistry, Geography, Kazakh History, Law, Physics, World History; question + options as one document | native | tokenizer audit, bits per byte |
| KazMMLU | same | CC BY-NC 4.0 | all 22,889 test items (9,870 Kazakh track, 13,019 Russian track) | native | benchmark |
| MMLU | `cais/mmlu`, `all`, test split | MIT | all 14,042 items | native English | benchmark (English retention) |
| Kazakh segmentations | `kk_segmentation.tsv`, written for this repo | CC BY 4.0 | 110 words | — | stem alignment |

Note: KazMMLU text is used both as running text (bits per byte) and as a benchmark, and the
benchmark score could be influenced by whether either model saw these items in training.

## Private inputs (git-ignored, never committed)

`data/private/` is ignored by git. The scripts pick these up only if present:

- `data/private/alash/*.txt`: plain-text slices of the author's Alash-era books, one document
  per file. Used as a native Kazakh text set in the tokenizer audit and bits per byte.
  Only statistics are published; check rights before placing books on a shared server.
- `data/private/knowledge_questions.tsv`: hand-written questions for the knowledge check;
  format in `knowledge_questions.example.tsv`.

ISSAI's model card lists FineWeb-2, CulturaX, FinePDFs and FineWiki as Kazakh training sources,
so text from those is not used as test text.
