# Kazakh Vocabulary Extension Audit: README plan (through benchmarking)

Working title: **What does extending Qwen3.5's vocabulary for Kazakh cost? A tokenizer and evaluation audit of ISSAI's Qwen3.5-4B-Base-Kazakh**

Use this file as the skeleton of the repo README. Fill each TODO with your own measured results and your own wording. Do not claim anything that a table in the repo does not show.

Legend: **[DONE]** you have the result. **[TODO]** to run. **[CHECK]** verify before writing.

---

## 1. Summary (write last, 5 sentences)
- Question asked.
- What was compared (two models, same data and settings).
- Main numbers (from the tables below).
- One surprise or negative result.
- What you could not test.

## 2. Motivation and question
Open Kazakh models spend many tokens per word. ISSAI extended Qwen3.5's vocabulary by 16,000 Kazakh tokens (model card). Rakhat Meiramov (ISSAI) noted that benchmarks look similar to the original but the adaptation may still have hurt the model. Position papers argue vocabulary expansion is "ad hoc" and that tokens-per-word is a weak proxy (Alqahtani et al., 2026).

**Main question:** does cutting tokens per word keep what the original model could do, and where does it lose something?

Sub-questions:
1. How much cheaper is Kazakh text, and what happens to Russian and English text? (tokenizer audit)
2. Does the adapted model predict Kazakh, Russian and English text as well as the original? (loss per byte)
3. Do the splits follow Kazakh stems and endings? (linguistic alignment)
4. Does the adaptation change what the model knows about Kazakh heritage topics? (small hand-verified question set)
5. Do standard benchmarks agree with the intrinsic results? (benchmarks)
6. How does ISSAI's design (adding only Kazakh tokens to Qwen) compare with Sherkala's design (MBZUAI, Inception and Cerebras: adding Kazakh, Russian and Turkish tokens to LLaMA-3.1)? Can I reproduce the fertility table from the Sherkala paper on my own text? (reference comparison, section 6.7)

## 3. Hypotheses (state before running; negative results are fine)
- **H1:** the adapted tokenizer cuts Kazakh tokens by about half. **[DONE: supported, see 6.1]**
- **H2:** the adapted tokenizer makes Russian text more expensive. **[DONE on one text; replicate]**
- **H3:** the adapted model has higher loss per byte on Russian than the original. **[TODO]**
- **H4:** the adapted model has lower loss per byte on Kazakh, and English is not worse. **[TODO]**
- **H5:** benchmark scores differ little while intrinsic measures differ more. **[TODO]**
- **H6:** an extension that also adds Russian pieces (Sherkala's design) does not make Russian text more expensive, unlike the Kazakh-only extension (ISSAI). Test as relative change on my own text, not by comparing absolute numbers across different base tokenizers. **[TODO, needs gated access]**

## 4. Models and tokenizers
| Name | Hugging Face id | Vocabulary | Notes |
|---|---|---|---|
| Original | `Qwen/Qwen3.5-4B-Base` | 248,077 tokens (embedding rows padded to 248,320) | Apache-2.0 |
| ISSAI adapted | `issai/Qwen3.5-4B-Base-Kazakh` | 264,077 tokens (rows padded to 264,192) | Apache-2.0; from ISSAI card: 16,000 added tokens, new embeddings initialized by Fast Vocabulary Transfer, continued pretraining on ~15.7B tokens split 50/50 Kazakh and English, 1 epoch |
| LLaMA-3.1 (reference) | `meta-llama/Llama-3.1-8B` | 128,256 (per Sherkala paper) | gated; Sherkala's starting point |
| Sherkala (reference) | `inceptionai/Llama-3.1-Sherkala-8B-Chat` | 159,766 (per paper; the page says +25% Kazakh tokens) | gated; cc-by-nc-sa-4.0 (non-commercial); the page confirmed for the Chat model, so **[CHECK]** whether a base repo exists and whether its tokenizer is the same |
Optional extra tokenizer: SozKZ (`stukenov/sozkz-core-gpt2-50k-kk-base-v1`, Kazakh-only 50K byte-level BPE; unverified; list only what you actually ran).
Base models only for the Qwen comparison (not instruct). Sherkala is compared as a **tokenizer** (and, if memory allows, as a model; see 6.7). **[CHECK]** `tie_word_embeddings`, parameter count, and the 427 vs. 426 tensor difference seen at load time.

Known limitation to state: the training data of both models is not fully disclosed, so held-out text may have been seen in training. ISSAI's card lists FineWeb-2, CulturaX, FinePDFs and FineWiki as Kazakh sources, so those are not used as test text here.

## 5. Data
| Set | Source | Size | Native or translated | Use |
|---|---|---|---|---|
| Belebele passages | `facebook/belebele` (kaz_Cyrl, rus_Cyrl, eng_Latn), CC BY-SA 4.0 | 300 aligned passages | **Translated** (FLORES) | tokenizer audit, loss per byte |
| Alash held-out slice | your own books | TODO: number of books, tokens | **Native** | tokenizer audit, loss per byte (only if rights and server policy allow) |
| Second Russian text | TODO (e.g. Russian questions from KazMMLU) | TODO | native | replicate the Russian result |
| Hand-verified questions | written from your books | TODO (aim for 30) | native | knowledge check |
| Belebele Kazakh/Russian | same dataset | 900 | translated | benchmark |
| KazMMLU | `MBZUAI/KazMMLU`, CC BY-NC 4.0 | TODO subset | native | benchmark |
| English MMLU subset | TODO id and subjects | TODO | native | English retention |

Rules: no copyrighted book text in the repo; share statistics and tiny samples only; record where each file came from; confirm who owns the digitized editions and whether the books may sit on the server.

## 6. Experiments

### 6.1 Tokenizer audit **[DONE for two tokenizers]**
Metrics: tokens per word (whitespace words), tokens per 1,000 characters, Kazakh-to-English parity on parallel passages.

Result (300 Belebele passages, Oct 25, 2026):

| Tokenizer | Lang | Tokens | Tokens/word | Tokens/1k chars | KK/EN parity |
|---|---|---|---|---|---|
| Original | kk | 67,604 | 3.48 | 444.5 | 2.29 |
| Original | ru | 40,513 | 1.85 | 251.6 | 2.29 |
| Original | en | 29,474 | 1.27 | 208.5 | 2.29 |
| ISSAI | kk | 34,865 | 1.80 | 229.2 | 1.18 |
| ISSAI | ru | 48,385 | 2.21 | 300.5 | 1.18 |
| ISSAI | en | 29,474 | 1.27 | 208.5 | 1.18 |

Reading: Kazakh tokens fell about 48%; English is unchanged; Russian rose about 19% (Russian-to-English ratio 1.37 to 1.64). Matches Rakhat's recalled figures (3.6 and 1.8).

TODO: repeat on Alash text and a second Russian text; add optional tokenizers; report with bootstrap confidence intervals over passages.

### 6.1b Reference: reproduce the Sherkala fertility table **[TODO]**
- Run LLaMA-3.1 and Sherkala tokenizers on the **same** 300 passages (kk, ru, en) as above.
- The Sherkala paper (Table 1) reports tokens per word: Kazakh 4.73 to 2.04 (-56.8%), Russian 2.56 to 2.21 (-13.8%), Turkish 2.23 to 1.82 (-18.4%) on its own held-out text, with vocabulary 128,256 to 159,766.
- Compare **relative changes** (percent) with ours, not absolute values, because texts and base tokenizers differ.
- Side-by-side table of the two extension designs:

| | Qwen3.5 to ISSAI | LLaMA-3.1 to Sherkala |
|---|---|---|
| Base vocabulary | 248,077 | 128,256 |
| Tokens added | 16,000, predominantly Cyrillic Kazakh (model card) | about 31,500 from separately trained Kazakh, Russian and Turkish BPE tokenizers (paper) |
| Kazakh change | -48% (measured) | TODO (measured) vs. -56.8% (paper) |
| Russian change | +19% (measured) | TODO (measured) vs. -13.8% (paper) |
| English change | 0% (measured) | TODO |

### 6.2 Why did Russian get more expensive? **[TODO]**
- Print the token splits of 20 Russian words under both tokenizers; count how many words are split into more pieces.
- Check how many of the 16,000 new tokens appear inside Russian splits.
- State only what the examples show. Do not claim a mechanism without evidence.

### 6.3 Stem and ending alignment **[TODO]**
- 100 Kazakh words you hand-segment into stem + endings.
- Metric: share of true boundaries that coincide with token boundaries; average pieces per word.
- Report for both tokenizers. Explain that low fertility does not by itself mean the splits are linguistically meaningful (Alqahtani et al., 2026).

### 6.4 Loss per byte **[TODO, core experiment]**
- For each model and each text set: total negative log-likelihood (nats) over all tokens, divided by total UTF-8 bytes and by ln 2 (bits per byte).
- Same text, one model at a time, bfloat16, fixed context length, no padding effects (document method).
- Report per language with bootstrap confidence intervals over passages.
- This is tokenizer-independent, so the two models compare fairly.
- Tests H3 and H4.

### 6.5 Hand-verified knowledge check **[TODO]**
- 30 questions about Alash and history, each with the answer and source page from your books.
- Few-shot prompt format for a base model, e.g. `Сұрақ: ...\nЖауап:`; same prompt, same greedy decoding, same word budget for both models.
- Mark each answer correct / wrong / invented by hand; report counts, not just percentages.
- Label it a pilot. Note that you are not a neutral grader (write the answers before running the models).

### 6.6 Benchmarks **[TODO, last]**
- Belebele Kazakh (900) and Russian; KazMMLU subset (Kazakh and Russian subjects); English MMLU subset.
- Evaluation harness or your own likelihood-scoring script; zero-shot; identical settings for both models. **[CHECK]** the harness supports Qwen3.5.
- Report accuracy with sample size and a confidence interval.
- Compare with the intrinsic results from 6.1 to 6.4: do they agree? If benchmarks look similar but loss per byte differs, say so (this is the point of the project).

### 6.7 Sherkala as a model (optional, memory permitting) **[TODO]**
- Sherkala is an 8B model: about 16 GB in bfloat16, which does not fit in the ~16 GB free on one shared A100 with room to run. Options: 8-bit or 4-bit loading (state that this changes the numbers slightly), a GPU with more free memory, or skip and keep Sherkala as a tokenizer-only reference.
- If run, use bits per byte (tokenizer-independent), but the models differ in size, base and data, so describe it as a reference point, not a fair head-to-head.
- Why it matters: Sherkala is the published Kazakh vocabulary-extension case study from MBZUAI and collaborators; this project applies its fertility measure to a newer Kazakh adaptation (ISSAI's) and checks the cost it did not report for Qwen.

## 7. Results (tables and plots)
- Table 1: tokenizer audit (6.1).
- Table 1b: Sherkala reproduction and design comparison (6.1b).
- Table 2: Russian splits examples (6.2).
- Table 3: boundary alignment (6.3).
- Table 4: loss per byte, per language, with intervals (6.4).
- Table 5: knowledge check counts (6.5).
- Table 6: benchmarks (6.6).
- Figure: tokens per word vs. bits per byte per language, one plot.

## 8. Discussion (your own words)
- Which hypotheses held, which did not.
- What the benchmarks missed, if anything.
- What a vocabulary extension should check before release (a short list, e.g. Russian and English retention).

## 9. Limitations (be specific)
- Belebele Kazakh is translated, not native.
- 300 passages and whitespace words; small samples.
- Training data of both models only partly known; possible overlap with test text.
- Base models only; one model size (4B).
- Hand-checked sets are small and graded by the author.
- No claim about causes unless tested.
- Results may not transfer to other languages or models.

## 10. Reproducibility
- `requirements.txt` (pinned), seeds, one command per table (`make audit`, `make bpb`, ...).
- Hardware used (e.g. one 16 GB share of an A100 40 GB) and runtimes.
- Date of the run and library versions (transformers, torch).
- Data fetch instructions in `data/README.md`, with licenses.


## 12. Repo layout
```
kazakh-vocab-audit/
  README.md
  requirements.txt
  Makefile
  data/README.md          sources, licenses, fetch commands (no book text)
  scripts/audit.py
  scripts/russian_splits.py
  scripts/stem_check.py
  scripts/bits_per_byte.py
  scripts/knowledge_check.py
  scripts/benchmarks.sh
  results/*.csv
  figures/*.png
  notes/limitations.md
```

## 13. Schedule (adjust to your time)
| Day | Work | Output |
|---|---|---|
| 1 (done) | setup, load both models, tokenizer audit v1 | Table 1 |
| 2 | accept gated terms (LLaMA-3.1, Sherkala), Sherkala fertility reproduction, Russian splits, second Russian text, Alash text audit, stem check | Tables 1b, 2, 3 |
| 3 | loss-per-byte script, run both models on all text sets | Table 4 |
| 4 | hand-verified questions written and run | Table 5 |
| 5 | benchmarks (Belebele first, then KazMMLU, then English) | Table 6 |
| 6 | analysis, figure, discussion, limitations | draft README |
| 7 | reproduce from scratch, fix, send results to Rakhat, ask permission, publish | final repo |

## 14. Final checklist before publishing
- [ ] Every number in the README comes from a CSV in `results/`.
- [ ] Hypotheses compared with results honestly, including misses.
- [ ] Limitations section is specific.
- [ ] No book text, tokens or secrets in the repo.
- [ ] ISSAI permission received.
- [ ] Fresh-clone reproduction works.
- [ ] Shared-node files deleted.
- [ ] The README says "reproduces and extends," not "new method."