# Limitations and checks

## Checked
- Both Qwen models tie input and output embeddings (`tie_word_embeddings: true`). The ISSAI
  checkpoint stores an explicit `lm_head.weight`; it is bit-identical to `embed_tokens`, so
  tied loading does not change the model. That tensor is the 427th at load time (426 for the
  original). The original checkpoint also contains a vision tower and an MTP head, which the
  causal-LM class does not load.
- Parameters as loaded: 4,205,751,296 (original) vs. 4,246,383,616 (ISSAI); the difference is
  the 15,872 extra embedding rows (264,192 - 248,320) x 2,560.
- Neither tokenizer has a BOS token; `<|endoftext|>` (id 248044) is shared and used as the
  scoring prefix. The answer letters " A".." E" are the same single ids in both tokenizers.
- The ISSAI tokenizer has the same BPE vocabulary, merges, normalizer and pre-tokenizer as the
  original; all 16,000 Kazakh pieces are stored as non-special *added tokens*.
- `inceptionai/Llama-3.1-Sherkala-8B-Chat` redirects to `inception42/...`. Only the Chat model
  exists; its tokenizer is used as the Sherkala reference. No base Sherkala repo was found.
- The fast linear-attention kernels (`flash-linear-attention`, `causal-conv1d`) were not
  installed; transformers used its reference PyTorch path, which is slower but exact.

## Limits of the evidence
- Belebele is translated (FLORES), not native Kazakh or Russian text.
- 300 documents per text set; whitespace words; 110 segmented words. Small samples.
- KazMMLU questions are short exam items, used both as running text and as a benchmark.
- Training data of both models is only partly disclosed; test text may overlap training data.
- Base models only, one size (4B). Results may not transfer to other languages or models.
- Benchmarks are zero-shot letter scoring with one prompt format. Other formats (few-shot,
  cloze scoring) can rank models differently.
- The segmentation file was drafted for this repo and covers frequent words, which are the
  most likely to become whole tokens; the alignment numbers are biased toward that case.
- The counterfactual tokenizer (Kazakh-letter tokens only) is a tokenizer-only result: the
  ISSAI model was trained with the full tokenizer, so we do not know how it would perform
  with the counterfactual one.
- Sherkala is compared as a tokenizer only (the 8B model did not fit next to other users'
  jobs on the shared GPU). Its numbers in the paper come from different held-out text.
- The knowledge check (section 6.5) has not been run: it needs questions written from the
  author's books.
