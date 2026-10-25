# One command per table. Model runs are one model per process, one at a time (shared GPUs).
PY      ?= ../.venv/bin/python
MODELS  := qwen3.5-orig qwen3.5-issai
RUN      = cd scripts && $(PY)

.PHONY: all audit splits stems bpb bench knowledge figure clean-results

all: audit splits stems bpb bench figure

audit:      ## Tables 1, 1b
	$(RUN) audit.py

splits:     ## Table 2 + counterfactual tokenizer
	$(RUN) russian_splits.py

stems:      ## Table 3
	$(RUN) stem_check.py

bpb:        ## Table 4
	for m in $(MODELS); do $(RUN) bits_per_byte.py $$m || exit 1; done
	$(RUN) bits_per_byte.py --summarize

bench:      ## Table 6
	for m in $(MODELS); do $(RUN) benchmarks.py $$m || exit 1; done
	$(RUN) benchmarks.py --summarize

knowledge:  ## Table 5 (needs data/private/knowledge_questions.tsv)
	for m in $(MODELS); do $(RUN) knowledge_check.py $$m || exit 1; done

figure:     ## figures/tokens_vs_bpb.png
	$(RUN) figure.py
