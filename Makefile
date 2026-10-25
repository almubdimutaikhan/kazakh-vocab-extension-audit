# One command per table. Model runs are one model per process, one at a time (shared GPUs).
PY      ?= ../.venv/bin/python
MODELS  := qwen3.5-orig qwen3.5-issai
RUN      = cd scripts && $(PY)

.PHONY: all audit splits stems bpb bench knowledge figure clean-results

all: audit splits stems bpb bench figure

audit:      ## Figures 1, 2
	$(RUN) audit.py

splits:     ## Figure 3 + counterfactual tokenizer
	$(RUN) russian_splits.py

stems:      ## Figure 4
	$(RUN) stem_check.py

bpb:        ## Figure 5
	for m in $(MODELS); do $(RUN) bits_per_byte.py $$m || exit 1; done
	$(RUN) bits_per_byte.py --summarize

bench:      ## Figures 6, 7
	for m in $(MODELS); do $(RUN) benchmarks.py $$m || exit 1; done
	$(RUN) benchmarks.py --summarize

knowledge:  ## knowledge check (needs data/private/knowledge_questions.tsv)
	for m in $(MODELS); do $(RUN) knowledge_check.py $$m || exit 1; done

figure:     ## figures/*.png (all README charts)
	$(RUN) figure.py
