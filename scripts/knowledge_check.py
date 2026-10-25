"""Table 5 (section 6.5): pilot knowledge check on hand-verified Alash / history questions.

    python knowledge_check.py qwen3.5-orig    # writes results/knowledge_answers_<model>.csv
    python knowledge_check.py --summarize     # counts of correct / wrong / invented

Input: data/private/knowledge_questions.tsv (git-ignored; written by the author from the books,
answers fixed BEFORE running the models). Columns: id, question, answer, source.
See data/knowledge_questions.example.tsv for the format.
After running, fill the `grade` column by hand (correct / wrong / invented) in
results/knowledge_grades.csv; --summarize counts it.
"""
from common import MODELS, PRIVATE, RESULTS, DATA, load_model, seed_everything  # first: sets HF_HOME
import sys
import pandas as pd

QUESTIONS = PRIVATE / "knowledge_questions.tsv"
MAX_NEW_TOKENS = 48
# Few-shot frame for a base model; general-knowledge shots unrelated to the test questions
SHOTS = [
    ("Қазақстанның астанасы қай қала?", "Астана."),
    ("Абай Құнанбайұлы қай жылы туған?", "1845 жылы."),
]


def build_prompt(question):
    lines = [f"Сұрақ: {q}\nЖауап: {a}\n" for q, a in SHOTS]
    lines.append(f"Сұрақ: {question.strip()}\nЖауап:")
    return "\n".join(lines)


def run(name):
    import torch
    if not QUESTIONS.exists():
        raise SystemExit(f"{QUESTIONS} not found; see {DATA / 'knowledge_questions.example.tsv'}")
    seed_everything()
    qs = pd.read_csv(QUESTIONS, sep="\t")
    tok, model = load_model(name)
    out = []
    for _, r in qs.iterrows():
        x = tok(build_prompt(r.question), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        text = tok.decode(y[0, x.input_ids.shape[1]:], skip_special_tokens=True)
        out.append({"id": r.id, "model": name, "answer_model": text.split("\nСұрақ:")[0].strip()})
    pd.DataFrame(out).to_csv(RESULTS / f"knowledge_answers_{name}.csv", index=False)


def summarize():
    grades = pd.read_csv(RESULTS / "knowledge_grades.csv")
    table = grades.groupby(["model", "grade"]).size().unstack(fill_value=0)
    table.to_csv(RESULTS / "knowledge_check.csv")
    print(table)


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        summarize()
    else:
        run(sys.argv[1])
