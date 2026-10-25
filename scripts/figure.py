"""Figure: tokens per word (tokenizer cost) vs. bits per byte (model quality), per language."""
from common import RESULTS, FIGURES  # first: sets HF_HOME
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"qwen3.5-orig": "#2a78d6", "qwen3.5-issai": "#eb6834"}  # categorical slots 1, 2
LABELS = {"qwen3.5-orig": "Qwen3.5-4B-Base (original)", "qwen3.5-issai": "ISSAI Kazakh-adapted"}
LANG = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
SET = {"belebele": "Belebele", "kazmmlu": "KazMMLU", "alash": "Alash"}

tok = pd.read_csv(RESULTS / "tokenizer_audit.csv")
bpb = pd.read_csv(RESULTS / "bits_per_byte.csv")
df = bpb[bpb.model.isin(COLORS)].merge(
    tok.rename(columns={"tokenizer": "model"})[["model", "set", "lang", "tokens_per_word"]],
    on=["model", "set", "lang"])

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=160, sharey=True)
fig.patch.set_facecolor("#fcfcfb")
for ax, lang in zip(axes, LANG):
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    d = df[df.lang == lang]
    for set_name, g in d.groupby("set"):
        o = g[g.model == "qwen3.5-orig"].iloc[0]
        i = g[g.model == "qwen3.5-issai"].iloc[0]
        ax.annotate("", xy=(i.tokens_per_word, i.bits_per_byte), xytext=(o.tokens_per_word, o.bits_per_byte),
                    arrowprops=dict(arrowstyle="->", color="#8a8984", lw=1, shrinkA=6, shrinkB=6))
        ax.annotate(SET[set_name], (o.tokens_per_word, o.bits_per_byte), textcoords="offset points",
                    xytext=(-9, 0), ha="right", va="center", fontsize=8, color="#52514e")
    for m in COLORS:
        g = d[d.model == m]
        ax.errorbar(g.tokens_per_word, g.bits_per_byte,
                    yerr=[g.bits_per_byte - g.ci_low, g.ci_high - g.bits_per_byte],
                    fmt="o", ms=7, color=COLORS[m], mec="#fcfcfb", mew=1.5, elinewidth=1.2, capsize=3,
                    label=LABELS[m], zorder=3)
    ax.set_title(LANG[lang], loc="left", fontsize=10.5, color="#0b0b0b")
    lo, hi = d.tokens_per_word.min(), d.tokens_per_word.max()
    pad = max(0.25, 0.15 * (hi - lo))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Tokens per word (lower = cheaper)")
axes[0].set_ylabel("Bits per UTF-8 byte\n(lower = better prediction)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2, fontsize=9)
fig.suptitle("Tokenizer cost vs. prediction quality, original → ISSAI on the same text",
             x=0.01, ha="left", fontsize=11.5, color="#0b0b0b")
fig.text(0.01, 0.01, "Arrows: original → ISSAI. Bars: 95% bootstrap CI over 300 documents. "
         "Compare bits per byte within a language only (Cyrillic uses 2 bytes per letter).",
         fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(FIGURES / "tokens_vs_bpb.png", facecolor=fig.get_facecolor())
print("wrote", FIGURES / "tokens_vs_bpb.png")
