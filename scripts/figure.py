"""All README figures, drawn from results/*.csv (no model runs needed).

    python figure.py            # writes figures/*.png
"""
from common import RESULTS, FIGURES  # first: sets HF_HOME
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ---- style: one palette, fixed entity colors across every figure --------------------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
TOK = {  # tokenizer -> (label, color)
    "qwen3.5-orig": ("Qwen3.5 original", BLUE),
    "qwen3.5-issai": ("ISSAI (Qwen + Kazakh)", ORANGE),
    "llama3.1": ("LLaMA-3.1", AQUA),
    "sherkala": ("Sherkala (LLaMA + Kazakh/Russian/Turkish)", YELLOW),
}
LANG = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
SETS = {"belebele": "Belebele", "kazmmlu": "KazMMLU"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", FIGURES / name)


def title(fig, main, sub, y=1.02):
    fig.text(0.0, y, main, fontsize=13, fontweight="bold", ha="left", va="bottom", transform=fig.transFigure)
    fig.text(0.0, y - 0.035, sub, fontsize=9, color=INK2, ha="left", va="bottom", transform=fig.transFigure)


def r2(v):  # round half up, so 0.015 prints as 0.02 like the README says
    return f"{np.floor(v * 100 + 0.5 + 1e-9) / 100:.2f}"


def pct(v):
    return f"{v:+.1f}%" if round(v, 1) != 0 else "0.0%"


tok = pd.read_csv(RESULTS / "tokenizer_audit.csv")
ext = pd.read_csv(RESULTS / "extension_change.csv")
cf = pd.read_csv(RESULTS / "russian_counterfactual.csv")
stem = pd.read_csv(RESULTS / "stem_alignment.csv")
bpb = pd.read_csv(RESULTS / "bits_per_byte.csv")
bench = pd.read_csv(RESULTS / "benchmarks.csv")
subj = pd.read_csv(RESULTS / "kazmmlu_by_subject.csv")
splits = pd.read_csv(RESULTS / "russian_splits.csv")


# ---- 1. tokens per word -------------------------------------------------------------------
def fig_tokens_per_word():
    panels = [("belebele", "kk"), ("belebele", "ru"), ("belebele", "en"), ("kazmmlu", "kk"), ("kazmmlu", "ru")]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.9), sharey=True)
    xs = [0, 1, 2.4, 3.4]  # a gap between the Qwen pair and the LLaMA pair
    for ax, (s, l) in zip(axes, panels):
        ax.grid(axis="x", visible=False)
        for x, (t, (_, c)) in zip(xs, TOK.items()):
            r = tok[(tok.tokenizer == t) & (tok.set == s) & (tok.lang == l)].iloc[0]
            ax.bar(x, r.tokens_per_word, width=0.8, color=c)
            ax.errorbar(x, r.tokens_per_word, yerr=[[r.tokens_per_word - r.tpw_ci_low], [r.tpw_ci_high - r.tokens_per_word]],
                        color=INK2, lw=1, capsize=2.5)
            ax.text(x, r.tpw_ci_high + 0.1, f"{r.tokens_per_word:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_title(f"{LANG[l]}\n{SETS[s]}", fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_ylim(0, 5.7)
    axes[0].set_ylabel("Tokens per whitespace word\n(lower = cheaper)")
    handles = [Line2D([0], [0], marker="s", ls="", ms=9, color=c, label=n) for n, c in TOK.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    title(fig, "How many tokens does one word cost?",
          "Each bar = total tokens ÷ total whitespace-separated words over 300 texts. Whisker = 95% bootstrap CI over texts. "
          "Within each pair, the right bar is the extended tokenizer.")
    save(fig, "1_tokens_per_word.png")


# ---- 2. change vs own base ----------------------------------------------------------------
def fig_change():
    rows = [("kk", "belebele"), ("kk", "kazmmlu"), ("ru", "belebele"), ("ru", "kazmmlu")]
    paper = {("kk", "belebele"): -56.8, ("ru", "belebele"): -13.8}
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.grid(axis="y", visible=False)
    y0 = np.arange(len(rows))[::-1] * 1.0
    for off, (ext_name, col, lab) in zip((0.19, -0.19), [("qwen3.5-issai", ORANGE, "ISSAI vs. Qwen3.5"),
                                                         ("sherkala", YELLOW, "Sherkala vs. LLaMA-3.1")]):
        for y, (l, s) in zip(y0, rows):
            r = ext[(ext.extended == ext_name) & (ext.lang == l) & (ext.set == s)].iloc[0]
            ax.barh(y + off, r["pct_change"], height=0.32, color=col)
            ax.errorbar(r["pct_change"], y + off, xerr=[[r["pct_change"] - r.pct_ci_low], [r.pct_ci_high - r["pct_change"]]],
                        color=INK2, lw=1, capsize=2.5)
            side = 1 if r["pct_change"] >= 0 else -1
            edge = r.pct_ci_high if side > 0 else r.pct_ci_low
            ax.text(edge + side * 1.2, y + off, f"{r['pct_change']:+.1f}%", va="center",
                    ha="left" if side > 0 else "right", fontsize=9)
            if ext_name == "sherkala" and (l, s) in paper:
                ax.plot(paper[(l, s)], y + off, marker="D", ms=7, mfc=SURFACE, mec=INK, mew=1.4, zorder=5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y0)
    ax.set_yticklabels([f"{LANG[l]}\n{SETS[s]}" for l, s in rows])
    ax.set_xlim(-75, 32)
    ax.set_xlabel("Change in total token count vs. the tokenizer it was extended from")
    ax.text(-74, 3.75, "← fewer tokens (cheaper)", fontsize=9, color=INK2, va="bottom")
    ax.text(31, 3.75, "more tokens (costlier) →", fontsize=9, color=INK2, va="bottom", ha="right")
    ax.set_ylim(-0.6, 3.9)
    handles = [Line2D([0], [0], marker="s", ls="", ms=9, color=ORANGE, label="ISSAI vs. Qwen3.5 (adds Kazakh only)"),
               Line2D([0], [0], marker="s", ls="", ms=9, color=YELLOW, label="Sherkala vs. LLaMA-3.1 (adds Kazakh + Russian + Turkish)"),
               Line2D([0], [0], marker="D", ls="", ms=7, mfc=SURFACE, mec=INK, mew=1.4, label="Sherkala paper (their own text)")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.17))
    title(fig, "Kazakh gets cheaper in both designs; only ISSAI makes Russian costlier",
          "Same text counted with the base and the extended tokenizer. Whisker = 95% bootstrap CI over texts.")
    save(fig, "2_change_vs_base.png")


# ---- 3. why Russian gets costlier ---------------------------------------------------------
def fig_russian_words():
    pick = splits.assign(k=splits.word.str.lower()).sort_values("freq", ascending=False).drop_duplicates("k").head(8)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(pick) + 1)
    ax.axis("off")
    ax.text(0.2, len(pick) + 0.45, "Russian word (times in 300 passages)", fontsize=9, color=INK2, va="center")
    ax.text(3.6, len(pick) + 0.45, "Qwen3.5 original", fontsize=9, color=INK2, va="center")
    ax.text(6.0, len(pick) + 0.45, "ISSAI", fontsize=9, color=INK2, va="center")

    def chips(x, y, pieces, color):
        for p in pieces:
            p = p.strip()
            w = 0.26 + 0.2 * len(p)
            ax.add_patch(FancyBboxPatch((x, y - 0.3), w, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=color, ec=SURFACE, lw=2))
            ax.text(x + w / 2, y, p, ha="center", va="center", fontsize=10.5,
                    color="#ffffff" if color != YELLOW else INK)
            x += w + 0.05
    for i, r in enumerate(pick.itertuples()):
        y = len(pick) - i - 0.1
        ax.text(0.2, y, f"{r.word}  ({r.freq})", fontsize=11, va="center")
        chips(3.6, y, r.orig_split.split("|"), BLUE)
        chips(6.0, y, r.issai_split.split("|"), ORANGE)
    title(fig, "Why Russian costs more: ISSAI's new pieces cut common Russian words apart",
          "Each box is one token. Blue = how Qwen3.5 tokenizes the word; orange = how the ISSAI tokenizer does.")
    save(fig, "3a_russian_word_splits.png")


def fig_counterfactual():
    variants = [("qwen3.5-issai", "ISSAI\nas released"),
                ("issai-no-midword-fragments", "Drop 1,880\nmid-word fragments"),
                ("issai-kazakh-letter-only", "Drop all 5,531 tokens\nwithout a Kazakh letter")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, (l, ttl) in zip(axes, [("kk", "Kazakh text"), ("ru", "Russian text")]):
        ax.grid(axis="x", visible=False)
        for i, (v, _) in enumerate(variants):
            for j, (s, a) in enumerate([("belebele", 1.0), ("kazmmlu", 0.45)]):
                r = cf[(cf.tokenizer == v) & (cf.lang == l) & (cf.set == s)].iloc[0]
                x = i + (j - 0.5) * 0.4
                ax.bar(x, r.pct_vs_orig, width=0.36, color=ORANGE, alpha=a)
                ax.text(x, r.pct_vs_orig + (1.2 if r.pct_vs_orig >= 0 else -1.2), pct(r.pct_vs_orig),
                        ha="center", va="bottom" if r.pct_vs_orig >= 0 else "top", fontsize=8.5)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n for _, n in variants], fontsize=8)
        ax.set_title(ttl, loc="left", fontsize=10.5)
        ax.set_ylim(-58, 28)
    axes[0].set_ylabel("Change in token count vs. original Qwen3.5\n(below 0 = cheaper)")
    handles = [Line2D([0], [0], marker="s", ls="", ms=9, color=ORANGE, label="Belebele"),
               Line2D([0], [0], marker="s", ls="", ms=9, color=ORANGE, alpha=0.45, label="KazMMLU")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.12))
    title(fig, "Removing 12% of the added tokens removes almost all of the extra Russian cost",
          "ISSAI tokenizer rebuilt without some added tokens, then the same text recounted. Tokenizer-only test: no model was retrained.")
    save(fig, "3b_russian_counterfactual.png")


# ---- 4. stems -----------------------------------------------------------------------------
def fig_stems():
    metrics = [("pieces_per_word", "Pieces per word", "fewer = more whole-word tokens"),
               ("boundary_recall", "Boundary recall", "share of true morpheme\nboundaries that get a cut"),
               ("boundary_precision", "Boundary precision", "share of cuts that fall on\na true boundary"),
               ("stem_kept_whole", "Stem kept whole", "share of words where the stem is one\npiece, cut exactly at its end")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (m, name, expl) in zip(axes, metrics):
        ax.grid(axis="x", visible=False)
        for i, (t, (_, c)) in enumerate(TOK.items()):
            v = stem[stem.tokenizer == t][m].iloc[0]
            ax.bar(i, v, width=0.7, color=c)
            ax.text(i, v + (0.1 if m == "pieces_per_word" else 0.02), r2(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks([])
        ax.set_title(name, loc="left", fontsize=10.5)
        ax.text(0, -0.06, expl, transform=ax.transAxes, fontsize=8, color=INK2, va="top")
        ax.set_ylim(0, 6.6 if m == "pieces_per_word" else 0.8)
    handles = [Line2D([0], [0], marker="s", ls="", ms=9, color=c, label=n) for n, c in TOK.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.2))
    title(fig, "Fewer tokens per word does not mean tokens that follow Kazakh morphology",
          "110 frequent Kazakh words with hand-segmented stem + endings (194 true boundaries).")
    save(fig, "4_stem_alignment.png")


# ---- 5. bits per byte ---------------------------------------------------------------------
def fig_bpb():
    rows = [("kk", "belebele"), ("kk", "kazmmlu"), ("ru", "belebele"), ("ru", "kazmmlu"), ("en", "belebele")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(rows))[::-1]
    for y, (l, s) in zip(ys, rows):
        o = bpb[(bpb.model == "qwen3.5-orig") & (bpb.lang == l) & (bpb.set == s)].iloc[0]
        i = bpb[(bpb.model == "qwen3.5-issai") & (bpb.lang == l) & (bpb.set == s)].iloc[0]
        d = bpb[(bpb.model == "issai_minus_orig") & (bpb.lang == l) & (bpb.set == s)].iloc[0]
        ax.plot([o.bits_per_byte, i.bits_per_byte], [y, y], color="#c3c2b7", lw=2.5, zorder=1)
        for r, c in ((o, BLUE), (i, ORANGE)):
            ax.errorbar(r.bits_per_byte, y, xerr=[[r.bits_per_byte - r.ci_low], [r.ci_high - r.bits_per_byte]],
                        fmt="o", ms=9, color=c, mec=SURFACE, mew=2, ecolor=c, elinewidth=1.3, capsize=2.5, zorder=3)
        ax.text(1.17, y, f"{d['pct_change']:+.1f}%", fontsize=10.5, fontweight="bold", va="center", ha="right")
        ax.text(1.19, y, f"ISSAI better on {int(d.docs_issai_better)}/300 texts", fontsize=8.5, color=INK2, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{LANG[l]}\n{SETS[s]}" for l, s in rows])
    ax.set_xlim(0.4, 1.12)
    ax.set_xlabel("Bits per UTF-8 byte (lower = the model is less surprised by the text)")
    ax.text(1.17, len(rows) - 0.4, "change", fontsize=9, color=INK2, ha="right")
    handles = [Line2D([0], [0], marker="o", ls="", ms=9, color=BLUE, label="Qwen3.5-4B-Base"),
               Line2D([0], [0], marker="o", ls="", ms=9, color=ORANGE, label="ISSAI Kazakh-adapted")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.45, -0.09))
    title(fig, "The adapted model predicts text better in all three languages",
          "Tokenizer-independent: loss is divided by bytes, not tokens. Whisker = 95% bootstrap CI over 300 texts. "
          "Compare within one language only.")
    fig.subplots_adjust(right=0.72)
    save(fig, "5_bits_per_byte.png")


# ---- 6. benchmarks ------------------------------------------------------------------------
def fig_bench():
    rows = [("belebele", "kk", "Belebele Kazakh"), ("kazmmlu", "kk", "KazMMLU Kazakh track"),
            ("belebele", "ru", "Belebele Russian"), ("kazmmlu", "ru", "KazMMLU Russian track"),
            ("belebele", "en", "Belebele English"), ("mmlu", "en", "MMLU English")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]}, sharey=True)
    ys = np.arange(len(rows))[::-1]
    a1.grid(axis="y", visible=False)
    a2.grid(axis="y", visible=False)
    for y, (b, l, name) in zip(ys, rows):
        g = bench[(bench.benchmark == b) & (bench.lang == l)]
        o, i, d = (g[g.model == m].iloc[0] for m in ("qwen3.5-orig", "qwen3.5-issai", "issai_minus_orig"))
        for r, c, dy in ((o, BLUE, 0.13), (i, ORANGE, -0.13)):
            a1.errorbar(r.accuracy * 100, y + dy, xerr=[[(r.accuracy - r.ci_low) * 100], [(r.ci_high - r.accuracy) * 100]],
                        fmt="o", ms=9, color=c, mec=SURFACE, mew=2, ecolor=c, elinewidth=1.3, capsize=2.5, zorder=3)
        a1.plot([o.chance * 100] * 2, [y - 0.3, y + 0.3], color=MUTED, lw=1.5)
        sig = d.ci_low > 0 or d.ci_high < 0
        a2.errorbar(d.accuracy * 100, y, xerr=[[(d.accuracy - d.ci_low) * 100], [(d.ci_high - d.accuracy) * 100]],
                    fmt="o", ms=9, color=INK, mfc=INK if sig else SURFACE, mec=INK, mew=1.5, elinewidth=1.3, capsize=2.5, zorder=3)
        a2.text(8.2, y, f"{d.accuracy * 100:+.1f}", va="center", fontsize=10, fontweight="bold" if sig else "normal",
                color=INK if sig else INK2)
        a2.text(9.8, y, f"{int(d.only_orig_correct):,} / {int(d.only_issai_correct):,}", va="center", fontsize=8.5, color=INK2)
    a1.set_yticks(ys)
    a1.set_yticklabels([r[2] for r in rows])
    a1.set_xlim(15, 100)
    a1.set_xlabel("Accuracy, % (| = chance level)")
    a1.set_title("Accuracy", loc="left", fontsize=10.5)
    a2.axvline(0, color=INK2, lw=1)
    a2.set_xlim(-7, 7)
    a2.set_xlabel("ISSAI − original, percentage points")
    a2.set_title("Difference (paired, same questions)", loc="left", fontsize=10.5)
    a2.text(9.8, len(rows) - 0.45, "only orig. / only ISSAI\nanswered right", fontsize=8, color=INK2, va="bottom")
    handles = [Line2D([0], [0], marker="o", ls="", ms=9, color=BLUE, label="Qwen3.5-4B-Base"),
               Line2D([0], [0], marker="o", ls="", ms=9, color=ORANGE, label="ISSAI Kazakh-adapted"),
               Line2D([0], [0], marker="o", ls="", ms=9, color=INK, label="difference, CI excludes 0"),
               Line2D([0], [0], marker="o", ls="", ms=9, mfc=SURFACE, mec=INK, mew=1.5, label="difference, CI includes 0")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.45, -0.1))
    title(fig, "Kazakh questions improve; Russian and English slip slightly",
          "Zero-shot multiple choice. Whisker = 95% CI (Wilson for accuracy, paired bootstrap for the difference).")
    fig.subplots_adjust(right=0.82, wspace=0.08)
    save(fig, "6_benchmarks.png")


def fig_subjects():
    s = subj.copy()
    s["diff"] *= 100
    def short(m):
        lang = "kk" if m.group(3) == "kaz" else "ru"
        return f" ({lang})" if m.group(1) == "High School" else f" ({lang}, professional)"
    s["name"] = s.subject.str.replace(r" \((High School|Professional( & University)?) in (kaz|rus)\)", short, regex=True)
    s = s[s["size_qwen3.5-orig"] >= 100].sort_values("diff")
    fig, ax = plt.subplots(figsize=(8.5, 8.2))
    ax.grid(axis="y", visible=False)
    cols = [BLUE if v >= 0 else RED for v in s["diff"]]
    ax.barh(range(len(s)), s["diff"], color=cols, height=0.7)
    for k, v in enumerate(s["diff"]):
        ax.text(v + (0.25 if v >= 0 else -0.25), k, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["name"], fontsize=8.8)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-8, 16)
    ax.set_xlabel("ISSAI − original accuracy, percentage points")
    handles = [Line2D([0], [0], marker="s", ls="", ms=9, color=BLUE, label="ISSAI better"),
               Line2D([0], [0], marker="s", ls="", ms=9, color=RED, label="original better")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    title(fig, "KazMMLU by subject: gains follow Kazakh content, not only the Kazakh language",
          "(kk) = Kazakh track, (ru) = Russian track; school subjects unless marked professional. Subjects with ~300 questions "
          "have a 95% CI of about ±4 points, those with ~1,000 about ±2.", y=0.985)
    fig.subplots_adjust(top=0.92)
    save(fig, "7_kazmmlu_subjects.png")


# ---- 0. hero: cost vs quality -------------------------------------------------------------
def fig_hero():
    df = bpb[bpb.model.isin(["qwen3.5-orig", "qwen3.5-issai"])].merge(
        tok.rename(columns={"tokenizer": "model"})[["model", "set", "lang", "tokens_per_word"]], on=["model", "set", "lang"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, lang in zip(axes, LANG):
        d = df[df.lang == lang]
        for set_name, g in d.groupby("set"):
            o, i = g[g.model == "qwen3.5-orig"].iloc[0], g[g.model == "qwen3.5-issai"].iloc[0]
            ax.annotate("", xy=(i.tokens_per_word, i.bits_per_byte), xytext=(o.tokens_per_word, o.bits_per_byte),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2, shrinkA=7, shrinkB=7))
            ax.text(o.tokens_per_word - 0.08, o.bits_per_byte, SETS[set_name], ha="right", va="center", fontsize=8.5, color=INK2)
        for m, c in (("qwen3.5-orig", BLUE), ("qwen3.5-issai", ORANGE)):
            g = d[d.model == m]
            ax.errorbar(g.tokens_per_word, g.bits_per_byte, yerr=[g.bits_per_byte - g.ci_low, g.ci_high - g.bits_per_byte],
                        fmt="o", ms=9, color=c, mec=SURFACE, mew=2, elinewidth=1.2, capsize=3, zorder=3)
        ax.set_title(LANG[lang], loc="left", fontsize=10.5)
        lo, hi = d.tokens_per_word.min(), d.tokens_per_word.max()
        pad = max(0.35, 0.2 * (hi - lo))
        ax.set_xlim(lo - pad - 0.3, hi + pad)
        ax.set_xlabel("Tokens per word (left = cheaper)")
    axes[0].set_ylabel("Bits per byte (lower = better prediction)")
    handles = [Line2D([0], [0], marker="o", ls="", ms=9, color=BLUE, label="Qwen3.5-4B-Base"),
               Line2D([0], [0], marker="o", ls="", ms=9, color=ORANGE, label="ISSAI Kazakh-adapted")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    title(fig, "Cost vs. quality: moving down-left is better on both",
          "Arrow = original → ISSAI on the same text. Kazakh: cheaper and better. Russian: costlier but still better. English: same cost, better.")
    save(fig, "0_cost_vs_quality.png")


if __name__ == "__main__":
    fig_hero()
    fig_tokens_per_word()
    fig_change()
    fig_russian_words()
    fig_counterfactual()
    fig_stems()
    fig_bpb()
    fig_bench()
    fig_subjects()
