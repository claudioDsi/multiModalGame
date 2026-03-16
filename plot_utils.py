from __future__ import annotations

import csv
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt

import numpy as np


def _text_length(text: str, *, unit: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    if unit == "chars":
        return len(text)
    if unit == "words":
        return len(text.split())
    raise ValueError("length_unit must be 'words' or 'chars'")


def _load_scores_and_lengths_from_report(
    report_csv_path: str,
    *,
    precision_column: str = "precision",
    recall_column: str = "recall",
    f1_column: str = "f1",
    original_text_column: str = "original_background_story",
    generated_text_column: str = "generated_background_story",
    length_unit: str = "words",
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Load P/R/F1 and original/generated lengths from the report CSV.
    Rows are kept only if they have valid numeric scores and non-empty texts.
    Returns: (precisions, recalls, f1s, orig_lens, gen_lens)
    """
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    orig_lens: List[float] = []
    gen_lens: List[float] = []

    with open(report_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no headers: {report_csv_path}")

        required = {
            precision_column,
            recall_column,
            f1_column,
            original_text_column,
            generated_text_column,
        }
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in report: {missing}")

        for row in reader:
            # Parse scores
            try:
                p = float((row.get(precision_column) or "").strip())
                r = float((row.get(recall_column) or "").strip())
                f1 = float((row.get(f1_column) or "").strip())
            except ValueError:
                continue

            orig_text = (row.get(original_text_column) or "").strip()
            gen_text = (row.get(generated_text_column) or "").strip()
            if not orig_text or not gen_text:
                continue

            o_len = _text_length(orig_text, unit=length_unit)
            g_len = _text_length(gen_text, unit=length_unit)
            if o_len <= 0 or g_len <= 0:
                continue

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            orig_lens.append(float(o_len))
            gen_lens.append(float(g_len))

    if not f1s:
        raise ValueError("No valid rows found (need numeric P/R/F1 and non-empty texts).")

    return precisions, recalls, f1s, orig_lens, gen_lens


def plot_report_violins_scores_and_lengths(
    report_csv_path: str,
    *,
    precision_column: str = "precision",
    recall_column: str = "recall",
    f1_column: str = "f1",
    original_text_column: str = "original_background_story",
    generated_text_column: str = "generated_background_story",
    length_unit: str = "words",  # "words" or "chars"
    # outputs
    scores_output_path: Optional[str] = None,
    lengths_output_path: Optional[str] = None,
    # titles
    #scores_title: str = "BERTScore Distributions (Precision / Recall / F1)",
    #lengths_title: str = "Text Length Distributions (Original vs Generated)",
    # display controls
    show: bool = True,
) -> None:
    """
    Produces TWO separate violin plots using matplotlib (no seaborn):
      I) Precision / Recall / F1 in one plot
      II) Original vs Generated text lengths in another plot

    If output paths are provided, saves PNGs. If show=True, displays both plots.
    """
    precisions, recalls, f1s, orig_lens, gen_lens = _load_scores_and_lengths_from_report(
        report_csv_path,
        precision_column=precision_column,
        recall_column=recall_column,
        f1_column=f1_column,
        original_text_column=original_text_column,
        generated_text_column=generated_text_column,
        length_unit=length_unit,
    )

    # --- Plot I: Scores (P/R/F1) ---
    fig_scores = plt.figure()
    ax_scores = fig_scores.add_subplot(111)

    ax_scores.violinplot(
        [precisions, recalls, f1s],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    ax_scores.set_xticks([1, 2, 3])
    ax_scores.set_xticklabels(["Precision", "Recall", "F1"], rotation=10, ha="right")
    #ax_scores.set_title(scores_title)
    ax_scores.set_ylabel("Score")
    ax_scores.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig_scores.tight_layout()

    if scores_output_path:
        fig_scores.savefig(scores_output_path, dpi=200, bbox_inches="tight")

    # --- Plot II: Lengths (Original/Generated) ---
    fig_lengths = plt.figure()
    ax_lengths = fig_lengths.add_subplot(111)

    ax_lengths.violinplot(
        [orig_lens, gen_lens],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    ax_lengths.set_xticks([1, 2])
    ax_lengths.set_xticklabels(
        [f"Original length", f"Generated length"],
        rotation=10,
        ha="right",
    )
    #ax_lengths.set_title(lengths_title)
    ax_lengths.set_ylabel("Length (words)")
    ax_lengths.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig_lengths.tight_layout()

    if lengths_output_path:
        fig_lengths.savefig(lengths_output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_scores)
        plt.close(fig_lengths)




def _load_f1_and_lengths(
    report_csv_path: str,
    *,
    f1_column: str = "f1",
    original_text_column: str = "original_background_story",
    generated_text_column: str = "generated_background_story",
    length_unit: str = "words",
) -> Tuple[List[float], List[float], List[float]]:
    """
    Load F1 scores, original lengths, and generated lengths from one report CSV.
    Keeps only rows with numeric F1 and non-empty original/generated text.
    """
    f1s: List[float] = []
    original_lengths: List[float] = []
    generated_lengths: List[float] = []

    with open(report_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no headers: {report_csv_path}")

        required = {f1_column, original_text_column, generated_text_column}
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {report_csv_path}: {missing}")

        for row in reader:
            try:
                f1 = float((row.get(f1_column) or "").strip())
            except ValueError:
                continue

            original_text = (row.get(original_text_column) or "").strip()
            generated_text = (row.get(generated_text_column) or "").strip()
            if not original_text or not generated_text:
                continue

            orig_len = _text_length(original_text, unit=length_unit)
            gen_len = _text_length(generated_text, unit=length_unit)

            if orig_len <= 0 or gen_len <= 0:
                continue

            f1s.append(f1)
            original_lengths.append(float(orig_len))
            generated_lengths.append(float(gen_len))

    if not f1s:
        raise ValueError(f"No valid rows found in {report_csv_path}")

    return f1s, original_lengths, generated_lengths


def plot_llm_comparison_violins(
    model1_csv_path: str,
    model2_csv_path: str,
    model3_csv_path: str,
    *,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    model3_name: str = "Model 3",
    f1_column: str = "f1",
    original_text_column: str = "original_background_story",
    generated_text_column: str = "generated_background_story",
    length_unit: str = "words",
    f1_output_path: str = "llm_f1_comparison_violin.png",
    lengths_output_path: str = "llm_lengths_comparison_violin.png",
base_font=15,
   # f1_title: str = "BERTScore F1 Comparison Across Models",
   # lengths_title: str = "Text Length Comparison Across Models and Original",
    show: bool =False,
) -> None:
    """
    Compare three LLM report CSV files with two violin plots:

    1) F1 violin plot for the three models
    2) Length violin plot for:
       - Original text
       - Generated text from model 1
       - Generated text from model 2
       - Generated text from model 3

    Assumes all three CSV files have the same structure as the original report.

    Outputs:
      - one PNG for F1 comparison
      - one PNG for length comparison
    """

    m1_f1, m1_orig_len, m1_gen_len = _load_f1_and_lengths(
        model1_csv_path,
        f1_column=f1_column,
        original_text_column=original_text_column,
        generated_text_column=generated_text_column,
        length_unit=length_unit,
    )
    m2_f1, m2_orig_len, m2_gen_len = _load_f1_and_lengths(
        model2_csv_path,
        f1_column=f1_column,
        original_text_column=original_text_column,
        generated_text_column=generated_text_column,
        length_unit=length_unit,
    )
    m3_f1, m3_orig_len, m3_gen_len = _load_f1_and_lengths(
        model3_csv_path,
        f1_column=f1_column,
        original_text_column=original_text_column,
        generated_text_column=generated_text_column,
        length_unit=length_unit,
    )

    # Use original lengths from the first report.
    # If the reports come from the same benchmark set, these should correspond
    # to the same original texts.
    original_lengths = m1_orig_len

    # --- Plot 1: F1 comparison ---
    fig_f1 = plt.figure()
    ax_f1 = fig_f1.add_subplot(111)

    ax_f1.violinplot(
        [m1_f1, m2_f1, m3_f1],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    ax_f1.set_xticks([1, 2, 3])
    ax_f1.set_xticklabels([model1_name, model2_name, model3_name], rotation=10, ha="right", fontsize=base_font)
    ax_f1.set_ylabel("F1 Score", fontsize=base_font)
   # ax_f1.set_title(f1_title)
    ax_f1.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig_f1.tight_layout()
    fig_f1.savefig(f1_output_path, dpi=200, bbox_inches="tight")

    # --- Plot 2: length comparison ---
    fig_len = plt.figure()
    ax_len = fig_len.add_subplot(111)

    ax_len.violinplot(
        [original_lengths, m1_gen_len, m2_gen_len, m3_gen_len],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    ax_len.set_xticks([1, 2, 3, 4])
    ax_len.set_xticklabels(
        [
            f"Original",
            f"{model1_name}",
            f"{model2_name}",
            f"{model3_name}",
        ],
        rotation=10,
        ha="right",
        fontsize=base_font
    )
    ax_len.set_ylabel("Length", fontsize=base_font)
    #ax_len.set_title(lengths_title)
    ax_len.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig_len.tight_layout()
    fig_len.savefig(lengths_output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_f1)
        plt.close(fig_len)




import csv
from typing import Optional
from wordcloud import WordCloud


def _collect_generated_text_from_report(
    report_csv_path: str,
    *,
    generated_text_column: str = "generated_background_story",
) -> str:
    """
    Read all generated background stories from a report CSV and concatenate them
    into a single corpus string.
    """
    texts: list[str] = []

    with open(report_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no headers: {report_csv_path}")

        if generated_text_column not in reader.fieldnames:
            raise ValueError(
                f"Missing column '{generated_text_column}' in {report_csv_path}"
            )

        for row in reader:
            text = (row.get(generated_text_column) or "").strip()
            if text:
                texts.append(text)

    if not texts:
        raise ValueError(f"No valid generated texts found in {report_csv_path}")

    return "\n".join(texts)


def _generate_wordcloud_image(
    text: str,
    output_path: str,
    *,
    title: Optional[str] = None,
    width: int = 1600,
    height: int = 900,
    max_words: int = 200,
    background_color: str = "white",
    collocations: bool = True,
) -> None:
    """
    Generate and save a single word cloud image from input text.
    """
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        collocations=collocations,
    ).generate(text)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_model_wordclouds(
    model1_csv_path: str,
    model2_csv_path: str,
    model3_csv_path: str,
    *,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    model3_name: str = "Model 3",
    generated_text_column: str = "generated_background_story",
    model1_output_path: str = "wordcloud_model1.png",
    model2_output_path: str = "wordcloud_model2.png",
    model3_output_path: str = "wordcloud_model3.png",
    width: int = 1600,
    height: int = 900,
    max_words: int = 200,
    background_color: str = "white",
    collocations: bool = True,
) -> None:
    """
    Generate three separate word cloud images, one for each model report CSV.

    Each CSV is expected to contain the same structure as the previous report,
    including a generated text column such as:
      - generated_background_story

    Outputs:
      - model1_output_path
      - model2_output_path
      - model3_output_path

    Requires:
      pip install wordcloud matplotlib
    """
    text1 = _collect_generated_text_from_report(
        model1_csv_path,
        generated_text_column=generated_text_column,
    )
    text2 = _collect_generated_text_from_report(
        model2_csv_path,
        generated_text_column=generated_text_column,
    )
    text3 = _collect_generated_text_from_report(
        model3_csv_path,
        generated_text_column=generated_text_column,
    )

    _generate_wordcloud_image(
        text1,
        model1_output_path,
        title=f"Word Cloud - {model1_name}",
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        collocations=collocations,
    )

    _generate_wordcloud_image(
        text2,
        model2_output_path,
        title=f"Word Cloud - {model2_name}",
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        collocations=collocations,
    )

    _generate_wordcloud_image(
        text3,
        model3_output_path,
        title=f"Word Cloud - {model3_name}",
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        collocations=collocations,
    )

    print("Word cloud images written to:")
    print(f"  - {model1_output_path}")
    print(f"  - {model2_output_path}")
    print(f"  - {model3_output_path}")

def plot_bar_chart(
        data: Dict,
        *,
        title: str = "Bar Plot",
        xlabel: str = "",
        ylabel: str = "",
        fontsize: int = 12,
        bar_width: float = 0.2,
        output_path: Optional[str] = None,
        show: bool = True,
):
    """
    Generic 3-dimension bar plot.

    Expected dictionary structure:

    {
        "ModelA": {
            "Precision": {"Dataset1": 0.8, "Dataset2": 0.75},
            "Recall": {"Dataset1": 0.78, "Dataset2": 0.70},
            "F1": {"Dataset1": 0.79, "Dataset2": 0.72},
        },
        "ModelB": {
            "Precision": {"Dataset1": 0.82, "Dataset2": 0.77},
            ...
        }
    }

    Dimensions:
    - Level 1 → outer group (e.g. models)
    - Level 2 → metric
    - Level 3 → x-axis categories

    Parameters
    ----------
    data : dict
        Nested dictionary with 3 levels.
    fontsize : int
        Font size for labels and ticks.
    bar_width : float
        Width of each bar.
    output_path : Optional[str]
        If provided, saves the figure.
    show : bool
        Whether to display the figure.
    """

    level1 = list(data.keys())
    level2 = list(next(iter(data.values())).keys())
    level3 = list(next(iter(next(iter(data.values())).values())).keys())

    x = np.arange(len(level3))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    total_bars = len(level1) * len(level2)
    offset_index = 0

    for l1 in level1:
        for l2 in level2:
            values = [data[l1][l2].get(cat, 0) for cat in level3]

            offset = (offset_index - total_bars / 2) * bar_width + bar_width / 2

            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=f"{l1}-{l2}",
            )

            offset_index += 1

    ax.set_xticks(x)
    ax.set_xticklabels(level3, fontsize=fontsize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)

    ax.legend(fontsize=fontsize - 2)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bar_from_dict(
    data: Dict[str, float],
    *,
    title: str = "Bar Plot",
    xlabel: str = "",
    ylabel: str = "",
    fontsize: int = 12,
    figsize: tuple = (8, 5),
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generic bar plot for flat dictionary data.

    Example input:
    data = {
        'gpt-oss': 0.96,
        'deepseek': 0.92,
        'glm': 0.30
    }

    Parameters
    ----------
    data : Dict[str, float]
        Dictionary with labels as keys and numeric values.
    fontsize : int
        Font size for plot text.
    figsize : tuple
        Size of the matplotlib figure.
    output_path : Optional[str]
        If provided, saves the plot image.
    show : bool
        Whether to display the plot.
    """

    labels = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.bar(labels, values)

    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)



if __name__ == "__main__":

    '''    plot_report_violins_scores_and_lengths(
        report_csv_path="bertscore_report_gpt_oss.csv",
        length_unit="words",
        show=True    )'''

    data = {
        'gpt-oss': 0.96, 'deepseek': 0.92, 'glm': 1.0
    }

    plot_bar_from_dict(
        data,
        title="",
        ylabel="V1 score",
        fontsize=15,
        output_path="conceptual_scores.pdf"
    )


'''
    plot_llm_comparison_violins(
        model1_csv_path="bertscore_report_gpt_oss.csv",
        model2_csv_path="bertscore_report_deepseek.csv",
        model3_csv_path="bertscore_report_glm.csv",
        model1_name="gpt-oss",
        model2_name="deepseek",
        model3_name="glm",
        length_unit="words",
        f1_output_path="f1_models_violin.png",
        lengths_output_path="lengths_models_violin.png",
        show=False,
    )

    generate_model_wordclouds(
        model1_csv_path="bertscore_report_gpt_oss.csv",
        model2_csv_path="bertscore_report_deepseek.csv",
        model3_csv_path="bertscore_report_glm.csv",
        model1_name="gpt-oss",
        model2_name="deepseek",
        model3_name="glm",
        model1_output_path="plots/wordcloud_gpt_oss.png",
        model2_output_path="plots/wordcloud_deepseek.png",
        model3_output_path="plots/wordcloud_glm.png",
    )

'''