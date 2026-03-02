from __future__ import annotations

import csv
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    plot_report_violins_scores_and_lengths(
        report_csv_path="bertscore_report.csv",
        length_unit="words",
        show=True    )