from __future__ import annotations

import os
import re
import csv
import ast
from typing import Dict, Optional, Tuple, List, Iterable
import math
import pandas as pd  # for pd.isna


# ---------------------------------------------------------------------------
# 0) Game-id extraction
# ---------------------------------------------------------------------------

def _extract_first_game_id(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, list) and value:
        return str(value[0])

    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass

        if "," in value:
            return value.split(",")[0].strip()

        return value

    return str(value)


# ---------------------------------------------------------------------------
# 1) TXT parsing helpers (reused)
# ---------------------------------------------------------------------------

def parse_character_assets_txt(txt_content: str) -> Dict[str, str]:
    """
    Parse a generated character-asset TXT into sections.

    Returns keys:
      - name
      - image_prompt
      - background_story
      - skill_tree
    """
    text = (txt_content or "").replace("\r\n", "\n").replace("\r", "\n")

    def grab_section(label: str, next_labels: List[str]) -> str:
        pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*\n(.*?)(?=^\s*(?:{'|'.join(map(re.escape, next_labels))})\s*:|\Z)"
        m = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
        return (m.group(1).strip() if m else "")

    return {
        "name": grab_section("Name", ["Image Prompt", "Background Story", "Skill Tree"]),
        "image_prompt": grab_section("Image Prompt", ["Background Story", "Skill Tree", "Name"]),
        "background_story": grab_section("Background Story", ["Skill Tree", "Image Prompt", "Name"]),
        "skill_tree": grab_section("Skill Tree", ["Background Story", "Image Prompt", "Name"]),
    }


def load_generated_assets_by_game_id(
    game_id: str,
    generated_txt_root: str,
    *,
    filename_ext: str = ".txt",
) -> Tuple[Dict[str, str], str]:
    """
    Load generated TXT for game_id from:
      <generated_txt_root>/<game_id>.txt

    Returns: (parsed_sections_dict, txt_path)
    """
    game_id = str(game_id).strip()
    txt_path = os.path.join(generated_txt_root, f"{game_id}{filename_ext}")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Generated TXT not found for game_id={game_id}: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    return parse_character_assets_txt(content), txt_path


# ---------------------------------------------------------------------------
# 2) CSV lookup + collection
# ---------------------------------------------------------------------------

def load_original_background_story_from_csv(
    csv_path: str,
    game_id: str,
    *,
    games_column: str = "games",
    original_story_column: str = "background_story",
) -> str:
    game_id = str(game_id).strip()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no headers: {csv_path}")
        if games_column not in reader.fieldnames:
            raise ValueError(f"Missing column '{games_column}' in {csv_path}")
        if original_story_column not in reader.fieldnames:
            raise ValueError(f"Missing column '{original_story_column}' in {csv_path}")

        for row in reader:
            row_gid = _extract_first_game_id(row.get(games_column))
            if row_gid and str(row_gid).strip() == game_id:
                return (row.get(original_story_column) or "").strip()

    raise ValueError(f"No row found in {csv_path} matching game_id={game_id} via '{games_column}'")


def collect_game_ids_from_csv(
    csv_path: str,
    *,
    games_column: str = "games",
    max_games: Optional[int] = None,
    offset: int = 0,
) -> List[str]:
    if max_games is not None and max_games < 0:
        raise ValueError("max_games must be None or >= 0")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    seen: set[str] = set()
    order: List[str] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or games_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain column '{games_column}'")

        for row in reader:
            gid = _extract_first_game_id(row.get(games_column))
            if not gid:
                continue
            gid = str(gid).strip()
            if gid and gid not in seen:
                seen.add(gid)
                order.append(gid)

    start = min(offset, len(order))
    end = len(order) if max_games is None else min(start + max_games, len(order))
    return order[start:end]


# ---------------------------------------------------------------------------
# 3) BERTScore core
# ---------------------------------------------------------------------------

def _set_hf_token(hf_token: Optional[str]) -> None:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token


def compute_bertscore_for_game(
    *,
    game_id: str,
    original_csv_path: str,
    generated_txt_root: str,
    games_column: str = "games",
    original_story_column: str = "background_story",
    lang: str = "en",
    model_type: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, object]:
    """
    Returns a dict including original story, generated story, generated skill tree, and scores.
    """
    _set_hf_token(hf_token)

    original_story = load_original_background_story_from_csv(
        csv_path=original_csv_path,
        game_id=game_id,
        games_column=games_column,
        original_story_column=original_story_column,
    )

    parsed, txt_path = load_generated_assets_by_game_id(
        game_id=game_id,
        generated_txt_root=generated_txt_root,
    )
    generated_story = (parsed.get("background_story") or "").strip()
    generated_skill_tree = (parsed.get("skill_tree") or "").strip()

    if not original_story:
        raise ValueError(f"Original story is empty for game_id={game_id}")
    if not generated_story:
        raise ValueError(f"Generated Background Story is empty for game_id={game_id} (file: {txt_path})")

    try:
        from bert_score import score as bertscore_score  # type: ignore
    except ImportError as e:
        raise ImportError("Missing dependency: bert-score. Install with: pip install bert-score") from e

    P, R, F1 = bertscore_score(
        cands=[generated_story],
        refs=[original_story],
        lang=lang,
        model_type=model_type,
        verbose=False,
    )

    return {
        "game_id": str(game_id),
        "original_background_story": original_story,
        "generated_background_story": generated_story,
        "generated_skill_tree": generated_skill_tree,
        "precision": float(P[0].item()),
        "recall": float(R[0].item()),
        "f1": float(F1[0].item()),
        "generated_txt_path": txt_path,
    }


# ---------------------------------------------------------------------------
# 4) Export to CSV (with new column)
# ---------------------------------------------------------------------------

def export_bertscore_results_to_csv(
    results: List[Dict[str, object]],
    output_csv_path: str,
) -> None:
    """
    Export gpt_oss_results to CSV including generated_skill_tree.
    """
    fieldnames = [
        "game_id",
        "precision",
        "recall",
        "f1",
        "original_background_story",
        "generated_background_story",
        "generated_skill_tree",
        "generated_txt_path",
        "error",
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"BERTScore gpt_oss_results exported to: {output_csv_path}")


def compute_bertscore_for_games_and_export(
    *,
    original_csv_path: str,
    generated_txt_root: str,
    output_csv_path: str,
    # You can provide explicit game_ids, otherwise collected from the CSV
    game_ids: Optional[Iterable[str]] = None,
    games_column: str = "games",
    original_story_column: str = "background_story",
    # Collection controls if game_ids is None
    max_games: Optional[int] = None,
    offset: int = 0,
    # BERTScore params
    lang: str = "en",
    model_type: Optional[str] = None,
    # HF token support
    hf_token: Optional[str] = None,
) -> None:
    """
    Compute BERTScore for multiple game IDs and export a CSV containing:
      - original background story
      - generated background story
      - generated skill tree (NEW)
      - precision/recall/f1
    """
    # Resolve game IDs
    if game_ids is not None:
        resolved: List[str] = []
        for gid in game_ids:
            first = _extract_first_game_id(gid)
            if first:
                resolved.append(str(first).strip())
        game_id_list = [g for g in resolved if g]
    else:
        game_id_list = collect_game_ids_from_csv(
            original_csv_path,
            games_column=games_column,
            max_games=max_games,
            offset=offset,
        )

    if not game_id_list:
        print("No game ids to process.")
        return

    results: List[Dict[str, object]] = []
    for gid in game_id_list:
        try:
            r = compute_bertscore_for_game(
                game_id=gid,
                original_csv_path=original_csv_path,
                generated_txt_root=generated_txt_root,
                games_column=games_column,
                original_story_column=original_story_column,
                lang=lang,
                model_type=model_type,
                hf_token=hf_token,
            )
            r["error"] = ""
            results.append(r)
        except Exception as e:
            results.append(
                {
                    "game_id": str(gid),
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "original_background_story": "",
                    "generated_background_story": "",
                    "generated_skill_tree": "",
                    "generated_txt_path": "",
                    "error": str(e),
                }
            )

    export_bertscore_results_to_csv(results, output_csv_path)

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stddev(values: List[float], *, sample: bool = True) -> float:
    """
    Standard deviation. sample=True -> sample std (n-1). If n<2 returns nan.
    """
    n = len(values)
    if n == 0:
        return float("nan")
    if n == 1:
        return float("nan")
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / (n - 1 if sample else n)
    return math.sqrt(var)


def _pearson_corr(x: List[float], y: List[float]) -> float:
    """
    Pearson correlation for paired lists. Returns nan if undefined.
    """
    n = min(len(x), len(y))
    if n < 2:
        return float("nan")

    x = x[:n]
    y = y[:n]
    mx = _mean(x)
    my = _mean(y)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / (sx * sy)


def _percentile(values: List[float], p: float) -> float:
    """
    Simple linear-interpolated percentile, p in [0,100].
    """
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)

    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


def compute_report_statistics(
    report_csv_path: str,
    *,
    original_column: str = "original_background_story",
    generated_column: str = "generated_background_story",
    # optional scores if present in the report
    f1_column: str = "f1",
    precision_column: str = "precision",
    recall_column: str = "recall",
    # length measure
    length_unit: str = "words",  # "words" or "chars"
    # output
    output_csv_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute descriptive statistics from the BERTScore report CSV:
      - Avg & stddev length (original/generated)
      - Avg & stddev compression ratio (generated/original)
      - Avg & stddev BERTScore (P/R/F1) if columns exist
      - Correlation between length metrics and F1 if F1 exists
      - Percentiles for lengths and F1 (p5, p25, p50, p75, p95)

    Skips rows where original or generated text is missing.
    Also skips score parsing for rows where score values are missing/non-numeric.

    Returns a dict of computed metrics. If output_csv_path is provided, writes a 1-row CSV.
    """
    if length_unit not in {"words", "chars"}:
        raise ValueError("length_unit must be either 'words' or 'chars'")

    orig_lens: List[float] = []
    gen_lens: List[float] = []
    ratios: List[float] = []

    f1s: List[float] = []
    ps: List[float] = []
    rs: List[float] = []

    def text_len(t: str) -> int:
        if length_unit == "chars":
            return len(t)
        return len(t.split())

    with open(report_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no headers: {report_csv_path}")

        if original_column not in reader.fieldnames:
            raise ValueError(f"Missing column '{original_column}'")
        if generated_column not in reader.fieldnames:
            raise ValueError(f"Missing column '{generated_column}'")

        has_f1 = f1_column in reader.fieldnames
        has_p = precision_column in reader.fieldnames
        has_r = recall_column in reader.fieldnames

        for row in reader:
            original = (row.get(original_column) or "").strip()
            generated = (row.get(generated_column) or "").strip()
            if not original or not generated:
                continue

            o_len = float(text_len(original))
            g_len = float(text_len(generated))
            if o_len <= 0:
                continue

            orig_lens.append(o_len)
            gen_lens.append(g_len)
            ratios.append(g_len / o_len)

            # parse scores if present
            if has_f1:
                try:
                    v = float((row.get(f1_column) or "").strip())
                    f1s.append(v)
                except ValueError:
                    pass
            if has_p:
                try:
                    v = float((row.get(precision_column) or "").strip())
                    ps.append(v)
                except ValueError:
                    pass
            if has_r:
                try:
                    v = float((row.get(recall_column) or "").strip())
                    rs.append(v)
                except ValueError:
                    pass

    if not orig_lens:
        raise ValueError("No valid rows found for computing statistics.")

    # Core length stats
    stats: Dict[str, float] = {
        "num_samples": float(len(orig_lens)),
        "length_unit": 0.0,  # placeholder for CSV numeric requirement; see note below
        "avg_original_length": _mean(orig_lens),
        "std_original_length": _stddev(orig_lens),
        "avg_generated_length": _mean(gen_lens),
        "std_generated_length": _stddev(gen_lens),
        "avg_length_ratio_generated_over_original": _mean(ratios),
        "std_length_ratio_generated_over_original": _stddev(ratios),
        "p05_original_length": _percentile(orig_lens, 5),
        "p25_original_length": _percentile(orig_lens, 25),
        "p50_original_length": _percentile(orig_lens, 50),
        "p75_original_length": _percentile(orig_lens, 75),
        "p95_original_length": _percentile(orig_lens, 95),
        "p05_generated_length": _percentile(gen_lens, 5),
        "p25_generated_length": _percentile(gen_lens, 25),
        "p50_generated_length": _percentile(gen_lens, 50),
        "p75_generated_length": _percentile(gen_lens, 75),
        "p95_generated_length": _percentile(gen_lens, 95),
        "p05_length_ratio": _percentile(ratios, 5),
        "p25_length_ratio": _percentile(ratios, 25),
        "p50_length_ratio": _percentile(ratios, 50),
        "p75_length_ratio": _percentile(ratios, 75),
        "p95_length_ratio": _percentile(ratios, 95),
    }

    # Score stats (if available)
    if f1s:
        stats.update(
            {
                "avg_f1": _mean(f1s),
                "std_f1": _stddev(f1s),
                "p05_f1": _percentile(f1s, 5),
                "p25_f1": _percentile(f1s, 25),
                "p50_f1": _percentile(f1s, 50),
                "p75_f1": _percentile(f1s, 75),
                "p95_f1": _percentile(f1s, 95),
                # Correlations
                "corr_f1_vs_original_length": _pearson_corr(orig_lens, f1s),
                "corr_f1_vs_generated_length": _pearson_corr(gen_lens, f1s),
                "corr_f1_vs_length_ratio": _pearson_corr(ratios, f1s),
            }
        )
    if ps:
        stats.update({"avg_precision": _mean(ps), "std_precision": _stddev(ps)})
    if rs:
        stats.update({"avg_recall": _mean(rs), "std_recall": _stddev(rs)})

    # Export (single-row) CSV, including length_unit as a string column.
    # Since Dict[str,float] can't hold strings, we write it separately in CSV.
    if output_csv_path:
        fieldnames = [
            "num_samples",
            "length_unit",
            "avg_original_length",
            "std_original_length",
            "avg_generated_length",
            "std_generated_length",
            "avg_length_ratio_generated_over_original",
            "std_length_ratio_generated_over_original",
            "p05_original_length",
            "p25_original_length",
            "p50_original_length",
            "p75_original_length",
            "p95_original_length",
            "p05_generated_length",
            "p25_generated_length",
            "p50_generated_length",
            "p75_generated_length",
            "p95_generated_length",
            "p05_length_ratio",
            "p25_length_ratio",
            "p50_length_ratio",
            "p75_length_ratio",
            "p95_length_ratio",
            "avg_precision",
            "std_precision",
            "avg_recall",
            "std_recall",
            "avg_f1",
            "std_f1",
            "p05_f1",
            "p25_f1",
            "p50_f1",
            "p75_f1",
            "p95_f1",
            "corr_f1_vs_original_length",
            "corr_f1_vs_generated_length",
            "corr_f1_vs_length_ratio",
        ]

        row_out: Dict[str, object] = {k: stats.get(k, "") for k in fieldnames}
        row_out["length_unit"] = length_unit  # string in exported CSV

        with open(output_csv_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_out)

    # Return dict (without string length_unit; caller already knows it)
    # If you want it included, change return type to Dict[str, object].
    return stats


if __name__ == "__main__":

    compute_bertscore_for_games_and_export(
        original_csv_path="characters.csv",
        generated_txt_root="glm_results",
        output_csv_path="bertscore_report_glm.csv",
        games_column="games",
        original_story_column="description",
        max_games=58,
        offset=0,
        lang="en",
        model_type=None,
        hf_token="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",  # set if needed
    )

    stats = compute_report_statistics(
        report_csv_path="bertscore_report_glm.csv",
        length_unit="words",
        output_csv_path="bertscore_report_stats_glm.csv",
    )

    print(stats["avg_original_length"], stats["avg_generated_length"], stats.get("avg_f1"))
