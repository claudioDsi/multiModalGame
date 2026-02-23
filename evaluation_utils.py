from __future__ import annotations

import os
import re
import csv
import ast
from typing import Dict, Optional, Tuple, List, Iterable

import pandas as pd  # for pd.isna


# ---------------------------------------------------------------------------
# 0) Game-id extraction
# ---------------------------------------------------------------------------

def _extract_first_game_id(value) -> str | None:
    """
    Extract the first game ID from a value that may be:
    - a scalar (int / str)
    - a comma-separated string ("123,456")
    - a stringified list ("[123, 456]")
    - an actual list

    Returns the first ID as a string, or None if not extractable.
    """
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
# 1) TXT parsing helpers
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


def load_generated_background_story_by_game_id(
    game_id: str,
    generated_txt_root: str,
    *,
    filename_ext: str = ".txt",
) -> Tuple[str, str]:
    """
    Load generated TXT for game_id from:
      <generated_txt_root>/<game_id>.txt

    Returns: (background_story, txt_path)
    """
    game_id = str(game_id).strip()
    txt_path = os.path.join(generated_txt_root, f"{game_id}{filename_ext}")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Generated TXT not found for game_id={game_id}: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    parsed = parse_character_assets_txt(content)
    return parsed.get("background_story", "").strip(), txt_path


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
    """
    Read the original background story for `game_id` from CSV, matching against
    the first game id extracted from `games_column` via _extract_first_game_id().
    """
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
    """
    Collect distinct game IDs in first-seen order, using _extract_first_game_id().
    """
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
    """
    Configure Hugging Face token (optional).
    This supports bert-score downloading private/gated models, and avoids rate limits.
    """
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token  # common alias used by some libs


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
    Compute BERTScore between original (CSV) and generated (TXT) background story for a game_id.
    Returns a dict including original text, generated text, and scores.
    """
    _set_hf_token(hf_token)

    original_story = load_original_background_story_from_csv(
        csv_path=original_csv_path,
        game_id=game_id,
        games_column=games_column,
        original_story_column=original_story_column,
    )
    generated_story, txt_path = load_generated_background_story_by_game_id(
        game_id=game_id,
        generated_txt_root=generated_txt_root,
    )

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
        "precision": float(P[0].item()),
        "recall": float(R[0].item()),
        "f1": float(F1[0].item()),
        "generated_txt_path": txt_path,
    }


# ---------------------------------------------------------------------------
# 4) Batch processing + export to CSV
# ---------------------------------------------------------------------------

def export_bertscore_results_to_csv(
    results: List[Dict[str, object]],
    output_csv_path: str,
) -> None:
    """
    Export results to CSV. Ensures stable column order.
    """
    fieldnames = [
        "game_id",
        "precision",
        "recall",
        "f1",
        "original_background_story",
        "generated_background_story",
        "generated_txt_path",
        "error",
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"BERTScore results exported to: {output_csv_path}")


def compute_bertscore_for_games_and_export(
    *,
    original_csv_path: str,
    generated_txt_root: str,
    output_csv_path: str,
    # You can provide explicit game_ids, otherwise they are collected from the CSV
    game_ids: Optional[Iterable[str]] = None,
    games_column: str = "games",
    original_story_column: str = "background_story",
    # Collection controls if game_ids is None
    max_games: Optional[int] = None,
    offset: int = 0,
    # BERTScore params
    lang: str = "en",
    model_type: Optional[str] = None,
    # NEW: HF token support
    hf_token: Optional[str] = None,
) -> None:
    """
    Compute BERTScore for multiple game IDs and export a CSV containing:
      - original description (background story)
      - generated background story
      - final scores (precision/recall/f1)

    Game ID resolution:
      - If `game_ids` is provided: uses those (each value normalized by _extract_first_game_id)
      - Else: collects distinct IDs from `original_csv_path` using `games_column`
        with (offset, max_games)
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
                    "generated_txt_path": "",
                    "error": str(e),
                }
            )

    export_bertscore_results_to_csv(results, output_csv_path)


if __name__ == "__main__":
    compute_bertscore_for_games_and_export(
        original_csv_path="characters.csv",
        generated_txt_root="results",
        output_csv_path="bertscore_report.csv",
        games_column="games",
        original_story_column="description",
        max_games=58,
        offset=0,
        lang="en",
        model_type=None,
        hf_token="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",  # set if needed
    )

