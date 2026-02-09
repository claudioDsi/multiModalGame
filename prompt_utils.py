from __future__ import annotations
import requests
import json
import re
from typing import Optional
import csv
from typing import Dict, List



def _call_llm(model: str, prompt: str, url: str = "http://localhost:11434/api/generate") -> str:
    """Stream a response from a local LLM endpoint (Ollama-compatible)."""
    data = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}
    full_response = []
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True, timeout=300)
    try:
        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode("utf-8"))
                # Expected schema: {"response": "...", "done": bool, ...}
                chunk = decoded_line.get("response", "")
                if chunk:
                    full_response.append(chunk)
    finally:
        response.close()
    return "".join(full_response)




def build_game_obfuscation_prompt(
    description: str,
    game_name: Optional[str] = None,
    main_character: Optional[str] = None,
) -> str:
    """
    Build a prompt that asks an LLM (e.g. via Ollama) to obfuscate a video game
    description by removing sensitive identifiers while preserving meaning.

    - description: original game description.
    - game_name: optional known game title to redact if present.
    - main_character: optional known main character name to redact if present.
    """

    # Escape braces defensively in case of downstream formatting
    def esc(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    # Optional lightweight pre-redaction (helps avoid prompt logging leaks)
    sanitized_description = description or ""
    if game_name:
        sanitized_description = re.sub(
            re.escape(game_name),
            "<The game>",
            sanitized_description,
            flags=re.IGNORECASE,
        )
    if main_character:
        sanitized_description = re.sub(
            re.escape(main_character),
            "<the character>",
            sanitized_description,
            flags=re.IGNORECASE,
        )

    desc_esc = esc(sanitized_description)

    redaction_hints = []
    if game_name:
        redaction_hints.append(f'- Game title to redact if found: "{esc(game_name)}"')
    if main_character:
        redaction_hints.append(
            f'- Main character name to redact if found: "{esc(main_character)}"'
        )

    hints_text = (
        "\n".join(redaction_hints)
        if redaction_hints
        else "- No explicit identifiers provided; infer and redact any unique IP identifiers."
    )

    prompt = f"""
You are an AI assistant specialized in video game assets.

GOAL
- Given a game description, produce an OBFUSCATED VERSION that preserves narrative,
  tone, and gameplay semantics while removing sensitive identifying information.

WHAT TO OBFUSCATE (REPLACE)
- Game title / franchise / series name → "<The game>"
- Main character name → "<the character>"
- Pronouns:
  - If gender is explicit, preserve it using placeholders:
    "<he/she/they>", "<his/her/their>", "<man/woman/person>"
  - If gender is not explicit, default to neutral placeholders.
- Unique locations (named islands, cities, planets) → "<a location>"
- Named factions, organizations, kingdoms → "<an organization>"
- Named artifacts or legendary items → "<an artifact>"
- Lore-specific events or timelines → "<a past event>"

RULES
- Preserve meaning, pacing, and genre.
- Do NOT introduce new lore or details.
- Keep edits minimal but ensure identifiers are fully removed.
- Remove all proper nouns that could uniquely identify the IP.
- Do NOT invent a main character if none is mentioned.
- Output ONLY the obfuscated description text.

REDACTION HINTS
{hints_text}

INPUT DESCRIPTION
\"\"\"text
{desc_esc}
\"\"\"

OUTPUT FORMAT
- Plain text only.
- No explanations, no bullet lists, no headings.
"""

    return prompt.strip()




def obfuscate_character_descriptions_from_csv(
    csv_path: str,
    model: str,
    output_csv_path: str,
    *,
    description_column: str = "summary",
    obfuscated_column: str = "obfuscated_summary",
    game_name_column: Optional[str] = None,
    main_character_column: Optional[str] = None,
) -> None:
    """
    Read a CSV containing a character (or game) description column and write a new CSV
    with an additional column containing the obfuscated content (LLM-generated).

    CSV requirements:
      - must contain column: `description_column`
      - optionally may contain:
          * `game_name_column` (per-row game title to redact)
          * `main_character_column` (per-row main character name to redact)

    Writes a new CSV to `output_csv_path` with all original columns plus:
      - `obfuscated_column` (default: "obfuscated_description")
    """
    rows: List[Dict[str, str]] = []

    # 1) Load CSV
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        in_fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"No rows found in {csv_path}")
        return

    # Ensure obfuscated column is present in output fieldnames
    out_fieldnames = list(dict.fromkeys(in_fieldnames + [obfuscated_column]))

    # 2) Process each row
    for row in rows:
        description = (row.get(description_column) or "").strip()

        game_name: Optional[str] = None
        if game_name_column:
            game_name = (row.get(game_name_column) or "").strip() or None

        main_character: Optional[str] = None
        if main_character_column:
            main_character = (row.get(main_character_column) or "").strip() or None

        # 3) Build prompt & call LLM
        prompt = build_game_obfuscation_prompt(
            description=description,
            game_name=game_name,
            main_character=main_character,
        )

        obfuscated = _call_llm(model=model, prompt=prompt)

        row[obfuscated_column] = (obfuscated or "").strip()

    # 4) Write out new CSV
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Obfuscated descriptions written to: {output_csv_path}")





def build_character_generation_prompt(obfuscated_description: str) -> str:
    """
    Build a prompt that asks an LLM to generate a NEW main character concept
    from an obfuscated game description (no IP leakage).

    Output must include:
    - Name
    - Image prompt (optimized for vision models)
    - Background story
    - Skill tree
    """
    def esc(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    desc = esc((obfuscated_description or "").strip())

    prompt = f"""
You are an AI assistant specialized in video game assets.

INPUT
Given this obfuscated game description:
\"\"\"text
{desc}
\"\"\"

TASK
Generate a NEW main character concept that fits the description.
- Do NOT use or guess any real game/franchise/character names.
- Do NOT introduce identifiable IP-specific proper nouns.
- Keep the character consistent with the tone/genre implied by the description.

OUTPUT REQUIREMENTS
Return ONLY the following sections, in this exact order and format:

Name: <character name>

Image Prompt:
<one single, detailed prompt optimized for a vision model; include appearance, outfit, pose, environment, lighting, art style tags; avoid brand/franchise names>

Background Story:
<6–10 sentences; origin, motivation, internal conflict, stakes, how they connect to the setting>

Skill Tree:
- Tier 1:
  - <skill name>: <1 sentence effect>
  - <skill name>: <1 sentence effect>
  - <skill name>: <1 sentence effect>
- Tier 2:
  - <skill name>: <1 sentence effect>
  - <skill name>: <1 sentence effect>
  - <skill name>: <1 sentence effect>
- Tier 3:
  - <skill name>: <1 sentence effect>
  - <skill name>: <1 sentence effect>
  - <skill name>: <1 sentence effect>

CONSTRAINTS
- No meta commentary, no apologies, no extra sections.
- Plain text only.
"""
    return prompt.strip()


def generate_characters_from_obfuscated_csv(
    csv_path: str,
    model: str,
    obfuscated_column: str,
    output_txt_path: str,
    *,
    id_column: str = "id",
    separator: str = "\n\n" + ("=" * 72) + "\n\n",
) -> None:
    """
    Read a CSV containing an obfuscated description column and generate a new main character
    for each row using the same LLM model.

    Inputs:
      - csv_path: input CSV file path
      - model: model name used by _call_llm (e.g., an Ollama model tag)
      - obfuscated_column: the column containing obfuscated descriptions
      - output_txt_path: plain text output file containing generated character assets

    Optional:
      - id_column: column used to label entries in the output (defaults to "id");
                   if missing, a row index label is used.
      - separator: text separator between entries in the output file
    """
    rows: List[Dict[str, str]] = []

    # 1) Load CSV
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"No rows found in {csv_path}")
        return

    # 2) Generate per row and write to a single plain text file
    outputs: List[str] = []
    for idx, row in enumerate(rows, start=1):
        obf = (row.get(obfuscated_column) or "").strip()
        if not obf:
            label = (row.get(id_column) or f"row_{idx}").strip()
            outputs.append(
                f"Entry: {label}\n\n"
                f"(Skipped: empty '{obfuscated_column}' column value.)"
            )
            continue

        prompt = build_character_generation_prompt(obf)
        result = _call_llm(model=model, prompt=prompt).strip()

        label = (row.get(id_column) or f"row_{idx}").strip()
        outputs.append(f"Entry: {label}\n\n{result}")

    with open(output_txt_path, "w", encoding="utf-8") as f_out:
        f_out.write(separator.join(outputs).strip() + "\n")

    print(f"Character assets written to: {output_txt_path}")
