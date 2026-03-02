from requests import post
import pandas as pd
import config as cf
import ast
import os





def build_query_mug_shot(name):
    return f'fields name,mug_shot;  where  name = {name};'

def get_mug_shot(csv_char):
    df_char = pd.read_csv(csv_char)

    for name in df_char['name']:
        q = build_query_mug_shot(name)
        header = {'headers': {'Client-ID': cf.CLIENT_ID, 'Authorization': 'Bearer ' + cf.token}, 'data': q}
        response = post(cf.CHARACTERS_URL, **header)
        res = pd.DataFrame(response.json())

    return res






def build_query_characters():
    return f'fields akas,character_gender,character_species,checksum,country_name,created_at,description,games,gender,mug_shot,name,slug,species,updated_at,url; where description != ""; limit 100;'

def build_query_games(game_id):
    return f'fields name,genres,summary,screenshots;  where id={game_id};'


def get_characters():

    q = build_query_characters()
    header = {'headers': {'Client-ID': cf.CLIENT_ID, 'Authorization': 'Bearer ' + cf.token}, 'data': q}
    response = post(cf.CHARACTERS_URL, **header)
    res = pd.DataFrame(response.json())
    return res


def build_query_screenshots(game_id):
    return f"fields image_id,url,width; where game={game_id};"


def get_screenshot(game_id):
    df = pd.DataFrame()
    q = build_query_screenshots(game_id)
    header = {'headers': {'Client-ID': cf.CLIENT_ID, 'Authorization': 'Bearer ' + cf.token}, 'data': q}
    response = post(cf.SCREENSHOTS_URL, **header)
    res = pd.DataFrame(response.json())
    if len(res) > 0:
        res['id_game'] = game_id
        res['url'] = res['url'].str[2:]
        df = pd.concat([df,res],axis=0)
    df = df.rename(columns={'image_id':'id_image'})
    return df


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

    # Case 1: already a list
    if isinstance(value, list) and value:
        return str(value[0])

    # Case 2: stringified Python list
    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass

        # Case 3: comma-separated values
        if "," in value:
            return value.split(",")[0].strip()

        # Case 4: single scalar string
        return value

    # Case 5: numeric scalar
    return str(value)


def get_games_by_id(*, output_csv: str = "games_with_character.csv") -> None:
    """
    Read game IDs from the `games` column of characters.csv.
    If multiple IDs exist per row, only the FIRST one is used.
    Fetch game data and append results to a CSV file.
    """

    df_character = pd.read_csv("characters.csv")

    if "games" not in df_character.columns:
        raise ValueError("characters.csv must contain a 'games' column")

    for row_idx, raw_value in enumerate(df_character["games"], start=1):
        game_id = _extract_first_game_id(raw_value)

        if not game_id:
            print(f"Row {row_idx}: no valid game ID found, skipping")
            continue

        query = build_query_games(game_id)
        headers = {
            "headers": {
                "Client-ID": cf.CLIENT_ID,
                "Authorization": "Bearer " + cf.token,
            },
            "data": query,
        }

        response = post(cf.GAMES_URL, **headers)
        response.raise_for_status()

        data = response.json()
        if not data:
            print(f"Row {row_idx}: no data returned for game_id {game_id}")
            continue

        df_result = pd.DataFrame(data)
        df_result["source_game_id"] = game_id

        write_header = not os.path.exists(output_csv)
        df_result.to_csv(
            output_csv,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8",
        )

        print(f"Row {row_idx}: appended {len(df_result)} rows for game_id {game_id}")