
from miner import get_characters, get_games_by_id
from prompt_utils import obfuscate_character_descriptions_from_csv, generate_characters_from_obfuscated_csv_to_files
import pandas as pd
if __name__ == '__main__':
    obfuscate_character_descriptions_from_csv(model="gpt-oss:120b-cloud",csv_path="games_with_character.csv", output_csv_path="games_obfuscated.csv")



    generate_characters_from_obfuscated_csv_to_files(csv_path="games_obfuscated.csv", model="glm-5:cloud",
                                           obfuscated_column="obfuscated_summary", output_dir="glm_results/")

    #get_games_by_id()



