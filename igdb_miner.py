from requests import post
import config as cf
import io, os
import ast
import pandas as pd
import time
import data_utils as du
from pandas.errors import EmptyDataError
from http.client import RemoteDisconnected
from urllib3.exceptions import MaxRetryError
from requests.exceptions import ConnectionError, Timeout


def build_query_video(game_id):
    return f"fields video_id,name,game; where game= {game_id} ;"


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



def get_all_screenshots(df_path):
    genres_dict = pd.read_csv(cf.PATH_GENRES)
    if not os.path.exists(cf.PATH_SCREENSHOTS):
        os.makedirs(cf.PATH_SCREENSHOTS)
    errors = pd.read_csv("progress/errors.csv")
    try:
        df = pd.read_csv(df_path)
    except:
        return 0
    
    for index,row in df.iterrows():
        id = row['id']
        genres = ast.literal_eval(row['genres']) if 'genres' in df.columns and row['genres'] != 'Missing' else []
        screenshots = get_screenshot(id)
        i = 0
        for _,row in screenshots.iterrows():
            i += 1
            if i ==50:
                time.sleep(2)
                i = 0
            id = row['id_game']
            screenshot = row['url']
            id_image = row['id_image']
            outs = []
            for g in genres:
                genre = genres_dict.loc[genres_dict.id == g, 'name'].item()
                if not os.path.exists(os.path.join(cf.PATH_SCREENSHOTS,"genres",str(genre))):
                    os.makedirs(os.path.join(cf.PATH_SCREENSHOTS,"genres",str(genre)))

                outs.append( os.path.join( cf.PATH_SCREENSHOTS, "genres" , str(genre) ,f"gameid_{id}_img_{id_image}.jpg") )
            
            try:
                du.download_image(screenshot, outs, size='thumbnail')
            except Exception as ex: 
                errors = pd.concat([errors, pd.DataFrame({'id_game':id, 'id_image':id_image,'reason':type(ex).__name__, 'reason_args':ex.args})])
                errors.to_csv("errors.csv")



def get_genres():
    response = post(cf.GENRES_URL, **cf.GENRES_HEADERS)
    df_genres = pd.read_json(io.StringIO(du.read_json_string(response)), orient='records')
    df_genres.to_csv('genres.csv', index=False)


def build_header_games(platform,rating_range):
    ids = get_platform_id(cf.PATH_PLATFORMS, platform)
    q = build_query_games(ids, rating_range[0] , rating_range[1])
    header = {'headers': {'Client-ID': cf.CLIENT_ID,
                          'Authorization': 'Bearer ' + cf.token},
              'data': q}
    return header


def get_video_ids(src, out_path):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(out_path + 'video_cat_nt.csv', 'a', encoding='utf8', errors='ignore') as video_file:
        with open(out_path + 'video_ids.csv', 'a', encoding='utf8', errors='ignore') as game_file:
            df_videos = pd.read_csv(out_path + 'video_ids.csv')

            for data in os.listdir(src):
                try:
                    df_games = pd.read_csv(src + data, sep=',')
                    for game_id in df_games['id'].values:
                        if game_id in df_videos['id_game'].values:
                            print("already downloaded")
                            continue

                        list_video_id = []
                        q = build_query_video(game_id)
                        header = {
                            'headers': {
                                'Client-ID': cf.CLIENT_ID,
                                'Authorization': 'Bearer ' + cf.token
                            },
                            'data': q
                        }

                        for attempt in range(cf.MAX_RETRIES):

                            try:
                                response = post(cf.VIDEOS_URL, **header)

                                break
                            except (ConnectionError, Timeout, MaxRetryError) as e:
                                print(f"Network error occurred: {e}")
                                # Optional: Implement a retry mechanism or continue to the next iteration
                                print(f"Attempt {attempt + 1} failed: {e}")
                                if attempt < cf.MAX_RETRIES - 1:
                                    print(f"Retrying in {cf.RETRY_INTERVAL} seconds...")
                                    time.sleep(cf.RETRY_INTERVAL)
                                else:
                                    print("Max retries reached, moving to the next item.")
                                    break
                                continue

                        if len(response.json()) > 0:
                            for v in response.json():
                                print("Writing id")
                                if 'video_id' in v.keys():
                                    url_video = f"{cf.YOU_TUBE_URL}{v['video_id']}"
                                    list_video_id.append(url_video)
                                if 'name' in v.keys():
                                    video_file.write(f"{url_video},{v['name']}\n")
                            video_ids = '#'.join(list_video_id)
                            game_file.write(f"{v['game']},{video_ids}\n")
                        else:
                            game_file.write(f"{game_id},Missing\n")
                            print("No video")
                        time.sleep(1)

                except RemoteDisconnected as e:
                    print(f"Error occurred: {e}")
                except EmptyDataError:
                    continue




def build_query_games(plat_ids, min_rate, max_rate):
    str_query = f"fields name,platforms,rating,genres,summary,storyline, first_release_date; where platforms=({','.join(str(i) for i in list(set(plat_ids)))}) & first_release_date >= {str(min_rate)} & first_release_date < {str(max_rate)}; sort rating desc; limit 500;"
    return str_query



def get_games_by_platform(platform, out_dir, rating_range):
    response = post(cf.GAMES_URL, **build_header_games(platform, rating_range))
    df_games = pd.read_json(io.StringIO(du.read_json_string(response)), orient='records')
    if 'summary' in df_games.columns:
        df_games = du.preprocess_column(df_games, 'summary')
    if 'storyline' in df_games.columns:
        df_games = du.preprocess_column(df_games, 'storyline')
    for col in df_games.columns.values:
        if du.has_nan(df_games,col):
            df_games = du.replace_nan_with_string(df_games, col, "Missing")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_games.to_csv(du.build_filename_string(out_dir,platform,rating_range), index=False)
    print("writing files")


def get_platform_ids():
    response = post(cf.PLATFORM_URL, **cf.PLATFORM_HEADERS)
    with open(cf.PATH_PLATFORMS, 'a', encoding='utf8', errors='ignore') as res:
        json_file = response.json()
        print("response: %s" % str(response.json()))
        # res.write('id,platform\n')
        for platform in json_file:
            res.write(str(platform.get('id')) + ',' + str(platform.get('name')) + '\n')
            

def get_platform_id(csv, type):
    df_play = pd.read_csv(csv, sep=',')
    list_ids = []
    for key, plat in zip(df_play['id'], df_play['name']):
        if str(plat).find(type) != -1:
            list_ids.append(key)
    return list_ids


