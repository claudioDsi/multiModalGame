# endpoints
GAMES_URL = "https://api.igdb.com/v4/games"
PLATFORM_URL = "https://api.igdb.com/v4/platforms"
GENRES_URL = "https://api.igdb.com/v4/genres"
VIDEOS_URL = "https://api.igdb.com/v4/game_videos"
SCREENSHOTS_URL = "https://api.igdb.com/v4/screenshots"
SEARCH_URL = "https://api.igdb.com/v4/search"
YOU_TUBE_URL = "http://youtube.com/watch?v="




#platform all

XB = "Xbox"
PL = "PlayStation"
NT = "Nintendo"
PC = "PC"


# local dirs

PATH_PL = "IGDB_metadata/" + PL + '/'
PATH_NT = "IGDB_metadata/" + NT + '/'
PATH_XB = "IGDB_metadata/" + XB + '/'
PATH_PC = "IGDB_metadata/" + PC + '/'

PATH_PL_VIDEOS = "videos/" + PL + '/'
PATH_NT_VIDEOS = "videos/" + NT + '/'
PATH_XB_VIDEOS = "videos/" + XB + '/'
PATH_PC_VIDEOS = "videos/" + PC + '/'

# downloaded video
VIDEO_DATA = '../all_video.csv'



# Screenshot download parameters
PATH_SCREENSHOTS = "screenshots/"
PATH_GENRES = "genres.csv"
GENRE_FOLDER = 'genres/'
GENRE_TEST = 'Turn-based strategy (TBS)'





PATH_PLATFORMS = "final_dataset/platforms.csv"



#error handling

MAX_RETRIES = 3  # Maximum number of retries
RETRY_INTERVAL = 5  # Seconds to wait between retries


# auth config
token = "your IGDB token"
CLIENT_ID = "your IGDB"
AUTH = {'Authorization': "Bearer {}".format(token)}

# headers
GENRES_HEADERS= {'headers': {'Client-ID': CLIENT_ID, 'Authorization': 'Bearer ' + token},'data': 'fields name; limit 500;'}
PLATFORM_HEADERS =  {'headers': {'Client-ID': CLIENT_ID, 'Authorization': 'Bearer ' + token},'data': 'fields id,name ;  limit 500;'}
#SEARCH_HEADERS =  {'headers': {'Client-ID': CLIENT_ID, 'Authorization': 'Bearer ' + token},'data': 'fields name,platform; search "Playstation"; limit 50;'}

#HLTB paramenters

HLTB_DATA = '../hltb.csv'