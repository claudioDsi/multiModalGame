from requests import post
response = post('https://api.igdb.com/v4/characters', **{'headers': {'Client-ID': 'Client ID', 'Authorization': 'Bearer access_token'},'data': 'fields akas,character_gender,character_species,checksum,country_name,created_at,description,games,gender,mug_shot,name,slug,species,updated_at,url;'})
print ("response: %s" % str(response.json()))