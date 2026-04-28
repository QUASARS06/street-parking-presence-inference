# import requests

# TOKEN = "MLY|26211270751827546|f6074fb2c180299d4b549f5114a7785f"
# # Use one actual key from your annotations directory
# TEST_KEY = "C8gApaHoD1KnEF2dPqfHxg"  # replace with any .json filename stem from your annotations

# # Test 1: basic auth check
# r = requests.get(
#     f"https://graph.mapillary.com/me",
#     params={"access_token": TOKEN}
# )
# print("Auth check:", r.status_code, r.json())

# # Test 2: try the key directly
# r = requests.get(
#     f"https://graph.mapillary.com/{TEST_KEY}",
#     params={"fields": "geometry", "access_token": TOKEN}
# )
# print("Direct key lookup:", r.status_code, r.json())

# # Test 3: try searching by image key field
# r = requests.get(
#     "https://graph.mapillary.com/images",
#     params={
#         "fields": "id,geometry,thumb_256_url",
#         "image_keys": TEST_KEY,   # v4 has this param for legacy key lookup
#         "access_token": TOKEN
#     }
# )
# print("Search by image_key:", r.status_code, r.json())

import requests, json
          
# Mapillary access token -- provide your own, replace this example
mly_key = 'MLY|26211270751827546|f6074fb2c180299d4b549f5114a7785f'

seq = 'kx7r2uYeFwT6XCfvqh30VZ'

url = f'https://graph.mapillary.com/image_ids?access_token={mly_key}&sequence_id={seq}'
          
response = requests.get(url)

image_ids = None

if response.status_code == 200:
   print("Successfully fetched image IDs for sequence.")
   json = response.json()
   image_ids = [obj['id'] for obj in json['data']]
   
   # make a dictionary to store each detection by image ID
   detections = {}
   for image_id in image_ids:
      dets_url = f'https://graph.mapillary.com/{image_id}/detections?access_token={mly_key}&fields=geometry,value'
      response = requests.get(dets_url)
      json = response.json()
      detections[image_id] = json['data']
      print(f"Fetched {len(json['data'])} detections for image {image_id}")