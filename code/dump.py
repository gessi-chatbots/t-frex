import sys, json

try:
    with open('data/reviews.json', 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            sys.exit("Input file is not a valid JSON file.")
except FileNotFoundError:
    sys.exit("Specified file not found.")

categories = ['PRODUCTIVITY',
              'COMMUNICATION',
              'TOOLS',
              'SOCIAL',
              'HEALTH_AND_FITNESS',
              'PERSONALIZATION',
              'TRAVEL_AND_LOCAL',
              'MAPS_AND_NAVIGATION',
              'LIFESTYLE',
              'WEATHER']

apps = []
for app in data:
    if app['categoryId'] in categories:
        apps.append(app)

# Write the list to the JSON file
with open('data/reviews-f.json', "w") as json_file:
    json.dump(apps, json_file, indent=4)