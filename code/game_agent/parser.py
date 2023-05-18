from map_parser import MapParser
import os
from random import shuffle


# Test map from local data:
# get Songs folder
osu_songs_directory = os.path.join(os.getenv('LOCALAPPDATA'), 'osu!', 'Songs')

# List Songs and shuffle the list
maps = os.listdir(osu_songs_directory)
# shuffle(maps)

song = "716630 S3RL - MTC (Radio Edit)"
difficulty = "S3RL - MTC (Radio Edit) (Okoratu) [pishi's Normal].osu"

# # Pick random map
# # map_path = os.path.join(osu_songs_directory, maps[0])
map_path = os.path.join(osu_songs_directory, song)

# Pick first .osu file
# file = [x for x in os.listdir(map_path) if x.endswith(".osu")][0]
osu_path = os.path.join(map_path, difficulty)
print(osu_path)

mp = MapParser(osu_path)

# mp = MapParser('examples/Niko - Made of Fire (lesjuh) [Oni].osu')

print("Version", mp.version)
print()
print("General", mp.general)
print()
print("Editor", mp.general)
print()
print("Metadata", mp.metadata)
print()
print("Difficulty", mp.difficulty)
print()
print("Events", mp.events)
print()

# print("Timing Points", mp.timingpoints)
# print()
# print("Colours", mp.colours)
# print()
# print("HitObjects", mp.hitobjects)
# print()