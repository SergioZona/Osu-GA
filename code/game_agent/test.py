import osu_beatmap_parser as obp

# Load the beatmap file
beatmap = obp.parse_beatmap_file('path/to/beatmap.osu')

# Print some of the beatmap information
print('Version:', beatmap.format_version)
print('Audio file:', beatmap.general.audio_file)
print('Mode:', beatmap.general.mode)

# Print the timing points
for tp in beatmap.timing_points:
    print('Offset:', tp.offset, 'BPM:', tp.bpm)

# Print the hit objects
for ho in beatmap.hit_objects:
    print('Type:', ho.object_name, 'Time:', ho.time, 'Position:', ho.position)