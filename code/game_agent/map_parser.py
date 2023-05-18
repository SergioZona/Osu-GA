import os
from osu_map_parser.map_parser import MapParser

class OsuMapParser:    
    def __init__(self, song_name):
        self.song_name = song_name
        self.song_path = self.get_song_path(song_name)
        self.info = self.parse_map(self.song_path)
        
    def get_song_path(self, song_name):
        osu_songs_directory = os.path.join(os.getenv('LOCALAPPDATA'), 'osu!', 'Songs')

        dash_index = song_name.find("-")
        left_bracket_index = song_name.rfind('[')
        right_bracket_index = song_name.rfind(']')
        name = song_name[dash_index+1:left_bracket_index].strip()
        difficulty = song_name[left_bracket_index:right_bracket_index+1].strip()

        # List all the directories inside osu_songs_directory
        for directory_name in os.listdir(osu_songs_directory):
            if name in directory_name:
                # We found the directory that contains the song
                osu_songs_directory = os.path.join(osu_songs_directory, directory_name)
                        
                for song in os.listdir(osu_songs_directory):
                    if difficulty in song:
                        # We found the directory that contains the song
                        osu_songs_directory = os.path.join(osu_songs_directory, song)
                        print(osu_songs_directory)
                        return osu_songs_directory
        
        return ''
        
    def parse_map(self, path):
        # mp.general
        # mp.metadata
        # mp.difficulty
        # mp.events
        # mp.timingpoints
        # mp.hitobjects
        mp = MapParser(path)
        return mp

    def get_info(self):
        return self.info


       