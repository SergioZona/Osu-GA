import pandas as pd
import numpy as np
import re

class HitObjects:    
   def __init__(self, hit_objects, timing_points, slider_multiplier):
      self.hit_objects = hit_objects
      self.timing_points = timing_points
      self.slider_multiplier = slider_multiplier
      self.df_beat = self.preprocess_timing_points(timing_points)
      self.df_final = self.get_final_data()
        
   def preprocess_timing_points(self, timing_points):
      # Convert list of tuples to a dictionary with time as keys and the important element as values
      dict_lst = dict((int(t), float(l[0])) for t, l in timing_points)

      # iterate over keys and values
      df_beat = pd.DataFrame.from_dict(dict_lst, orient='index', columns=['beat_length'])

      # set column 'A' as the index and change its name to 'index_name'
      # reset index and rename the new column
      df_beat.reset_index(inplace=True)
      df_beat.rename(columns={'index': 'upper_range'}, inplace=True)
      # shift the 'lower_range' column by one row and assign it to 'upper_range'
      df_beat['lower_range'] = df_beat['upper_range'].shift(1).fillna(0).apply(lambda x: int(x))
      df_beat['upper_range'] = df_beat['upper_range'].apply(lambda x: int(x))
      df_beat = df_beat[["lower_range", "upper_range", "beat_length"]]
      return df_beat
   
   def get_beat_legth(self, time):
      for row in self.df_beat.itertuples(index=False):
         row_as_tuple = tuple(row)
         lower_range = row_as_tuple[0]
         upper_range = row_as_tuple[1]
         beat_lengths = row_as_tuple[2]
         
         if (time >= lower_range) and (time < upper_range):
               return beat_lengths      
      return None

   def slider_pattern(self, text):    
      # Pattern: "some-text|coord_x:coord_y,repetitions,length,length|text-to-delete"
      # Note: Number can be negative, that is why -* is necessary.
      pattern = re.compile(r"^.*?\|\-*\d+:\-*\d+,\d+,\d+,\d+\|")
      match = pattern.search(text)
      if match:
         result = match.group(0)
         return(result) 
      
   def extras_preprocessing(self, text,obj_type):
      # Slider
      if obj_type == 1 or obj_type == 5:
         return ''
      elif obj_type == 2 or obj_type == 6:
         slider_info = self.slider_pattern(text).split('|')
         slider_points = []
         for i in slider_info:
               if (len(i) > 1) and (',' not in i):
                  coords = i.split(':')
                  x = int(coords[0])
                  y = int(coords[1])
                  slider_points.append((x,y))
               elif (len(i) > 1) and ',' in i:
                  end_info = i.split(',')

                  coords = end_info[0].split(':')
                  x = int(coords[0])
                  y = int(coords[1])

                  repetitions = int(end_info[1])
                  length = int(end_info[2])
                  slider_points.append((x,y))
                  slider_points.append(repetitions)
                  slider_points.append(length) 
         return slider_points
      elif obj_type == 12:
         return int(text)

   def get_final_data(self):
      df = pd.DataFrame(columns=["x", "y", "time", "type", "extras"])

      for hit_object in self.hit_objects:
         # print(hit_object)
         data = hit_object.split(",", maxsplit=5)
         x = int(data[0])
         y = int(data[1])
         time = float(data[2])
         obj_type = int(data[3])
         sound =  data[4]
         length = 0
         repetitions = 0
         spinner_time = 0
         
         try:
               #extras_preprocessing(x,obj_type)
               extras = data[5].rsplit(',', 1)[0]
               extras = self.extras_preprocessing(extras,obj_type)
               
               if obj_type == 2 or obj_type == 6:
                  length = float(extras.pop())
                  repetitions = extras.pop()       
               elif obj_type == 12:
                  spinner_time = float(extras)
                  extras = ''
         except:
               extras = ''
         
         df_temp = pd.DataFrame([[x,y,time, obj_type, length, repetitions, spinner_time, extras]], columns=["x", "y", "time", "type", "length", "repetitions", "spinner_time", "extras"])
         df = pd.concat([df, df_temp])
      df['x'] = df['x'].astype('int64')
      df['y'] = df['y'].astype('int64')
      df['time'] = df['time'].astype('float64')
      df['type'] = df['type'].astype('int64')
      df['repetitions'] = df['repetitions'].astype('int64')
      df['beat_length'] = df['time'].apply(lambda x: self.get_beat_legth(x)).astype('float64')
      df['spinner_time'] = df['spinner_time'].astype('float64')
      df['slider_multiplier'] = float(self.slider_multiplier)
      df['time_to_next_action'] = df['time_to_next_action'] = df['time'].shift(-1) - df['time']
      df.iloc[-1, df.columns.get_loc('time_to_next_action')] = 60000
      df = df.fillna(0)
      df = df[["x", "y", "time", "time_to_next_action", "type", "length", "repetitions", "spinner_time", "beat_length", "slider_multiplier", "extras"]]
      return df

   def prepare_input_data(data):
      # Get the relevant features from the HitObjects instance
      x = data.df_final['x'].values.reshape(-1, 1)
      y = data.df_final['y'].values.reshape(-1, 1)
      t = data.df_final['time'].values.reshape(-1, 1)
      hit_type = data.df_final['type'].values.reshape(-1, 1)
      length = data.df_final['length'].values.reshape(-1, 1)

      # Combine the features into a single numpy array
      input_data = np.concatenate([x, y, t, hit_type, length], axis=1)

      return input_data