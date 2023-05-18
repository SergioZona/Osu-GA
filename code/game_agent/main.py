import cv2 as cv
import numpy as np
import pandas as pd
import os
from time import time, sleep
from windowcapture import WindowCapture
from vision import Vision
import win32api
import win32con

import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences

import pyautogui
pyautogui.PAUSE = 0
import keyboard
import sched

import win32gui

from game_agent import GameAgent
from map_parser import OsuMapParser
from preprocessing import HitObjects
import multiprocessing 
import threading


"""--------------------"""
"""-AUXILIAR FUNCTIONS-"""
"""--------------------"""

def preprocess_circles(data, max_num_lists, max_list_length):
    # Limit the number of lists
    data = data[:max_num_lists]
    
    # Pad the sequences
    padded_data = pad_sequences(data, maxlen=max_list_length, padding='post')
    
    # Pad the number of sequences if necessary
    if len(padded_data) < max_num_lists:
        num_missing = max_num_lists - len(padded_data)
        padded_data = np.concatenate((padded_data, np.zeros((num_missing, max_list_length))))
    
    return padded_data


def find_window_startswith(window_name):
    def callback(handle, data):
        if win32gui.IsWindowVisible(handle) and window_name.lower() in win32gui.GetWindowText(handle).lower():
            data.append(win32gui.GetWindowText(handle))
        return True
    windows = []
    win32gui.EnumWindows(callback, windows)
    windows = [w for w in windows if w.lower().startswith(window_name.lower())]
    if windows:                      
        return windows[0]
    else:
        return None

"""---------------------"""
"""------MAIN CODE------"""
"""---------------------"""

# Change the working directoqry to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub

# Define the input and output sizes
input_size = 9
output_size = 4

# Create an instance of the DQNAgent
agent = GameAgent(input_size, output_size)
scheduler = sched.scheduler(time, sleep)

# initialize the WindowCapture class
wincap = WindowCapture('osu!')
# initialize the Vision class
circles_v = Vision('../data/images/circle.jpg')
errors_v = Vision('../data/images/error.jpg')
score_50_v = Vision('../data/images/score_50.jpg')
score_100_v = Vision('../data/images/score_100.jpg')
start_v = Vision('../data/images/start.jpg')

# Variables
window_name = find_window_startswith('osu!')
hwnd = win32gui.FindWindow(None, window_name)
x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)
window_w = x1-x0
window_h = y1-y0

# Thanks to: https://www.reddit.com/r/osugame/comments/vvua1l/osupixels_to_normal_coordinates/

"""The playfield left position formula can be written as:
Playfield.Left = (ScreenResolution.Width / 2) - (Playfield.Width / 2)
substituting Playfield.Width:
Playfield.Left = (ScreenResolution.Width / 2) - (((4/3) * Playfield.Height) / 2)
Since the height of the playfield is already calculated as Playfield.Height = ScreenResolution.Height * 0.8, we can substitute that as well:
Playfield.Left = (ScreenResolution.Width / 2) - (((4/3) * ScreenResolution.Height * 0.8) / 2)
which simplifies to:
Playfield.Left = (ScreenResolution.Width / 2) - ((2/3) * ScreenResolution.Height)
Therefore, the playfield left is calculated in this way to ensure that the playfield is centered on the screen horizontally, while taking into account the aspect ratio of 4:3 and the height of the playfield being 80% of the screen resolution height.
"""

playfield_w = window_w / 512
playfield_h = window_h / 384

osu_scale = playfield_h / 384

playfield_left = (window_w / 2) - ((2/3) * window_h)
playfield_top = 0.02 * playfield_h

def get_coords(hitObject_x, hitObject_y, screen_width = 800, screen_height =600):
    playfield_height = 0.8 * screen_height
    playfield_width = (4 / 3) * playfield_height
    
    playfield_x = (screen_width - playfield_width) / 2 + (0.01 * playfield_width)
    playfield_y = (screen_height - playfield_height) / 2 + (0.08 * playfield_height)
    
    osu_scale = playfield_height / 384
    
    mapped_x = (hitObject_x * osu_scale) + playfield_x + x0
    mapped_y = (hitObject_y * osu_scale) + playfield_y + y0
    
    return mapped_x, mapped_y

started = False
mp = None
data = None
df_data = None
song_time = None
actions = []
charged = False
df_length = 0
row = 0

def charge_df():
    charge_time = time()
    song_name = str(find_window_startswith(window_name))
    print("Entro")

    prueba = time() - charge_time
    sleep(0.5)
    pyautogui.keyDown('esc')
    sleep(1)
    # Release the "ESC" key
    pyautogui.keyUp('esc')
    sleep(0.5)        

    if song_name != '':
        mp = OsuMapParser(song_name)

        hit_objects = mp.get_info().hitobjects
        timing_points = list(mp.get_info().timingpoints.items())
        slider_multiplier = mp.get_info().difficulty['SliderMultiplier']
        data = HitObjects(hit_objects=hit_objects, timing_points=timing_points, slider_multiplier=slider_multiplier)
        df_data = data.df_final
    
        df_data['prediction'] = df_data.apply(lambda row: agent.act([row['x'], row['y'], row['time'], row['type'], row['length'], row['repetitions'], row['spinner_time'], row['beat_length'], row['slider_multiplier']]), axis=1)
        # Assuming your DataFrame is called "df"
        df_data[['x_pred', 'y_pred', 'slider_time', 'delay_time']] = df_data['prediction'].apply(lambda x: pd.Series(x))

        # Remove the original "prediction" column
        df_data.drop('prediction', axis=1, inplace=True)
        print(df_data.to_string())
    
    return df_data   

# def execute_schedule():
#     scheduler.run()

# thread = threading.Thread(target=execute_schedule)

# execute_scheduler = Process(target=test)

# Create a lock to synchronize access to pyautogui functions
pyautogui_lock = threading.Lock()

# Function to perform the click action
def perform_action(x_coord, y_coord):
    with pyautogui_lock:
        pyautogui.click(x_coord, y_coord)

while(True):
    loop_time = time()

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # Get errors  
    errors = errors_v.find(screenshot, 0.80, 'point', gpu=True)
    score_50 = score_50_v.find(screenshot, 0.60, 'point', gpu=True)
    score_100 = score_100_v.find(screenshot, 0.60, 'point', gpu=True)
    start = start_v.find(screenshot, 0.40, 'point', gpu=True)

    if len(start) > 0 and not charged:
        if not started:
            df_data = charge_df()
            df_length = df_data.shape[0]
            charged = True
            started=True

            row = 0
            actions = []

            for index, row in df_data.iterrows():
                x = row['x']
                y = row['y']
                time_action = row['time']
                time_to_next_action = row['time_to_next_action']
                type = row['type']
                length = row['length']
                repetitions = row['repetitions']
                spinner_time = row['spinner_time']
                beat_length = row['beat_length']
                slider_multiplier = row['slider_multiplier']
                extras = row['extras']

                x_pred = row['x_pred']
                y_pred = row['y_pred']

                x_coord, y_coord = get_coords(x,y)
                time_action = time_action/1000 
                print(x_coord, y_coord)

                # Circles
                if type==1 or type==5:
                    actions.append((time_action, x_coord, y_coord))                  
                    #print("Accion circles")
                # Sliders
                elif type==2 or type==6:
                    actions.append((time_action, x_coord, y_coord))                  
                    #print("Accion sliders")
                # Spinner
                elif type==12:
                    actions.append((time_action, x_coord, y_coord))                  
                    #print("Accion spinner")
            
            print("Acciones agregadas")
            sleep(1)
            # Replay all
            pyautogui.click((window_w/2) + x0, (window_h/2)+ y0 + 50)
            
            # Start the process if it's not already running
            # Create process objects for each program
            song_time = time()
            for action in actions:
                time_action, x_coord, y_coord = action
                # Variable delay that depend on the prediction.
                thread = threading.Timer(time_action+0.8, perform_action, args=(x_coord, y_coord))
                thread.start()
        
    # elif started:

    # elif len(start) == 0:
    #     started = False
    #     mp = None
    #     data = None
    #     df_data = None
    #     song_time = None
        

    # if click_action == 1:
    #     pyautogui.moveTo(x_coord + x0, y_coord + y0)
    #     #pyautogui.dragTo(x_coord + x0, y_coord + y0, duration=duration, button="left")
    #     pyautogui.click()

    # if len(errors) > 0:
    #     reward = len(errors)*0
    
    # if len(score_50) > 0:
    #     reward = len(score_50)*1
    
    # if len(score_100) > 0:
    #     reward = len(score_100)*2

    #print('Detecting figure: {}'.format(time() - loop_time))

    # debug the loop rate

    # print('FPS {}'.format(1 / (time() - loop_time)))


    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(100) == ord('q'):
        # Save the weights of the trained model
        done = True
        # train the agent on the episode data
        # Cancel events based on a condition
        # for event in scheduler.queue:
        #     # Check the condition to cancel the event
        #     scheduler.cancel(event)

        cv.destroyAllWindows()
        break

print('Done.')