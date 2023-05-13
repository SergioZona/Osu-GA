import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences


import pyautogui
import win32gui

from dqn_agent import DQNAgent

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

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the input and output sizes
max_num_circles = 50
circles_info_length = 3
input_size = (max_num_circles, circles_info_length)
output_size = 4
# Returns 
# [x_coord_to_click, 
#  y_coord_to_click, 
#  click_action {0 -> No click, 1 -> Click_position}, 
#  dragTo_duration { Between 0 and 1}]

# Create an instance of the DQNAgent
agent = DQNAgent(input_size, output_size)

# initialize the WindowCapture class
wincap = WindowCapture('osu!')
# initialize the Vision class
circles_v = Vision('./data/images/circle.jpg')
errors_v = Vision('./data/images/error.jpg')
score_50_v = Vision('./data/images/score_50.jpg')
score_100_v = Vision('./data/images/score_100.jpg')

# Variables
window_name = find_window_startswith('osu!')
hwnd = win32gui.FindWindow(None, window_name)
x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)
w = x1-x0
h = y1-y0

i = 0
loop_time = time()
state = []
next_state = []
done = False # Set this flag to True if the game is over - Example of other flag (boolean): (i==10000)
train = True

while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # display the processed image    
    circles = circles_v.detect_circles(screenshot)
    lines = circles_v.detect_slide_notes(screenshot)

    print("Lines", lines)

    errors = errors_v.find(screenshot, 0.80, 'point', gpu=True)
    score_50 = score_50_v.find(screenshot, 0.60, 'point', gpu=True)
    score_100 = score_100_v.find(screenshot, 0.60, 'point', gpu=True)

    # TODO: Detect the other scores and give reward according to them.
    #points = vision.find(screenshot, 0.7, 'points', gpu=True)

    # We define the variables for each iteration
    x_coord = -1
    y_coord = -1
    inner_radio = -1
    external_radio = -1
    prediction = -1
    reward = 3 # Maximum reward by default - 300: 3
    action = 0
  
    # Execution
    if circles is not None: #and circles_e is not None:
        circles = preprocess_circles(circles[0], max_num_circles, circles_info_length)
        next_state = np.array(circles)

        # Predict
        # action = agent.act(state)
        x_coord, y_coord, click_action, duration = agent.act(state)
        action = [x_coord, y_coord, click_action, duration]
        #print(action)

        # # perform the action
        if click_action == 1:
           pyautogui.moveTo(x_coord + x0, y_coord + y0)
           #pyautogui.dragTo(x_coord + x0, y_coord + y0, duration=duration, button="left")
           pyautogui.click()

        if len(errors) > 0:
            reward = len(errors)*0
        
        if len(score_50) > 0:
            reward = len(score_50)*1
        
        if len(score_100) > 0:
            reward = len(score_100)*2
        
        # let the agent learn from the experience
        agent.remember(state, action, reward, next_state, done)

        # Update the current state.
        state = next_state        


    #print('Detecting figure: {}'.format(time() - loop_time))

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(100) == ord('q'):
        # Save the weights of the trained model
        done = True
        agent.remember(state, action, reward, next_state, done)
        # train the agent on the episode data

        if train:
            agent.replay(10)

        agent.save()
        cv.destroyAllWindows()
        break

print('Done.')