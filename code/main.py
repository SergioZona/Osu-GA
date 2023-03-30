import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

import tensorflow as tf
from tensorflow import keras

import pyautogui
import win32gui

from dqn_agent import DQNAgent

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the input and output sizes
input_size = 4
output_size = 2 # Two possible actions: do nothing or click the circle

# Create an instance of the DQNAgent
agent = DQNAgent(input_size, output_size)

# Load the weights of a pre-trained model (if available)
# agent.load_weights("./data/model/dq_learning/dq.h5")


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

# initialize the WindowCapture class
wincap = WindowCapture('osu!')
# initialize the Vision class
circles_v = Vision('./data/images/circle.jpg')
errors_v = Vision('./data/images/error.jpg')
score_50_v = Vision('./data/images/score_50.jpg')
score_100_v = Vision('./data/images/score_100.jpg')



# Variables
# Load model: 
model = keras.models.load_model("./data/model/neural_network")

window_name = find_window_startswith('osu!')
hwnd = win32gui.FindWindow(None, window_name)
x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)

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
    circles_e = circles_v.detect_external_circles(screenshot)

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

    # Internal circle radius
    if circles is not None:
        inner_radio = circles[0][0][2]
        external_radio = inner_radio

    # External circle radius
    if circles_e is not None:
        min_radio = float('inf')
        for circle in circles_e:
            actual_radio = circle[0][2]
            if actual_radio < min_radio:                
                x_coord = circle[0][0]
                y_coord = circle[0][1]
                external_radio = circle[0][2]    
    
    # Execution
    if circles is not None and circles_e is not None:
        next_state = np.array([x_coord, y_coord, inner_radio, external_radio])

        # Predict
        action = agent.act(state)

        if action == 1:
            pyautogui.moveTo(x_coord + x0, y_coord + y0)
            pyautogui.click()

        if len(errors) > 1:
            reward += 0
        
        if len(score_50) > 1:
            reward += 1
        
        if len(score_100) > 1:
            reward += 2
        
        state = next_state
            
        # let the agent learn from the experience
        agent.remember(state, action, reward, next_state, done)
        # # perform the action

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
            agent.replay(200)

        agent.save()
        cv.destroyAllWindows()
        break

print('Done.')