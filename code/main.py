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

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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

# Variables
# Load model: 
model = keras.models.load_model("./data/model/neural_network")

window_name = find_window_startswith('osu!')
hwnd = win32gui.FindWindow(None, window_name)
x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)


loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # display the processed image    
    circles = circles_v.detect_circles(screenshot)
    circles_e = circles_v.detect_external_circles(screenshot)

    errors = errors_v.find(screenshot, 0.80, 'point', gpu=True)
    #points = vision.find(screenshot, 0.7, 'points', gpu=True)

    x_coord = -1
    y_coord = -1
    inner_radius = -1
    external_radius = -1
    prediction = -1

    if circles is not None:
        x_coord = circles[0][0][0]
        y_coord = circles[0][0][1]
        inner_radius = circles[0][0][2]
        external_radius = inner_radius

    if circles_e is not None:
        external_radius = circles_e[0][0][2]
    
    if circles is not None and circles_e is not None:
        inputs = np.array([x_coord, y_coord, inner_radius, external_radius])
        inputs = np.expand_dims(inputs, axis=0)
        prediction = model.predict(inputs)[0][0]

    if prediction > 0.5:
        pyautogui.moveTo(x_coord + x0, y_coord + y0)
        pyautogui.click()

    #print('Detecting figure: {}'.format(time() - loop_time))

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(100) == ord('q'):
        model.save('./data/model/neural_network')
        cv.destroyAllWindows()
        break

print('Done.')