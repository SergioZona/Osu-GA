import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WindowCapture('osu!')
# initialize the Vision class
points_v = Vision('./data/circulo.jpg')
errors_v = Vision('./data/fallo.jpg')

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # display the processed image    
    points = points_v.find(screenshot, 0.75, 'cross', gpu=True)
    errors = errors_v.find(screenshot, 0.75, 'cross', gpu=True)
    #points = vision.find(screenshot, 0.7, 'points', gpu=True)

    #print('Detecting figure: {}'.format(time() - loop_time))

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(100) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')