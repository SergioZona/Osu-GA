import cv2
import numpy as np
import time

# Define the parameters of the circle
x = 200
y = 200
radius = 100
speed = 10

# Define the video capture object
cap = cv2.VideoCapture(0)

# Start a loop to capture and process video frames
while True:
    # Capture a frame from the video source
    ret, frame = cap.read()

    # Create a black image with the same size as the video frame
    mask = np.zeros_like(frame)

    # Calculate the new radius of the circle
    radius -= speed

    # If the radius is less than or equal to 0, reset it to the initial value
    if radius <= 0:
        radius = 100

    # Draw the circle on the mask
    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)

    # Convert the mask to grayscale and threshold it
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

    # Find the contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the original frame
    cv2.imshow("Frame", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()