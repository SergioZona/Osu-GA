import tensorflow as tf
from tensorflow import keras

import cv2
import numpy as np



# # Check if there is a saved model to continue training
# ckpt = tf.train.get_checkpoint_state('./data/model')
# if ckpt and ckpt.model_checkpoint_path:
#     print('Loading model: {}'.format(ckpt.model_checkpoint_path))
#     model.load_weights(ckpt.model_checkpoint_path)

# # Train the model in real-time with OpenCV
# cap = cv2.VideoCapture(0)

# # Create a saver object to save the model
# saver = tf.compat.v1.train.Saver()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Preprocess the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=100)
#     if circles is not None:
#         x, y, r = int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])
#         x_norm = x / frame.shape[1]
#         y_norm = y / frame.shape[0]
#         r_norm1 = r / frame.shape[1]
#         r_norm2 = r / frame.shape[0]
#         # Predict if the circle should be pressed or not
#         inputs = np.array([x_norm, y_norm, r_norm1, r_norm2])
#         inputs = np.expand_dims(inputs, axis=0)
#         prediction = model.predict(inputs)[0][0]
#         if prediction > 0.5:
#             # Press the circle
#             pass
#         else:
#             # Do not press the circle
#             pass

#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         # Save the model before exiting the program
#         save_path = saver.save(tf.get_default_session(), './data/model/model.ckpt')
#         print("Model saved in path: %s" % save_path)
#         break

# Define a simple sequential model
def create_model():
    # Define the input and output sizes
    input_size = 4
    output_size = 1

    # Define the neural network architecture
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(input_size,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_size, activation='sigmoid')
    ])

    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# Save model:
model.save('./data/model/neural_network')

# # Release the capture and close the window
# cap.release()
# cv2.destroyAllWindows()
