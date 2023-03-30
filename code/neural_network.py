import tensorflow as tf
from tensorflow import keras

import cv2
import numpy as np

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
