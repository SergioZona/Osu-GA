import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create a simple model
model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(10,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Generate some random training data
num_samples = 100
input_data = np.random.rand(num_samples, 10)
target_data = np.random.randint(0, 2, size=(num_samples, 1))

# Training loop
batch_size = 32
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    num_batches = num_samples // batch_size
    
    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size
        
        # Get a batch of training data
        input_batch = input_data[start:end]
        print(input_batch)

        target_batch = target_data[start:end]
        print(target_batch)

        
        # Train the model on the batch
        loss = model.train_on_batch(input_batch, target_batch)
        
        print(f"Batch {batch + 1}/{num_batches} - Loss: {loss}")

# After training, you can use the model for inference
# For example, predict on some new data
new_data = np.random.rand(5, 10)
predictions = model.predict(new_data)
print("Predictions:")
print(predictions)