import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = []
        self.model = self._build_model()

    def _build_model(self):
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
