import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = []

        exist_path = os.path.exists('./data/model/dq')
        if not exist_path:
            self.model = self._build_model()            
            self.model.save('./data/model/dq')
        else:
            self.model = keras.models.load_model("./data/model/dq")

    def _build_model(self):
        # Define the input and output sizes
        input_size = self.state_size
        output_size = self.action_size

        # Define the neural network architecture
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(input_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(output_size, activation='linear')
        ])

        # Define the optimizer and loss function
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        loss_fn = 'mse' #tf.keras.losses.BinaryCrossentropy()

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

    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             next_state = np.expand_dims(next_state, axis=0)
    #             q_values = self.model.predict(next_state)[0]

    #             target = reward + self.gamma * np.amax(q_values)
    #         print(state)
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        # Get a batch of experiences from the replay memory
        minibatch = random.sample(self.memory, batch_size)

        # Extract the state, action, reward, next_state, and done flags from the minibatch
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Predict the Q-values for the current states using the agent's neural network
        q_values = self.model.predict(states)

        # Predict the Q-values for the next states using the agent's target network
        target_q_values = self.model.predict(next_states)

        # Compute the target Q-values using the Bellman equation
        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(target_q_values[i])

        # Train the agent's neural network on the updated Q-values
        self.model.fit(states, q_values, epochs=1, verbose=0)

        # Update the agent's epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self):
        # Save model:
        self.model.save('./data/model/neural_network')
