import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os
import traceback
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, memory_size=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)

        exist_path = os.path.exists('./data/model/dqn_agent')
        if not exist_path:
            self.model = self._build_model()   
            self.target_model = self._build_model()         
            self.model.save('./data/model/dqn_agent')
            self.target_model.save('./data/model/dqn_agent_target')
            np.save('./data/model/epsilon.npy', (self.epsilon,))

        else:
            self.model = keras.models.load_model("./data/model/dqn_agent")
            self.target_model = keras.models.load_model("./data/model/dqn_agent_target")
            self.epsilon = np.load('./data/model/epsilon.npy')[0]
        
        print(self.epsilon)

    def _build_model(self):
        # Define the input and output sizes
        input_size = self.state_size # (50,3)
        output_size = self.action_size

        # Define the neural network architecture
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=input_size),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(output_size, activation='linear')
        ])

        # Define the optimizer and loss function
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        loss_fn = 'mse' #tf.keras.losses.BinaryCrossentropy()

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss_fn)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if len(state) == 0 or np.random.rand() <= self.epsilon:
            x_coord = np.random.randint(0, 800)
            y_coord = np.random.randint(0, 600)
            click_action = np.random.randint(0, 2)
            duration = np.random.uniform(0.0, 1.0)
            return (x_coord, y_coord, click_action, duration)
        else:       
            act_values = self.model.predict(np.array([state]))
            x_coord = act_values[0][0]
            y_coord = act_values[0][1]
            click_action = int(act_values[0][2] > 0)
            duration = act_values[0][3]
            return (x_coord, y_coord, click_action, duration)

    def replay(self, batch_size):
        try:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:

                # Length verification because state can be empty
                if len(state) != 0:
                    target = reward
                    if not done:
                        target = reward + self.gamma * np.amax(self.target_model.predict(np.array([next_state]))[0])
                    
                    target_f = self.model.predict(np.array([state]))
                    target_f[0][0] = target if action[2] == 0 else target_f[0][0]
                    target_f[0][1] = target if action[2] == 0 else target_f[0][1]
                    target_f[0][2] = action[2]
                    target_f[0][3] = target if action[2] == 1 else target_f[0][3]
                    self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

                    # Note that the click_action is now included in the target values, 
                    # and its value is set to action[2] which is either 0 or 1 depending on whether a click action was performed or not. 
                    # Also, the values of x_coord and y_coord are updated only if click_action is 0, 
                    # and the value of duration is updated only if click_action is 1.

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        except Exception:
            print("Could not train the model. Bash error.")
            traceback.print_exc()
        
    def save(self):
        # Save the model
        self.model.save('./data/model/dqn_agent')
        self.target_model.set_weights(self.model.get_weights())    
        self.target_model.save('./data/model/dqn_agent_target')

        # Saving the epsilon for further use
        self.model.save('./data/model/dqn_agent', include_optimizer=False)
        self.target_model.save('./data/model/dqn_agent_target', include_optimizer=False)
        np.save('./data/model/epsilon.npy', (self.epsilon,))