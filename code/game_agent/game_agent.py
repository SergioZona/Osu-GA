import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

class GameAgent:
    def __init__(self, input_shape, output_shape, learning_rate = 0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        exist_path = os.path.exists('./data/model/game_agent')
        if not exist_path:
            self.model = self.build_model()   
            self.target_model = self.build_model()         
            self.model.save('./data/model/game_agent')
            self.target_model.save('./data/model/game_agent_target')
            np.save('./data/model/learning_rate.npy', (learning_rate,))

        else:
            self.model = keras.models.load_model("./data/model/game_agent")
            self.target_model = keras.models.load_model("./data/model/game_agent_target")
            self.learning_rate = np.load('./data/model/learning_rate.npy')[0]

    def build_model(self):
        def activation_x(x):
            return keras.backend.clip(x, 0, 800)  # Constrain values between 0 and 800

        def activation_y(x):
            return keras.backend.clip(x, 0, 600)  # Constrain values between 0 and 600

        def activation_slider_time(x):
            return keras.backend.clip(x, 0, 5)  # Constrain values between 0 and 5

        def activation_delay_time(x):
            return keras.backend.clip(x, 0, 1)  # Constrain values between 0 and 1

        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        dense1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        
        output1 = tf.keras.layers.Dense(1, activation=activation_x, name='x')(dense2)  # Output 1 between 0 and 800
        output2 = tf.keras.layers.Dense(1, activation=activation_y, name='y')(dense2)  # Output 2 between 0 and 600
        output3 = tf.keras.layers.Dense(1, activation=activation_slider_time, name='slider_time')(dense2)  # Output 3 between 0 and 5
        output4 = tf.keras.layers.Dense(1, activation=activation_delay_time, name='delay_time')(dense2)  # Output 4 between 0 and 1
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=[output1, output2, output3, output4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model
        # model = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=self.input_shape),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(32, activation='relu'),
        #     tf.keras.layers.Dense(self.output_shape, activation='linear')
        # ])
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        # return model
    
    def act(self, state):
        state = np.array(state).reshape((1, self.input_shape))
        outputs = self.model.predict(state)
        x = outputs[0][0][0]
        y = outputs[1][0][0]
        slider_time = outputs[2][0][0]
        delay_time = outputs[3][0][0]
        return [x, y, slider_time, delay_time]
    
    def train(self, states, rewards):
        # Convert states to numpy array
        states = np.array(states)

        # Normalize rewards between 0 and 1 
        normalized_rewards = np.array(rewards) / 3.0

        # Create target values with rewards as labels
        targets = normalized_rewards

        # Train the model on the states and rewards using train_on_batch
        self.model.train_on_batch(states, targets)
    
    def save(self):
        # Save the model
        self.model.save('./data/model/game_agent')
        self.target_model.set_weights(self.model.get_weights())    
        self.target_model.save('./data/model/game_agent_target')

        # Saving the epsilon for further use
        self.model.save('./data/model/game_agent', include_optimizer=False)
        self.target_model.save('./data/model/game_agent_target', include_optimizer=False)
        np.save('./data/model/epsilon.npy', (self.epsilon,))