import numpy as np
from stable_baselines3 import PPO
import torch

class EpsilonGreedyPPO(PPO):
    def __init__(self, *args, epsilon=0.1, **kwargs):
        super(EpsilonGreedyPPO, self).__init__(*args, **kwargs)
        self.epsilon = epsilon  # Epsilon for exploration

    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Override the predict method to implement epsilon-greedy.
        - If random (epsilon), select a random action.
        - Otherwise, use the PPO model's policy to select the action.
        """
        print("Predict method called")  # Debugging line
        # With probability epsilon, pick a random action
        if np.random.random() < self.epsilon:
            action = self.action_space.sample()
            print('Random action chosen: ', action)
            return action, state
        else:
            print('Basic action')
            # Otherwise, use the model's policy
            return super(EpsilonGreedyPPO, self).predict(observation, state, mask, deterministic)
