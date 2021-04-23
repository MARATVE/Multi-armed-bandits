import math
import random
import numpy as np
import random
import efficient_heap as heap
import logging

# Adapted from Jeremy Kuns implementation
# https://github.com/j2kun/exp3/blob/main/exp3.py


class exp3_efficient_bandit(object):
    def __init__(self, choices, reward_min=0, reward_max=1, gamma=0.07, reward_function=None, model_path=None):
        if model_path is None:
            self.weights = np.full(len(choices), 1/len(choices))
        else:
            self.weights = np.load(model_path)

        self.distribution = heap.sumheap(self.weights)

        self.reward_min = reward_min
        self.reward_max = reward_max
        self.choice = 0
        self.gamma = gamma
        self.reward_function = reward_function

    def save_model(self, model_path):
        np.save(model_path, self.weights)

    def draw(self):
        self.choice = heap.hsample(self.distribution)
        return self.choice

    def give_reward(self, reward_data):
        reward = self.reward_function(self.choice, reward_data)

        
        scaled_reward = (reward - self.reward_min) / \
            (self.reward_max - self.reward_min)

        offset = len(self.distribution)//2

        estimated_reward = 1.0 * scaled_reward / \
            (self.distribution[offset + self.choice])

        # If using negative, extract original value for proability updates
        self.weights[self.choice] = self.weights[self.choice] * math.exp(estimated_reward *
                                                                         self.gamma / len(self.weights))

        heap.update(self.distribution, self.choice, self.weights[self.choice])
        regret = self.reward_max - reward
        return regret
