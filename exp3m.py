import math
import random
import numpy as np
import random
import logging
import time


class exp3_m(object):
    def __init__(self, choices, reward_min=0, reward_max=1, gamma=0.07, reward_function=None, model_path=None):
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.gamma = gamma
        self.S_0 = set()
        self.probabilities = []
        self.reward_function = reward_function

        if model_path is not None:
            self.weights = np.load(model_path)
        else:
            self.weights = np.ones(len(choices))

    def depround(self, probabilities):
        one_probs = set()
        candidates = set(range(len(probabilities)))
        probs = probabilities.copy()

        # Calling np.random.uniform in the loop doubles execution time, allocate ahead of time
        randoms = set(np.random.uniform(0, 1, len(candidates)))
        # We assume that all probabilities initally are 0 < p < 1
        while len(candidates) > 1:
            i = candidates.pop()
            j = candidates.pop()

            alpha = min(1 - probs[i], probs[j])
            beta = min(probs[i], 1 - probs[j])

            threshold = randoms.pop()

            if threshold > (beta/(alpha+beta)):
                probs[i] = probs[i] + alpha
                probs[j] = probs[j] - alpha
            else:
                probs[i] = probs[i] - beta
                probs[j] = probs[j] + beta

            # Put back into pool or element has been chosen
            if probs[i] == 1:
                one_probs.add(i)
            elif probs[i] > 0:
                candidates.add(i)

            if probs[j] == 1:
                one_probs.add(j)
            elif probs[j] > 0:
                candidates.add(j)

        return np.array(list(one_probs))

    def draw(self, k):
        max_j = np.argmax(self.weights)
        K = len(self.weights)
        self.S_0 = set()
        # Step 1

        sorted_weight_indices = np.argsort(self.weights)[::-1]
        s_0_candidates = set()
        if self.weights[max_j] >= (1/k - self.gamma/K) * (np.sum(self.weights)/(1-self.gamma)):
            rhs = (1/k - self.gamma/K)/(1 - self.gamma)
            alpha_t = 0
            # Find alpha_t
            for i, index in enumerate(sorted_weight_indices):
                x = i
                y = np.sum(self.weights[sorted_weight_indices[i:]])
                alpha_t_candidate = -(y * rhs)/(x*rhs - 1)
                s_0_candidates.add(index)
                if alpha_t_candidate == rhs:
                    alpha_t = alpha_t_candidate
                    self.S_0 = s_0_candidates
                    break

        # Step 2
        W = set(sorted_weight_indices)
        weights_prime = np.zeros(K)
        diff_indices = list(W.difference(self.S_0))
        weights_prime[diff_indices] = self.weights[diff_indices]
        if len(self.S_0) > 0:
            S0_indices = list(self.S_0)
            weights_prime[S0_indices] = alpha_t

        # Step 3
        w_prime_sum = np.sum(weights_prime)
        gamma_factor = (1 - self.gamma)
        gamma_term = self.gamma/K
        self.probabilities = 1/w_prime_sum * weights_prime * gamma_factor
        self.probabilities = self.probabilities + gamma_term
        self.probabilities = self.probabilities * k

        # Step 4
        self.choices = self.depround(self.probabilities)
        return self.choices

    def give_reward(self, reward_data):
        rewards = [self.reward_function(choice, reward_data)
                   for i, choice in enumerate(self.choices)]

        regret = np.sum([self.reward_max - r for r in rewards])

        for i, reward in enumerate(rewards):
            self.update_probability(reward, i, len(self.choices))

        return regret/len(self.choices)

    def update_probability(self, reward, i, k):
        x_hat = reward/self.probabilities[i]
        if i not in self.S_0:
            self.weights[i] = self.weights[i] * \
                math.exp(k * self.gamma * x_hat/(len(self.probabilities)))
