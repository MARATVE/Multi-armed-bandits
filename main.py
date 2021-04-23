import random
import numpy as np
import exp3
import exp3m
import reward_functions as r
import matplotlib.pyplot as plt

numbers = np.random.choice(range(1000), 100)

bandit = exp3.exp3_efficient_bandit(
    numbers, reward_function=r.max_point)

regrets = []
for i in range(10000):
    bandit.draw()
    regrets.append(bandit.give_reward(numbers))


plt.plot(range(10000), np.cumsum(regrets))
plt.savefig("regret.png")
