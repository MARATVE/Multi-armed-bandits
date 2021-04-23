import random
import numpy as np
import exp3
import exp3m
import reward_functions as r
import matplotlib.pyplot as plt
from multiprocessing import Process

numbers = np.random.choice(range(10000), 1000)
rounds = 100000


def binary_exp3(rounds, numbers):
    regrets = []
    bandit = exp3.exp3_efficient_bandit(
        numbers, reward_function=r.binary_reward)
    for i in range(rounds):
        bandit.draw()
        regrets.append(bandit.give_reward(numbers))
        if i % int(rounds/10) == 0:
            plt.plot(np.cumsum(regrets))
            plt.title("exp3_binary")
            plt.savefig(f"plots/exp3_binary")


def binary_exp3m(rounds, numbers):
    regrets = []
    bandit = exp3m.exp3_m(numbers, reward_function=r.binary_reward)
    for i in range(rounds):
        bandit.draw(100)
        regrets.append(bandit.give_reward(numbers))
        if i % int(rounds/10) == 0:
            plt.plot(np.cumsum(regrets))
            plt.title("exp3m_binary, k = 100")
            plt.savefig(f"plots/exp3m_binary")


def max_point_exp3(rounds, numbers):
    regrets = []
    bandit = exp3.exp3_efficient_bandit(
        numbers, reward_function=r.max_point)
    for i in range(rounds):
        bandit.draw()
        regrets.append(bandit.give_reward(numbers))
        if i % int(rounds/10) == 0:
            plt.plot(np.cumsum(regrets))
            plt.title("exp3_maxpoint")
            plt.savefig(f"plots/exp3_maxpoint")


def max_point_exp3m(rounds, numbers):
    regrets = []
    bandit = exp3m.exp3_m(numbers, reward_function=r.max_point)
    for i in range(rounds):
        bandit.draw(100)
        regrets.append(bandit.give_reward(numbers))
        if i % int(rounds/10) == 0:
            plt.plot(np.cumsum(regrets))
            plt.title("exp3m_maxpoint, k = 100")
            plt.savefig(f"plots/exp3m_maxpoint")


p1 = Process(target=binary_exp3, args=(rounds, numbers,))
p2 = Process(target=binary_exp3m, args=(rounds, numbers,))
p3 = Process(target=max_point_exp3, args=(rounds, numbers,))
p4 = Process(target=max_point_exp3m, args=(rounds, numbers,))

p1.start()
#p2.start()
p3.start()
#p4.start()
