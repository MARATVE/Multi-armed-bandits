import random
import numpy as np
import exp3
import exp3m
import reward_functions as r
import matplotlib.pyplot as plt
from multiprocessing import Process
import pandas as pd

numbers = np.random.choice(range(10000), 1000)


def write_regret(results, annotation, f_name):
    with open(f"results/{f_name}.csv", 'w+') as f:
        f.write('#' + annotation+'\n')
        f.write("round,regret\n")
        r = np.cumsum(results)
        f.write(
            '\n'.join([f"{i},{r}" for i, r in enumerate(r)]))


def binary_exp3(rounds, numbers):
    regrets = []
    bandit = exp3.exp3_efficient_bandit(
        numbers, reward_function=r.binary_reward)
    for i in range(rounds):
        bandit.draw()
        regrets.append(bandit.give_reward(numbers))

    write_regret(
        regrets, f"exp3_binary rounds = {rounds}", f"exp3_binary/{rounds}-{len(numbers)}")


def binary_exp3m(rounds, numbers):
    regrets = []
    bandit = exp3m.exp3_m(numbers, reward_function=r.binary_reward)
    for i in range(rounds):
        bandit.draw(int(rounds/10))
        regrets.append(bandit.give_reward(numbers))

    write_regret(
        regrets, f"exp3m_binary rounds = {rounds}", f"exp3m_binary/{rounds}-{len(numbers)}")


def max_point_exp3(rounds, numbers):
    regrets = []
    bandit = exp3.exp3_efficient_bandit(
        numbers, reward_function=r.max_point)
    for i in range(rounds):
        bandit.draw(int(rounds/10))
        regrets.append(bandit.give_reward(numbers))

    write_regret(
        regrets, f"exp3 maxpoint rounds = {rounds}", f"exp3_max/{rounds}-{len(numbers)}")


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
    write_regret(
        regrets, f"exp3m maxpoint rounds = {rounds}", f"exp3_max/{rounds}-{len(numbers)}")


rounds_exp3 = [1000, 1000*10, 1000*10**2, 1000*10**3, 4000*10**3]
n_exp3 = [1000, 1000*10, 1000*10**2, 1000*10**3, 4000*10**3]
rounds_exp3m = [1000, 1000 * 10]

for i, rounds in enumerate(rounds_exp3):
    numbers = np.random.choice(range(rounds * 10), n_exp3[i])
    p1 = Process(target=binary_exp3, args=(rounds, numbers,))
    p1.start()
    p2 = Process(target=max_point_exp3, args=(rounds, numbers,))
    p2.start()

for i, rounds in enumerate(rounds_exp3):
    numbers = np.random.choice(range(rounds * 10), rounds)
    p1 = Process(target=binary_exp3m, args=(rounds, numbers,))
    p1.start()
    p2 = Process(target=max_point_exp3m, args=(rounds, numbers,))
    p2.start()
