import numpy as np


def euclidean_reward(choice_index, points):
    p = points[choice_index]

    avg_dist = 0
    max_dist = 0
    for i, p2 in enumerate(points):
        if i != choice_index:
            dist = np.linalg.norm(p - p2)
            max_dist = max(dist, max_dist)
            avg_dist += dist

    return (avg_dist / len(points))/max_dist


def max_point(choice_index, points):
    r = points[choice_index] / np.max(points)
    return r


def binary_reward(choice_index, points):
    return points[choice_index] % 10 == 0
