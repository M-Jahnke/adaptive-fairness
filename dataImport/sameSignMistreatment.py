import numpy as np
import random


def sameSignMistreatment(L=2500):
    x = np.block([[np.random.multivariate_normal([1, 2], [[5, 2], [2, 5]], L)],
                  [np.random.multivariate_normal([2, 3], [[10, 1], [1, 4]], L)]])
    x = np.block([[x], [np.random.multivariate_normal([0, -1], [[7, 1], [1, 7]], L)]])
    x = np.block([[x], [np.random.multivariate_normal([-5, 0], [[5, 1], [1, 5]], L)]])

    sensitive = np.block([[np.zeros((L, 1), dtype=bool)], [np.ones((L, 1), dtype=bool)], [np.zeros((L, 1), dtype=bool)],
                          [np.ones((L, 1), dtype=bool)]])

    y = np.block([[np.ones((L, 1))], [np.ones((L, 1))], [np.zeros((L, 1))], [np.zeros((L, 1))]])

    training = random.sample(tuple(np.arange(0, np.size(y, 0))), 2 * L)
    test = np.setdiff1d(np.arange(0, np.size(y, 0)), training)

    training = np.asarray(training)

    return x, y, sensitive, training, test
