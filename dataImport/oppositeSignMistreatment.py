import numpy as np
import random


def oppositeSignMistreatment(L=2500):
    x = [
        np.random.multivariate_normal([2, 0], [[5, 1], [1, 5]], L),
        np.random.multivariate_normal([2, 3], [[5, 1], [1, 5]], L),
        np.random.multivariate_normal([-1, -3], [[5, 1], [1, 5]], L),
        np.random.multivariate_normal([-1, 0], [[5, 1], [1, 5]], L)
    ]

    sensitive = [np.zeros(L, 1), np.ones(L, 1), np.zeros(L, 1), np.ones(L, 1)] == 1

    y = [np.ones(L, 1), np.ones(L, 1), np.zeros(L, 1), np.zeros(L, 1)]

    # training = randsample(1:length(y), 2 * L);
    training = random.sample(np.arange(0, np.size(y, 0)), 2 * L)
    test = np.setdiff1d(np.arange(0, np.size(y, 0)), training)

    return x, y, sensitive, training, test
