import numpy as np


def sameSignMistreatment(L=2500):
    x = [
        np.random.multivariate_normal([1, 2], [[5, 2], [2, 5]], L),
        np.random.multivariate_normal([2, 3], [[10, 1], [1, 4]], L),
        np.random.multivariate_normal([0, -1], [[7, 1], [1, 7]], L),
        np.random.multivariate_normal([-5, 0], [[5, 1], [1, 5]], L)
    ]

    sensitive = [np.zeros(L, 1), np.ones(L, 1), np.zeros(L, 1), np.ones(L, 1)] == 1
    y = [np.ones(L, 1), np.ones(L, 1), np.zeros(L, 1), np.zeros(L, 1)]

    # training = randsample(1:length(y), 2*L);
    training = np.random.standard_normal(2 * L)
    test = np.setdiff1d(np.arange(1, len(y) + 1), training)

    return [x, y, sensitive, training, test]
