import numpy as np


def convertToDouble(val):
    x = val
    if (min(x) == -1):
        nx = (x == -1)
        x = [nx, x * (1 - nx) / np.mean(x * (1 - nx))]
    else:
        x = x / np.mean(x)

    return x
