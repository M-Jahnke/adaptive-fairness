import numpy as np


def convertToValues(data):
    # convert to integer values
    # data = dataset2cell(data)
    data = data.tolist()
    classes = np.unique(data)
    # mapObj = containers.Map(classes,1:size(classes,1))
    mapObj = list(zip(classes, np.arange(0, np.size(classes, 0))))
    # for i=2:size(data,1)
    x = None
    for i in range(1, np.size(data, 0)):
        # x[i - 1] = mapObj[char(data[i])]
        x[i - 1] = list(
            map(str, data[i]))  # could shrink the dimensionality from data[i] to 1 if data is multi-dimensional array

    # convert to binary
    x = de2bi(x)
    return x


def de2bi(d):
    d = d.flatten(1)
    # m = numel(d)
    m = np.size(d)
    n = np.floor(np.log(max(d)) / np.log(2) + 1)

    b = np.zeros(m, n)
    # for i=1:m
    for i in range(0, m):
        # b(i,:) = bitget(d(i), 1: n);
        bit_rep = np.binary_repr(d[i], n)
        b[i, :] = [int(d) for d in bit_rep]  # maybe n, n-1, n-2, ..., 1 instead of 1, 2, 3, ..., n
    return b
