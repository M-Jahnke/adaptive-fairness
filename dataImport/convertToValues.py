import numpy as np


def convertToValues(data):
    # convert to integer values
    data = data.tolist()
    classes = np.unique(data).tolist()
    x = []
    for i in range(0, np.size(data, 0)):
        # x[i - 1] = mapObj[char(data[i])]
        x.append(classes.index(data[i]))  # could shrink the dimensionality from data[i] to 1 if data is multi-dimensional array

    # convert to binary
    x = np.asarray(x) # can be deleted with flatten, probably
    x = de2bi(x)
    return x


def de2bi(d):
    d = d.flatten(1)
    m = int(np.size(d)) # 45175, anzahl der Elemente in der Liste
    n = int(np.floor(np.log(max(d)) / np.log(2) + 1)) # 3, anzahl der nötigen Bits um alle Klassen  unterschiedlich zu kodieren

    b = np.empty((m, n), dtype=int)
    for i in range(0, m):
        bit_rep = np.binary_repr(d[i], n)[::-1] # reverse string, LSB!
        for j in range(0, n):
            b[i][j] = int(bit_rep[j])
    return b
