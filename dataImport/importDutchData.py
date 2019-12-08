import numpy as np
import pandas
import random

from dataImport.convertToDouble import convertToDouble


def importDutchData():
    data = pandas.read_csv('dataImport/dutch.csv', header=0)

    y = np.ones((np.size(data, 0), 1))
    for i in range(0, np.size(y, 0)):
        if (str(data.iloc[i, 11]) == '0'):
            y[i] = 0

    sensitive = [True if str(data.iloc[i, 0]) == '0' else False for i in range(np.size(data, 0))]

    x = np.block([convertToDouble(data.iloc[:, 0]).reshape((np.size(data, 0), 1)), convertToDouble(data.iloc[:, 1]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 2]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 3]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 4]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 5]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 6]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 7]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 8]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 9]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 10]).reshape((np.size(data, 0), 1))])

    # generate training and test data
    training = random.sample(tuple(np.arange(0, np.size(y, 0))), round(np.size(y, 0) * 0.5))
    test = np.setdiff1d(np.arange(0, np.size(y, 0)), training)

    sensitive = np.asarray(sensitive)
    training = np.asarray(training)
    test = np.asarray(test)

    return x, y, sensitive, training, test
