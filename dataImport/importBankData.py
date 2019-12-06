import numpy as np
import pandas
import random

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def importBankData():
    data = pandas.read_csv('dataImport/bank-full.csv', header=0)

    y = np.ones((np.size(data, 0), 1))
    for i in range(0, np.size(y, 0)):
        if (str(data.iloc[i, 16]) == 'no'):
            y[i] = 0

    sensitive = [True if data.iloc[i, 2] == 'married' else False for i in range(0, np.size(data, 0))]

    x = np.block([convertToDouble(data.iloc[:, 0]).reshape((np.size(data, 0), 1)), convertToValues(data.iloc[:, 1])])
    x = np.block([x, convertToValues(data.iloc[:, 2])])
    x = np.block([x, convertToValues(data.iloc[:, 3])])
    x = np.block([x, convertToValues(data.iloc[:, 4])])
    x = np.block([x, convertToDouble(data.iloc[:, 5]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToValues(data.iloc[:, 6])])
    x = np.block([x, convertToValues(data.iloc[:, 7])])
    x = np.block([x, convertToValues(data.iloc[:, 8])])
    x = np.block([x, convertToDouble(data.iloc[:, 9]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToValues(data.iloc[:, 10])])
    x = np.block([x, convertToDouble(data.iloc[:, 11]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 12]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToDouble(data.iloc[:, 13]).T])
    x = np.block([x, convertToDouble(data.iloc[:, 14]).reshape((np.size(data, 0), 1))])
    x = np.block([x, convertToValues(data.iloc[:, 15])])

    # generate training and test data
    training = random.sample(tuple(np.arange(0, np.size(y, 0))), round(np.size(y, 0) * 0.667))
    test = np.setdiff1d(tuple(np.arange(0, np.size(y, 0))), training)

    sensitive = np.asarray(sensitive)
    training = np.asarray(training)
    test = np.asarray(test)

    return x, y, sensitive, training, test
