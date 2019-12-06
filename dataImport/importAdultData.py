import numpy as np
import pandas
import random

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def importAdultData():
    incomeData = pandas.read_csv('dataImport/adult.csv', header=None)

    # y = ones(size(incomeData, 1), 1);

    y = convertToDouble(incomeData.iloc[:, 12])  # 0 or 1


    # seems unnecessary (y only contains 0 or 1)
    '''
    for i in range(1, len(y)):
        if y[i] == 0:
            y[i] = 0
        else:
            y[i] = 1
    '''

    # females are sensitive
    sensitive = [True if incomeData.iloc[i, 7] == 'Female' else False for i in range(0, np.size(incomeData, 0))]

    x = np.block([convertToDouble(incomeData.iloc[:, 0]).reshape((np.size(incomeData, 0), 1)), convertToValues(incomeData.iloc[:, 1])])
    x = np.block([x, convertToValues(incomeData.iloc[:, 2])])
    x = np.block([x, convertToValues(incomeData.iloc[:, 3])])
    x = np.block([x, convertToValues(incomeData.iloc[:, 4])])
    x = np.block([x, convertToValues(incomeData.iloc[:, 5])])
    x = np.block([x, convertToValues(incomeData.iloc[:, 6])])
    x = np.block([x, convertToValues(incomeData.iloc[:, 7])])
    x = np.block([x, convertToDouble(incomeData.iloc[:, 8]).reshape((np.size(incomeData, 0), 1))])
    x = np.block([x, convertToDouble(incomeData.iloc[:, 9]).reshape((np.size(incomeData, 0), 1))])
    x = np.block([x, convertToDouble(incomeData.iloc[:, 10]).reshape((np.size(incomeData, 0), 1))])
    x = np.block([x, convertToValues(incomeData.iloc[:, 11])])

    # generate training and test data
    training = random.sample(tuple(np.arange(0, np.size(y, 0))), round(np.size(y, 0) * 0.667))
    test = np.setdiff1d(tuple(np.arange(0, np.size(y, 0))), training)

    sensitive = np.asarray(sensitive)
    training = np.asarray(training)
    test = np.asarray(test)

    return x, y, sensitive, training, test
