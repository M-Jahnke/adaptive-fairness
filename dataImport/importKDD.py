import numpy as np
import pandas
import random

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def importKDD():
    income_data = pandas.read_csv('dataImport/kdd.csv', header=0, na_values=['nan'])

    # cut the dataset (too large)
    income_data = income_data.sample(frac=1).reset_index(drop=True)  # shuffle
    income_data = income_data.iloc[:5000]  # keep 5000 rows, drop rest

    y_temp = convertToDouble(income_data.iloc[2:, 40])
    y = np.ones((np.size(y_temp, 0), 1))

    length = np.size(income_data.iloc[2:, 0])
    x = np.block([convertToDouble(income_data.iloc[2:, 0]).reshape((length, 1)), convertToValues(income_data.iloc[2:, 1])])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 2]).reshape((length, 1))])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 3]).reshape((length, 1))])
    x = np.block([x, convertToValues(income_data.iloc[2:, 4])])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 5]).reshape((length, 1))])
    x = np.block([x, convertToValues(income_data.iloc[2:, 6])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 7])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 8])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 9])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 10])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 11])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 12])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 13])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 14])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 15])])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 16]).reshape(length, 1)])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 17]).reshape(length, 1)])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 18]).reshape(length, 1)])
    x = np.block([x, convertToValues(income_data.iloc[2:, 19])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 20])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 21])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 22])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 23])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 24])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 25])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 26])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 27])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 28])])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 29]).reshape((length, 1))])
    x = np.block([x, convertToValues(income_data.iloc[2:, 30])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 31])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 32])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 33])])
    x = np.block([x, convertToValues(income_data.iloc[2:, 34])])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 35]).reshape((length, 1))])
    x = np.block([x, convertToValues(income_data.iloc[2:, 36])])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 37]).reshape((length, 1))])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 38]).reshape((length, 1))])
    x = np.block([x, convertToDouble(income_data.iloc[2:, 39]).reshape((length, 1))])


    for i in range(0, np.size(y_temp, 0)):
        if y_temp[i] == 0:
            y[i] = 0
        else:
            y[i] = 1

    # females are sensitive
    sensitive = [True if str(income_data.iloc[i, 12]) == 'Female' else False for i in range(2, np.size(income_data, 0))]

    # generate training and test data
    training = random.sample(tuple(np.arange(0, np.size(y, 0))), round(np.size(y, 0) * 0.667))
    test = np.setdiff1d(np.arange(0, np.size(y, 0)), training)

    sensitive = np.asarray(sensitive)
    training = np.asarray(training)
    test = np.asarray(test)

    return x, y, sensitive, training, test
