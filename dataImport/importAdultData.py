import numpy as np
import pandas
import random

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def importAdultData():
    # incomeData = dataset('file', '+dataImport/adult.csv', 'ReadVarNames', false, 'Delimiter', ',');
    incomeData = pandas.read_csv('./adult.csv', hedaer=None, delimeter=',')

    # y = ones(size(incomeData, 1), 1);

    y = convertToDouble(incomeData[:, 12])  # 0 or 1

    # seems unnecessary (y only contains 0 or 1)
    '''
    for i in range(1, len(y)):
        if y[i] == 0:
            y[i] = 0
        else:
            y[i] = 1
    '''

    # females are sensitive
    # sensitive = str(incomeData[:, 7]) == 'Female' # old pytthon version
    sensitive = [True if str(incomeData[i, 7]) == 'Female' else False for i in range(incomeData)]

    x = [
        convertToDouble(incomeData[:, 0]),
        convertToValues(incomeData[:, 1]),
        convertToValues(incomeData[:, 2]),
        convertToValues(incomeData[:, 3]),
        convertToValues(incomeData[:, 4]),
        convertToValues(incomeData[:, 5]),
        convertToValues(incomeData[:, 6]),
        convertToValues(incomeData[:, 7]),
        convertToDouble(incomeData[:, 8]),
        convertToDouble(incomeData[:, 9]),
        convertToDouble(incomeData[:, 10]),
        convertToValues(incomeData[:, 11])]

    # generate training and test data
    # training = randsample(1:length(y), round(length(y) * 0.667));
    training = random.sample(np.arange(0, np.size(y, 0)), round(np.size(y, 0) * 0.667))
    test = np.setdiff1d(np.arange(0, np.size(y, 0)), training)

    return x, y, sensitive, training, test
