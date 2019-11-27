import numpy as np
import pandas

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def importBankData():
    # data = dataset('File', '+dataImport/bank-full.csv', 'ReadVarNames', True, 'Delimiter', ',')
    data = pandas.read_csv('./bank-full.csv', delimeter=',', header=0)

    y = np.ones(len(data), 1)
    # for i=1:length(y)
    for i in range(0, len(y)):
        if (str(data[i, 16]) == 'no'):
            y[i] = 0

    # sensitive = strcmp(cellstr(data(:, 3)), 'married') == 1; # females are sensitive
    # sensitive = str(data[:, 2]) == 'married' # females are sensitie
    sensitive = [True if str(data[i, 2]) == 'married' else False for i in range(0, len(data))]

    x = [
        convertToDouble(data[:, 0]),
        convertToValues(data[:, 1]),
        convertToValues(data[:, 2]),
        convertToValues(data[:, 3]),
        convertToValues(data[:, 4]),
        convertToDouble(data[:, 5]),
        convertToValues(data[:, 6]),
        convertToValues(data[:, 7]),
        convertToValues(data[:, 8]),
        convertToDouble(data[:, 9]),
        convertToValues(data[:, 10]),
        convertToDouble(data[:, 11]),
        convertToDouble(data[:, 12]),
        convertToDouble(data[:, 13]),
        convertToDouble(data[:, 14]),
        convertToValues(data[:, 15])
    ]

    # generate training and test data
    # training = randsample(1:len(y), round(len(y) * 0.667))
    training = np.random.standard_normal(len(y * 0.667))
    test = np.setdiff1d(np.arange(0, len(y)), training)

    return x, y, sensitive, training, test
