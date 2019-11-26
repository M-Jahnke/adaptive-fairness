import numpy as np
import pandas

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def importBankData():
    #data = dataset('File', '+dataImport/bank-full.csv', 'ReadVarNames', True, 'Delimiter', ',')
    data = pandas.read_csv('./bank-full.csv', delimeter=',', header=0)
    # data = dataset('File', '+dataImport/bank.csv', 'ReadVarNames', true, 'Delimiter', ';')
    # testSplitPoint = size(data, 1)
    # dataTest = dataset('File', 'bank.csv', 'ReadVarNames', true, 'Delimiter', ';')
    # data = cat(1, data, dataTest)

    y = np.ones(len(data), 1)
    #for i=1:length(y)
    for i in range(1, len(y)):
        if (str(data(i, 17) == 'no') == 1):
            y[i] = 0

    #sensitive = strcmp(cellstr(data(:, 3)), 'married') == 1; # females are sensitive
    sensitive = str(data[:, 3]) == 'married' #females are sensitie

    x = [
        convertToDouble(data[:, 1]),
        convertToValues(data[:, 2]),
        convertToValues(data[:, 3]),
        convertToValues(data[:, 4]),
        convertToValues(data[:, 5]),
        convertToDouble(data[:, 6]),
        convertToValues(data[:, 7]),
        convertToValues(data[:, 8]),
        convertToValues(data[:, 9]),
        convertToDouble(data[:, 10]),
        convertToValues(data[:, 11]),
        convertToDouble(data[:, 12]),
        convertToDouble(data[:, 13]),
        convertToDouble(data[:, 14]),
        convertToDouble(data[:, 15]),
        convertToValues(data[:, 16])
    ]

    # generate training and test data
    #training = randsample(1:len(y), round(len(y) * 0.667))
    training = np.random.standard_normal(len(y * 0.667))
    test = np.setdiff1d(np.arange(1, len(y)), training)

    return [x, y, sensitive, training, test]
