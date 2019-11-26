import numpy as np
import pandas

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def MyimportAdultData():
    #incomeData = dataset('file', '+dataImport/adult.csv', 'ReadVarNames', false, 'Delimiter', ',');
    incomeData = pandas.read_csv('./adult.csv', hedaer=None, delimeter=',')

    # y = ones(size(incomeData, 1), 1);

    y = convertToDouble(incomeData[:, 12]) # 0 or 1

    # seems unnecessary (y only contains 0 or 1)
    '''
    for i in range(1, len(y)):
        if y[i] == 0:
            y[i] = 0
        else:
            y[i] = 1
    '''

    # females are sensitive
    sensitive = str(incomeData[:, 8]) == 'Female'

    x = [
        convertToDouble(incomeData[:, 1]),
        convertToValues(incomeData[:, 2]),
        convertToValues(incomeData[:, 3]),
        convertToValues(incomeData[:, 4]),
        convertToValues(incomeData[:, 5]),
        convertToValues(incomeData[:, 6]),
        convertToValues(incomeData[:, 7]),
        convertToValues(incomeData[:, 8]),
        convertToDouble(incomeData[:, 9]),
        convertToDouble(incomeData[:, 10]),
        convertToDouble(incomeData[:, 11]),
        convertToValues(incomeData[:, 12])]

    # generate training and test data
    # training = randsample(1:length(y), round(length(y) * 0.667));
    training = np.random.standard_normal(round(len(y) * 0.667))
    test = np.setdiff1d(np.arange(1, len(y)+1), training)

    return [x, y, sensitive, training, test]
