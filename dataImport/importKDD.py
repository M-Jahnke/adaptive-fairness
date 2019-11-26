import numpy as np
import pandas

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def ImportKDD():
    # incomeData = dataset('file', '+dataImport/kdd.csv', 'ReadVarNames', true, 'Delimiter', ',');
    income_data = pandas.read_csv('./kdd.csv', delimiter=',', header=0)

    # y_temp = dataImport.convertToDouble(incomeData(3:end, 41));
    y_temp = convertToDouble(income_data[3:, 41])
    y = np.ones(np.size(y_temp, 1), 1)

    x = [
        convertToDouble(income_data[3:, 0]),
        convertToValues(income_data[3:, 1]),
        convertToDouble(income_data[3:, 2]),
        convertToDouble(income_data[3:, 3]),
        convertToValues(income_data[3:, 4]),
        convertToDouble(income_data[3:, 5]),
        convertToValues(income_data[3:, 6]),
        convertToValues(income_data[3:, 7]),
        convertToValues(income_data[3:, 8]),
        convertToValues(income_data[3:, 9]),
        convertToValues(income_data[3:, 10]),
        convertToValues(income_data[3:, 11]),
        convertToValues(income_data[3:, 12]),
        convertToValues(income_data[3:, 13]),
        convertToValues(income_data[3:, 14]),
        convertToValues(income_data[3:, 15]),
        convertToDouble(income_data[3:, 16]),
        convertToDouble(income_data[3:, 17]),
        convertToDouble(income_data[3:, 18]),
        convertToValues(income_data[3:, 19]),
        convertToValues(income_data[3:, 20]),
        convertToValues(income_data[3:, 21]),
        convertToValues(income_data[3:, 22]),
        convertToValues(income_data[3:, 23]),
        convertToValues(income_data[3:, 24]),
        convertToValues(income_data[3:, 25]),
        convertToValues(income_data[3:, 26]),
        convertToValues(income_data[3:, 27]),
        convertToValues(income_data[3:, 28]),
        convertToDouble(income_data[3:, 29]),
        convertToValues(income_data[3:, 30]),
        convertToValues(income_data[3:, 31]),
        convertToValues(income_data[3:, 32]),
        convertToValues(income_data[3:, 33]),
        convertToValues(income_data[3:, 34]),
        convertToDouble(income_data[3:, 35]),
        convertToValues(income_data[3:, 36]),
        convertToDouble(income_data[3:, 37]),
        convertToDouble(income_data[3:, 38]),
        convertToDouble(income_data[3:, 39])]

    #for i=1:length(y_temp)
    for i in range(1, len(y)):
        if y_temp(i) == 0:
            y[i] = 0
        else:
            y[i] = 1

    # females are sensitive
    # sensitive = strcmp(cellstr(incomeData(3:end, 13)), 'Female') == 1;
    sensitive = str(income_data[3:, 13]) ==  'Female'

    # generate training and test data
    # training = randsample(1:length(y), round(length(y) * 0.667));
    training = np.random.standard_normal(round(len(y) * 0.667))
    test = np.setdiff1d(np.arange(1, len(y)+1), training)

    return [x, y, sensitive, training, test]
