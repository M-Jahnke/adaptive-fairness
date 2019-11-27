import numpy as np
import pandas

from dataImport.convertToDouble import convertToDouble
from dataImport.convertToValues import convertToValues


def importKDD():
    # incomeData = dataset('file', '+dataImport/kdd.csv', 'ReadVarNames', true, 'Delimiter', ',');
    income_data = pandas.read_csv('./kdd.csv', delimiter=',', header=0)

    # y_temp = dataImport.convertToDouble(incomeData(3:end, 41));
    y_temp = convertToDouble(income_data[2:, 40])
    y = np.ones(np.size(y_temp, 0), 1)

    x = [
        convertToDouble(income_data[2:, 0]),
        convertToValues(income_data[2:, 1]),
        convertToDouble(income_data[2:, 2]),
        convertToDouble(income_data[2:, 3]),
        convertToValues(income_data[2:, 4]),
        convertToDouble(income_data[2:, 5]),
        convertToValues(income_data[2:, 6]),
        convertToValues(income_data[2:, 7]),
        convertToValues(income_data[2:, 8]),
        convertToValues(income_data[2:, 9]),
        convertToValues(income_data[2:, 10]),
        convertToValues(income_data[2:, 11]),
        convertToValues(income_data[2:, 12]),
        convertToValues(income_data[2:, 13]),
        convertToValues(income_data[2:, 14]),
        convertToValues(income_data[2:, 15]),
        convertToDouble(income_data[2:, 16]),
        convertToDouble(income_data[2:, 17]),
        convertToDouble(income_data[2:, 18]),
        convertToValues(income_data[2:, 19]),
        convertToValues(income_data[2:, 20]),
        convertToValues(income_data[2:, 21]),
        convertToValues(income_data[2:, 22]),
        convertToValues(income_data[2:, 23]),
        convertToValues(income_data[2:, 24]),
        convertToValues(income_data[2:, 25]),
        convertToValues(income_data[2:, 26]),
        convertToValues(income_data[2:, 27]),
        convertToValues(income_data[2:, 28]),
        convertToDouble(income_data[2:, 29]),
        convertToValues(income_data[2:, 30]),
        convertToValues(income_data[2:, 31]),
        convertToValues(income_data[2:, 32]),
        convertToValues(income_data[2:, 33]),
        convertToValues(income_data[2:, 34]),
        convertToDouble(income_data[2:, 35]),
        convertToValues(income_data[2:, 36]),
        convertToDouble(income_data[2:, 37]),
        convertToDouble(income_data[2:, 38]),
        convertToDouble(income_data[2:, 39])]

    # for i=1:length(y_temp)
    for i in range(0, len(y_temp)):
        if y_temp(i) == 0:
            y[i] = 0
        else:
            y[i] = 1

    # females are sensitive
    # sensitive = strcmp(cellstr(incomeData(3:end, 13)), 'Female') == 1; #MATLAB
    # sensitive = str(income_data[3:, 13]) ==  'Female' # Python
    sensitive = [True if income_data[2:i] == 'Female' else False for i in range(3, len(income_data))]

    # generate training and test data
    # training = randsample(1:length(y), round(length(y) * 0.667));
    training = np.random.standard_normal(round(len(y) * 0.667))
    test = np.setdiff1d(np.arange(1, len(y) + 1), training)

    return x, y, sensitive, training, test
