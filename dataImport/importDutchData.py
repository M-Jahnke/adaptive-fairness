import numpy as np
import pandas

from dataImport.convertToDouble import convertToDouble


def importDutchData():
    # data = dataset('file', '+dataImport/dutch.csv', 'ReadVarNames', true, 'Delimiter', ',');
    data = pandas.read_csv('./dutch.csv', delimeter=',', header=0)
    # data = data(2: size(data, 1),:);

    # y = ones(size(data, 1), 1);
    y = np.ones(np.size(data, 0), 1)
    # for i=1:length(y)
    for i in range(0, len(y)):
        if (str(data[i, 11]) == '0'):
            y[i] = 0

    # sensitive = strcmp(cellstr(data(:, 1)), '0') == 1; # MATLAB
    # sensitive = str(data[:, 0]) ==  '0' # Python
    sensitive = [True if str(data[i, 0]) == '0' else False for i in range(len(data))]

    x = [
        convertToDouble(data[:, 0]),
        convertToDouble(data[:, 1]),
        convertToDouble(data[:, 2]),
        convertToDouble(data[:, 3]),
        convertToDouble(data[:, 4]),
        convertToDouble(data[:, 5]),
        convertToDouble(data[:, 6]),
        convertToDouble(data[:, 7]),
        convertToDouble(data[:, 8]),
        convertToDouble(data[:, 8]),
        convertToDouble(data[:, 10])
    ]

    # size(x)# for what?

    # generate training and test data
    training = np.random.standard_normal(round(len(y) * 0.5))
    test = np.setdiff1d(np.arange(0, len(y)), training)
    return x, y, sensitive, training, test
