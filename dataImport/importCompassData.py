import pandas
import numpy as np


def importCompassData():
    # data = dataset('File', '+dataImport/compass.csv', 'ReadVarNames', false, 'Delimiter', ',');
    data = pandas.read_csv('./compass.csv', delimeter=',', header=0)

    # y_temp = str2double(dataset2cell(data(:, 8)))
    y_temp = data[:, 7].astype(np.float)
    # y = ones(size(data, 1), 1)
    y = np.ones(np.size(data, 0), 1)

    # for i=1:length(y_temp)
    for i in range(0, len(y_temp)):
        if y_temp[i] == -1:
            y[i] = 0
        else:
            y[i] = 1

    # sensitive = strcmp(cellstr(data(:, 5)), '0') == 1;
    # sensitive = strcmp(cellstr(data(:, 4)), '0') == 1; # MATLAB
    # sensitive = str(data[:, 3]) == '0' Python
    sensitive = [True if str(data[i, 3]) == '0' else False for i in range(data)]

    x = [
        data[:, 0].astype(np.float),
        data[:, 1].astype(np.float),
        data[:, 2].astype(np.float),
        data[:, 3].astype(np.float),
        data[:, 4].astype(np.float),
        data[:, 5].astype(np.float)
    ]

    y = y[2:, 0]
    x = x[2:, :]

    training = np.arange(0, np.floor(np.size(data, 1) * 0.667))
    test = np.arange((len(training) + 1), len(y) + 1)

    return x, y, sensitive, training, test
