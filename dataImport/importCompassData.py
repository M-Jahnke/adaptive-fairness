import pandas
import numpy as np

def importCompassData():
    #data = dataset('File', '+dataImport/compass.csv', 'ReadVarNames', false, 'Delimiter', ',');
    data = pandas.read_csv('./compass.csv', delimeter=',', header=0)

    # y_temp = str2double(dataset2cell(data(:, 8)))
    y_temp = map(float, data[:, 8].tolist())
    #y = ones(size(data, 1), 1)
    y = np.ones(np.size(data, 1), 1)

    #for i=1:length(y_temp)
    for i in range(0, len(y_temp)):
        if y_temp[i] == -1:
            y[i] = 0
        else:
            y[i] = 1

    # sensitive = strcmp(cellstr(data(:, 5)), '0') == 1;
    # sensitive = strcmp(cellstr(data(:, 4)), '0') == 1;
    sensitive = str(data[:, 3]) == '0'

    x = [
        map(float, data[:, 0].tolist()),
        map(float, data[:, 1].tolist()),
        map(float, data[:, 2].tolist()),
        map(float, data[:, 3].tolist()),
        map(float, data[:, 4].tolist()),
        map(float, data[:, 5].tolist())
    ]

    y = y[3:, 1]
    x = x[3:, :]

    training = np.arange(1, np.floor(np.size(data, 1) * 0.667) + 1)
    test = np.arange((len(training) + 1), len(y) + 1)

    return [x, y, sensitive, training, test]