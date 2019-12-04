import pandas
import numpy as np


def importCompassData():
    # data = dataset('File', '+dataImport/compass.csv', 'ReadVarNames', false, 'Delimiter', ',');
    data = pandas.read_csv('dataImport/compass.csv', header=0)

    # y_temp = str2double(dataset2cell(data(:, 8)))
    y_temp = np.asarray(data.iloc[:, 7], dtype=float)
    # y = ones(size(data, 1), 1)
    y = np.ones((np.size(data, 0), 1))

    # for i=1:length(y_temp)
    for i in range(0, np.size(y_temp, 0)):          #wieso hier auf 1 und 0 setzen ?
        if y_temp[i] == -1:
            y[i] = 0
        else:
            y[i] = 1

    # sensitive = strcmp(cellstr(data(:, 5)), '0') == 1;
    # sensitive = strcmp(cellstr(data(:, 4)), '0') == 1; # MATLAB
    sensitive = [True if str(data.iloc[i, 3].item()) == '0' else False for i in range(0, np.size(data, 0))]

    x = [
        data.iloc[:, 0].astype(np.float),
        data.iloc[:, 1].astype(np.float),
        data.iloc[:, 2].astype(np.float),
        data.iloc[:, 3].astype(np.float),
        data.iloc[:, 4].astype(np.float),
        data.iloc[:, 5].astype(np.float)
    ]

    y = y[2:]
    x = x[2:][:]

    # allow multidimensional slicing (?)
    x = np.asarray(x)
    #print("shape of x: ", x.shape)
    y = np.asarray(y)
    sensitive = np.asarray(sensitive)
    #print("shape of sensitive after creation", sensitive.shape)

    training = np.arange(0, np.floor(np.size(data, 0) * 0.667))         #sollte doch auch 5278 werte haben ?? wird zum slicing benutzt?
    #print("shape of training ", training.shape)
    test = np.arange((np.size(training, 0)), np.size(y, 0))

    return x, y, sensitive, training, test
