import pandas
import numpy as np


def importCompassData():
    data = pandas.read_csv('dataImport/compass.csv', header=0)

    y_temp = np.asarray(data.iloc[:, 7], dtype=float)
    y = np.ones((np.size(data, 0), 1))

    for i in range(0, np.size(y_temp, 0)):          #wieso hier auf 1 und 0 setzen ?
        if y_temp[i] == -1:
            y[i] = 0
        else:
            y[i] = 1

    sensitive = [True if str(data.iloc[i, 3].item()) == '0' else False for i in range(0, np.size(data, 0))]

    x = [
        data.iloc[:, 0].astype(np.float),
        data.iloc[:, 1].astype(np.float),
        data.iloc[:, 2].astype(np.float),
        data.iloc[:, 3].astype(np.float),
        data.iloc[:, 4].astype(np.float),
        data.iloc[:, 5].astype(np.float)
    ]

    #y = y[2:] # siehe Erklärung für x
    #x = x[2:][:] # wir schneiden hier die ersten zwei Spalten weg. In MATLAB werden die erstem beiden Zeilen abgeschnitten und so von 5280 auf 5278 gekürzt. Wir haben aber schon 5278. Evtl. Header in MATLAB?

    # allow multidimensional slicing (?)
    x = np.asarray(x).T # should be [5278 6] (from MATLAB)
    y = np.asarray(y) # should be [5278 1] (from MATLAB)
    sensitive = np.asarray(sensitive).reshape((np.size(sensitive, 0), 1)) # should be [5279 1] (from MATLAB), Forscher prüfen 'race' <-- Header auf 0 oder 1, was keinen Sinn macht

    # training size should be [1 3521] (from MATLAB)
    training = np.arange(0, np.floor(np.size(data, 0) * 0.667))         #sollte doch auch 5278 werte haben ?? wird zum slicing benutzt? | Nein, soll 3521 Werte haben! Trainingsset und Testset sollte disjunkt sein. Header wird mit reingezählt von Forschern
    test = np.arange((np.size(training, 0)), np.size(y, 0)) # enthält indizes 3250 bis Ende

    return x, y, sensitive, training, test
