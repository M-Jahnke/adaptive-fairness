from classifiers.SimpleLogisticClassifier import SimpleLogisticClassifier
from dataImport.importCompassData import importCompassData

import numpy as np

from obtainMetrics import obtainMetrics

x, y, sensitive, training, test = importCompassData()

folds = 5

accs = 0
pRules = 0
DFPRs = 0
DFNRs = 0

for fold in range(0, folds):
    classifier = SimpleLogisticClassifier(defaultConvergence=0.0001)
    if (folds != 1):
        training = np.random.standard_normal(round(np.size(training, 0)))
        test = np.setdiff1d(np.arange(0, np.size(y, 0)), training)

    classifier.train(x[training,], y[training])
    _, acc, AUC, pRule, DFPR, DFNR = obtainMetrics(classifier, x[test,], y[test], sensitive[test])
    accs = accs + acc / folds
    pRules = pRules + pRule / folds
    DFPRs = DFPRs + DFPR / folds
    DFNRs = DFNRs + DFNR / folds
    print(
        f"\nCurrent evaluation: acc = {accs * folds / fold}, pRule = {pRules * folds / fold}, DFPR = {DFPRs * folds / fold}, DFNR = {DFNRs * folds / fold}\n\n\n")
