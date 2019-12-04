from classifiers.AdaptiveWeights import AdaptiveWeights
from classifiers.SimpleLogisticClassifier import SimpleLogisticClassifier
from dataImport.importCompassData import importCompassData

import numpy as np

from obtainMetrics import obtainMetrics
from obtainMetrics2 import obtainMetrics2

x, y, sensitive, training, test = importCompassData()
folds = 5
# x = [x sensitive] # introduces disparate treatment (for synthetic datasets)


# variables to store average results
accs = 0
pRules = 0
DFPRs = 0
DFNRs = 0
b_accs = 0

# validationFunction = @(c,x,y,s)obtainMetrics(c,x,y,s,[1 0 1 0 0]) # for disparate impact
validationFunction = lambda c, x, y, s: obtainMetrics(c, x, y, s, [2, 0, 0, -1, -1])  # for disparate mistreatment
validationFunction2 = lambda c, x, y, s: obtainMetrics2(c, x, y, s, [2, 0, 0, -1, -1])  # for disparate mistreatment

for fold in range(0, folds):
    if (folds != 1):
        training = np.asarray(np.random.standard_normal(np.size(training, 0)), dtype=np.int)
        test = np.asarray(np.setdiff1d(np.arange(1, np.size(y, 0)), training), dtype=np.int)
    print(f"Fold started: {fold}")
    classifier = AdaptiveWeights(SimpleLogisticClassifier(defaultConvergence=0.0001))
    classifier.train(x[training], y[training], sensitive[training], validationFunction)             # x ist array mit shape [4,5278]
    _, acc, _, pRule, DFPR, DFNR, b_acc, TP_NP, TP_P, TN_NP, TN_P = validationFunction2(classifier, x[test,], y[test],
                                                                                        sensitive[test])

    accs = accs + acc / folds
    pRules = pRules + pRule / folds
    DFPRs = DFPRs + DFPR / folds
    DFNRs = DFNRs + DFNR / folds
    b_accs = b_accs + b_acc / folds
    print(f"\nTP_P={TP_P}, TP_NP={TP_NP}, TN_P={TN_P}, TN_NP={TN_NP}")

    print(
        f"\nCurrent evaluation on fold {fold}: acc = {accs * folds / fold}, balanced acc = {b_accs * folds / fold}, pRule = {pRules * folds / fold}, DFPR = {DFPRs * folds / fold}, DFNR = {DFNRs * folds / fold}\n\n\n")
