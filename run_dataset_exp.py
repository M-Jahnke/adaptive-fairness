from classifiers.AdaptiveWeights import AdaptiveWeights
from classifiers.SimpleLogisticClassifier import SimpleLogisticClassifier
from dataImport.importCompassData import importCompassData
from dataImport.importAdultData import importAdultData
from dataImport.importBankData import importBankData
from dataImport.importDutchData import importDutchData
from dataImport.importKDD import importKDD

import numpy as np
import random

from obtainMetrics import obtainMetrics
from obtainMetrics2 import obtainMetrics2


def run_dataset_exp(dataset, output_directory, iterations):
    random.seed(12345)

    if (dataset == 'compass'):
        x, y, sensitive, training, test = importCompassData()
    elif (dataset == 'adult'):
        x, y, sensitive, training, test = importAdultData()
    elif (dataset == 'bank'):
        x, y, sensitive, training, test = importBankData()
    elif (dataset == 'dutch'):
        x, y, sensitive, training, test = importDutchData()
    elif (dataset == 'kdd'):
        x, y, sensitive, training, test = importKDD()
    else:
        return

    folds = iterations

    accs = 0
    pRules = 0
    DFPRs = 0
    DFNRs = 0
    b_accs = 0
    fileID = open(output_directory, 'w')

    validationFunction = lambda c, x, y, s: obtainMetrics(c, x, y, s, [1, 0, 1, 0, 0])  # for disparate impact (adult and bank)
    validationFunction2 = lambda c, x, y, s: obtainMetrics2(c, x, y, s, [1, 0, 1, 0, 0])  # for disparate impact (adult and bank)

    #validationFunction = lambda c, x, y, s: obtainMetrics(c, x, y, s, [2, 0, 0, -0, -1]) # for disparate mistreatment (Compass, Dutch and KDD)
    #validationFunction2 = lambda c, x, y, s: obtainMetrics2(c, x, y, s, [2, 0, 0, -0, -1]) # for disparate mistreatment (Compass, Dutch and KDD)

    for fold in range(1, folds+1):
        print(f"iteration {fold}")
        if (folds != 1):
            # training = randsample(np.arange(1, len(y)), len(training))
            training = random.sample(np.arange(0, np.size(y, 0)).tolist(), np.size(training, 0))
            test = np.setdiff1d(np.arange(0, np.size(y, 0)), training)

        classifier = AdaptiveWeights(SimpleLogisticClassifier(defaultConvergence=0.0001))

        classifier.train(x[training, :], y[training], sensitive[training], validationFunction)
        _, acc, _, pRule, DFPR, DFNR, b_acc, TP_NP, TP_P, TN_NP, TN_P = validationFunction2(classifier, x[test,],
                                                                                              y[test], sensitive[test])

        accs = accs + acc / folds
        pRules = pRules + pRule / folds
        DFPRs = DFPRs + DFPR / folds
        DFNRs = DFNRs + DFNR / folds
        b_accs = b_accs + b_acc / folds

        str = f"TP_P= {TP_P}, TP_NP= {TP_NP}, TN_P= {TN_P}, TN_NP= {TN_NP}"
        fileID.write(str)
        print(str)
        str = f"Current evaluation on fold {fold}: acc = {accs * folds / fold}, balanced acc = {b_accs * folds / fold}, pRule = {pRules * folds / fold}, DFPR = {DFPRs * folds / fold}, DFNR = {DFNRs * folds / fold}"
        fileID.write(str)
        print(str)

    fileID.close()
