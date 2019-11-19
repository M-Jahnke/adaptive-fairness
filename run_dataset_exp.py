from classifiers.AdaptiveWeights import AdaptiveWeights
from classifiers.SimpleLogisticClassifier import SimpleLogisticClassifier
import numpy as np

from obtainMetrics import obtainMetrics
from obtainMetrics2 import obtainMetrics2


def run_dataset_exp(dataset, output_directory, iterations):
    # rng(12345)
    np.random.seed(12345)

    if (dataset == 'compass'):
        [x, y, sensitive, training, test] = dataImport.importCompassData()
    elif (dataset == 'adult'):
        [x, y, sensitive, training, test] = dataImport.MyimportAdultData()
    elif (dataset == 'bank'):
        [x, y, sensitive, training, test] = dataImport.importBankData()
    elif (dataset == 'dutch'):
        [x, y, sensitive, training, test] = dataImport.importDutchData()
    elif (dataset == 'kdd'):
        [x, y, sensitive, training, test] = dataImport.ImportKDD()

    folds = iterations

    accs = 0
    pRules = 0
    DFPRs = 0
    DFNRs = 0
    b_accs = 0
    fileID = open(output_directory, 'w')

    # validationFunction = obtainMetrics(c, x, y, s, [2, 0, 0, -0, -1]) # for disparate mistreatment
    # validationFunction2 = obtainMetrics2(c, x, y, s, [2, 0, 0, -0, -1]) # for disparate mistreatment

    for fold in range(1, len(folds) + 1):
        print(f"iteration {fold}")
        if (folds != 1):
            # training = randsample(np.arange(1, len(y)), len(training))
            training = np.random.standard_normal(len(training))  # ?
            test = np.setdiff1d(np.arange(1, len(y)), training)  # ?

        classifier = AdaptiveWeights(SimpleLogisticClassifier(0.0001))

        validationFunction = obtainMetrics(classifier, x, y, sensitive, [2, 0, 0, -0, -1])  # for disparate mistreatment
        validationFunction2 = obtainMetrics2(classifier, x, y, sensitive,
                                             [2, 0, 0, -0, -1])  # for disparate mistreatment

        classifier.train(x[training, :], y(training), sensitive(training), validationFunction)
        _, acc, _, pRule, DFPR, DFNR, b_acc, TP_NP, TP_P, TN_NP, TN_P = validationFunction2(classifier, x[test, :],
                                                                                              y(test), sensitive(test))

        accs = accs + acc / folds
        pRules = pRules + pRule / folds
        DFPRs = DFPRs + DFPR / folds
        DFNRs = DFNRs + DFNR / folds
        b_accs = b_accs + b_acc / folds
        print(f"{fileID} TP_P= {TP_P}, TP_NP= {TP_NP}, TN_P= {TN_P}, TN_NP= {TN_NP}")
        print(
            f"{fileID}\nCurrent evaluation on fold {fold}: acc = {accs * folds / fold}, balanced acc = {b_accs * folds / fold}, pRule = {pRules * folds / fold}, DFPR = {DFPRs * folds / fold}, DFNR = {DFNRs * folds / fold}\n\n\n")

    fileID.close()
