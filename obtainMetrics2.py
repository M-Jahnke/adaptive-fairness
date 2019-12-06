import numpy as np
import sklearn as sk
from Utils import compareMatrixWithScalar


def obtainMetrics2(classifier, x, y, sensitive, objectiveWeights=None):
    objectiveWeights = np.zeros(5, 1) if objectiveWeights is None else objectiveWeights
    decisionThreshold = 0.5
    nonSensitive = np.logical_not(sensitive)

    # obtain classification scores
    scores = classifier.predict(x)

    positiveClassification = (scores > decisionThreshold)
    positive = (y > decisionThreshold)

    correctClassifcation = 1 - np.logical_xor(positiveClassification, positive)
    accuracy = sum(correctClassifcation) / np.size(y, 0)

    tp_non_protected = 1 - sum(np.logical_and(compareMatrixWithScalar(positiveClassification[sensitive], 0),
                                              compareMatrixWithScalar(positive[sensitive], 1))) / \
                       sum(compareMatrixWithScalar(positive[sensitive], 1))

    tp_protected = 1 - sum(np.logical_and(compareMatrixWithScalar(positiveClassification[nonSensitive], 0),
                                          compareMatrixWithScalar(positive[nonSensitive], 1))) / \
                   sum(compareMatrixWithScalar(positive[nonSensitive], 1))

    tn_non_protected = 1 - sum(np.logical_and(compareMatrixWithScalar(correctClassifcation[sensitive], 0),
                                              compareMatrixWithScalar(positive[sensitive], 0))) / \
                       sum(compareMatrixWithScalar(positive[sensitive], 0))

    tn_protected = 1 - sum(np.logical_and(compareMatrixWithScalar(correctClassifcation[nonSensitive], 0),
                                          compareMatrixWithScalar(positive[nonSensitive], 0))) / \
                   sum(positive[nonSensitive], 0)

    tp_predicted_count = sum(np.logical_and(compareMatrixWithScalar(positiveClassification[sensitive], 1),
                                            compareMatrixWithScalar(positive[sensitive], 1))) + \
                         sum(np.logical_and(compareMatrixWithScalar(positiveClassification[nonSensitive], 1),
                                            compareMatrixWithScalar(positive[nonSensitive], 1)))

    overall_positive = sum(compareMatrixWithScalar(positive[sensitive], 1)) + sum(
        compareMatrixWithScalar(positive[nonSensitive], 1))

    tn_predicted_count = sum(np.logical_and(compareMatrixWithScalar(correctClassifcation[sensitive], 1),
                                            compareMatrixWithScalar(positive[sensitive], 0))) + \
                         sum(np.logical_and(compareMatrixWithScalar(correctClassifcation[nonSensitive], 1),
                                            compareMatrixWithScalar(positive[nonSensitive], 0)))

    overall_negative = sum(compareMatrixWithScalar(positive[sensitive], 0)) + sum(
        compareMatrixWithScalar(positive[nonSensitive], 0))

    balanced_acc = (tp_predicted_count / (overall_positive) + tn_predicted_count / overall_negative) * 0.5

    # FPR parity
    DFPR = sum(np.logical_and(compareMatrixWithScalar(correctClassifcation[sensitive], 0),
                              compareMatrixWithScalar(positive[sensitive], 0))) / sum(
        compareMatrixWithScalar(positive[sensitive], 0)) - \
           sum(np.logical_and(compareMatrixWithScalar(correctClassifcation[nonSensitive], 0),
                              compareMatrixWithScalar(positive[nonSensitive], 0))) / sum(
        compareMatrixWithScalar(positive[nonSensitive], 0))

    # FNR parity
    DFNR = sum(np.logical_and(compareMatrixWithScalar(positiveClassification[sensitive], 0),
                              compareMatrixWithScalar(positive[sensitive], 1))) / sum(
        compareMatrixWithScalar(positive[sensitive], 1)) - \
           sum(np.logical_and(compareMatrixWithScalar(positiveClassification[nonSensitive], 0),
                              compareMatrixWithScalar(positive[nonSensitive], 1))) / sum(
        compareMatrixWithScalar(positive[nonSensitive], 1))

    # pRule
    a = sum(positiveClassification[sensitive]) / sum(positiveClassification[nonSensitive]) * sum(nonSensitive) / sum(
            sensitive)
    b = sum(positiveClassification[nonSensitive]) / sum(positiveClassification[sensitive]) * sum(sensitive) / sum(
            nonSensitive)
    if np.isnan(a) and np.isnan(b):
        #raise Exception('PRule cannot be computed from two NaN values')
        pRule = min(a, b)
    elif np.isnan(a) and not np.isnan(b):
        pRule = b
    elif not np.isnan(a) and np.isnan(b):
        pRule = a
    else:
        pRule = min(a, b)

    #pRule = min(
    #    sum(positiveClassification[sensitive]) / sum(positiveClassification[nonSensitive]) * sum(nonSensitive) / sum(
    #        sensitive),
    #    sum(positiveClassification[nonSensitive]) / sum(positiveClassification[sensitive]) * sum(sensitive) / sum(
    #        nonSensitive))

    # AUC evaluation
    AUC = sk.metrics.roc_auc_core(y, scores) if (objectiveWeights[1] != 0) else 0

    objective = objectiveWeights[0] * accuracy + objectiveWeights[1] * AUC + objectiveWeights[2] * pRule + \
                objectiveWeights[3] * abs(DFPR) + objectiveWeights[4] * abs(DFNR)

    return objective, accuracy, AUC, pRule, DFPR, DFNR, balanced_acc, tp_non_protected, tp_protected, tn_non_protected, tn_protected
