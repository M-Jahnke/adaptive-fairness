import numpy as np
import sklearn as sk


def obtainMetrics2(classifier, x, y, sensitive, objectiveWeights=np.zeros(5, 1)):
    decisionThreshold = 0.5
    nonSensitive = not sensitive

    # obtain classification scores
    scores = classifier.predict(x)

    positiveClassification = (scores > decisionThreshold)
    positive = (y > decisionThreshold)

    correctClassifcation = 1 - np.logical_xor(positiveClassification, positive)
    accuracy = sum(correctClassifcation) / len(y)

    tp_non_protected = 1 - sum(positiveClassification(sensitive) == 0 & positive(sensitive) == 1) / sum(
        positive(sensitive) == 1)
    tp_protected = 1 - sum(positiveClassification(nonSensitive) == 0 & positive(nonSensitive) == 1) / sum(
        positive(nonSensitive) == 1)

    tn_non_protected = 1 - sum(correctClassifcation(sensitive) == 0 & positive(sensitive) == 0) / sum(
        positive(sensitive) == 0)
    tn_protected = 1 - sum(correctClassifcation(nonSensitive) == 0 & positive(nonSensitive) == 0) / sum(
        positive(nonSensitive) == 0)

    tp_predicted_count = sum(positiveClassification(sensitive) == 1 & positive(sensitive) == 1) + sum(
        positiveClassification(nonSensitive) == 1 & positive(nonSensitive) == 1)
    averall_positive = sum(positive(sensitive) == 1) + sum(positive(nonSensitive) == 1)

    tn_predicted_count = sum(correctClassifcation(sensitive) == 1 & positive(sensitive) == 0) + sum(
        correctClassifcation(nonSensitive) == 1 & positive(nonSensitive) == 0)
    averall_negative = sum(positive(sensitive) == 0) + sum(positive(nonSensitive) == 0)

    balanced_acc = (tp_predicted_count / (averall_positive) + tn_predicted_count / averall_negative) * 0.5

    # FPR parity
    DFPR = sum(correctClassifcation(sensitive) == 0 & positive(sensitive) == 0) / sum(positive(sensitive) == 0) - sum(
        correctClassifcation(nonSensitive) == 0 & positive(nonSensitive) == 0) / sum(positive(nonSensitive) == 0)
    # FNR parity
    DFNR = sum(positiveClassification(sensitive) == 0 & positive(sensitive) == 1) / sum(positive(sensitive) == 1) - sum(
        positiveClassification(nonSensitive) == 0 & positive(nonSensitive) == 1) / sum(positive(nonSensitive) == 1)

    # pRule
    pRule = min(
        sum(positiveClassification(sensitive)) / sum(positiveClassification(nonSensitive)) * sum(nonSensitive) / sum(
            sensitive),
        sum(positiveClassification(nonSensitive)) / sum(positiveClassification(sensitive)) * sum(sensitive) / sum(
            nonSensitive))

    # AUC evaluation
    AUC = sk.metrics.roc_auc_core(y, scores) if (objectiveWeights[2] != 0) else 0

    objective = objectiveWeights[1] * accuracy + objectiveWeights[2] * AUC + objectiveWeights[3] * pRule + \
                objectiveWeights[4] * abs(DFPR) + objectiveWeights[5] * abs(DFNR)
    return objective, accuracy, AUC, pRule, DFPR, DFNR, balanced_acc, tp_non_protected, tp_protected, tn_non_protected, tn_protected
