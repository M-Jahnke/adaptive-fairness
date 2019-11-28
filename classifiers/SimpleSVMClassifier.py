
from dataclasses import dataclass
from typing import Any
import classifiers
import numpy as np
from sklearn import svm


class SimpleSVMClassifier: # < handle

    def __init__(self):
        self.model = svm.LinearSVC()

    def train(self, x, y, trainingWeights):
        self.model.fit(x, y, trainingWeights)

    #score ist prediction score des Klassifizierers, label ist actual class label
    def predict(self, x):
        [label, score] = self.model.predict(self.model, x)
        return label * np.abs(score[0]) + (1-label) * (1-np.abs(score[0]))
