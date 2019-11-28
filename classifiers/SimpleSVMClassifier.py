
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

    def predict(self, x):
        return self.model.predict(self.model, x)
