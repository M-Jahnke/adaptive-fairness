
from dataclasses import dataclass
from typing import Any
import classifiers
import numpy as np
from sklearn import svm

class SimpleSVMClassifier: # < handle

    def __init__(self, model):
        self.model = model

    def train(self, x, y, trainingWeights):
        # if nargin < 4 then trainingWeights = np.ones(length(y), 1)
        #self.model = fitcsvm(x, y, 'Weights', trainingWeights)
        self.model =  svm.fit(x, y, trainingWeights)

    def predict(self, x):
        [label, score] = predict(self.model, x)
        return label * np.abs(score[0]) + (1-label) * (1-np.abs(score[0]))
