from dataclasses import dataclass
import numpy as np


def FairLogisticClassifier():
    
    def __init__(self, defaultConvergence = 0.001, defaultTrainingRate = 0.1, defaultRegularization = 0, defaultMaxIterations = 10000, trainingErrorTracking = False):
        self.defaultConvergence = defaultConvergence
        self.defaultTrainingRate = defaultTrainingRate
        self.defaultRegularization = defaultRegularization
        self.defaultMaxIterations = defaultMaxIterations
        self.trainingErrorTracking = trainingErrorTracking
        self.trackedError = []
        self.W = None
        
    def train(self, x, y, sensitive, CULEPparams, trainingWeights=None, previousW=None):
        zeroes = np.ones([x.shape[0], 1])
        np.append(x,zeroes)
        previousW = np.zeros([x.shape[0], 2]) if previousW is None else previousW
        mislabelBernoulliMean = [CULEPparams[0], CULEPparams[1]]
        convexity = [CULEPparams[2], CULEPparams[3]]
        nonSensitive = np.logical_not(sensitive)
        self.W = previousW
        xT = np.transpose(x)
        prevError = 1
        velocities = np.ones([x.shape[0], 2])
        trainingWeights = np.ones(y.shape[0], 1) if trainingWeights is None else trainingWeights
        trainingWeightsNext = trainingWeights
        for i in range(0, self.maxIterations):
            planes = x * self.W
            scores = sigmoid(planes)
            errors = scores - y
            error = np.linalg.norm(errors)
            if abs(error-prevError) < self.defaultConvergence:
                break
            prevError = error
            derivatives = sigmoidDerivative(planes)

            accumulation = xT * (derivatives * errors * trainingWeights) / y.shape[0] + self.defaultRegularization * self.W / y.shape[0]
            velocities = velocities * 0.2 + 0.8 * np.power(accumulation, 2)
            self.W = self.W - self.defaultTrainingRate * accumulation / np.sqrt(velocities + 0.1)

            if self.trainingErrorTracking:
                self.trackedError.add(error/y.shape[0])

            trainingWeights[sensitive] = convexLoss(scores[sensitive] - y[sensitive], convexity[0]) * \
                                         mislabelBernoulliMean[0] + convexLoss(y[sensitive] - scores[sensitive],
                                                                                    convexity[0]) * (
                                                     1 - mislabelBernoulliMean[0])
            trainingWeights[nonSensitive] = convexLoss(scores[nonSensitive] - y[nonSensitive], convexity[1]) * (
                        1 - mislabelBernoulliMean[1]) + convexLoss(y[nonSensitive] - scores[nonSensitive],
                                                                        convexity[1]) * (mislabelBernoulliMean[1])
            
            trainingWeightsNext = trainingWeightsNext/np.sum(trainingWeightsNext) * trainingWeightsNext.shape[0]
            trainingWeights = trainingWeights + self.defaultTrainingRate * np.subtract(trainingWeightsNext, trainingWeights)

    def predict(self, x):
        ones = np.ones([x.shape[0], 1])
        np.append(x, ones)
        planes = np.multiply(x, self.w)
        return sigmoid(planes)
    
    def enableTrainingErrorTracking(self):
        self.trainingErrorTracking = True


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoidDerivative(x):
    return np.exp(-x) / np.power((1 + np.exp(-x)), 2)


def convexLoss(x, beta):
    x = x*beta
    y = np.exp(x)
    return y