from dataclasses import dataclass
import classifiers
import numpy as np


def FairLogisticClassifier():
    
    def __init__(self, defaultConvergence = None, defaultTrainingRate = None, defaultRegularization = None, defaultMaxIterations = None, trainingErrorTracking = None):
        self.defaultConvergence = 0.001 if defaultConvergence is None else defaultConvergence
        self.defaultTrainingRate = 0.1 if defaultTrainingRate is None else defaultTrainingRate
        self.defaultRegularization = 0 if defaultRegularization is None else defaultRegularization
        self.defaultMaxIterations = 10000 if defaultMaxIterations is None else defaultMaxIterations
        self.trainingErrorTracking = False if trainingErrorTracking is None else trainingErrorTracking
        self.trackedError = []
        self.W = None
        
    def train(self, x, y, sensitive, CULEPparams, trainingWeights=None, previousW=None):
        zeroes = np.ones([x.shape[0],1])
        np.append(x,zeroes)
        previousW = np.zeros([x.shape[0], 2]) if preciousW is None else previousW
        mislabelBernoulliMean1 = CULEPparams[0]
        mislabelBernoulliMean2 = CULEPparams[1]
        convexity1 = CULEPparams[2]
        convexity2 = CULEPparams[3]
#        nonSensitive = not sensitive
        self.W = previousW
        xT = np.transpose(x)
        prevError = 1
        velocities = np.ones([x.shape[0], 2])
        trainingWeights = np.ones(y.shape[0], 1) if trainingWeights is None else trainingWeights
        trainingWeightsNext = trainingWeights
        for i in range(0, self.maxIterations):
            planes = np.multiply(x,self.W)
            scores = sigmoid(planes)
            errors = np.subtract(scores, y)
            error = norm(errors)
            if(abs(error-prevError)<self.defaultConvergence):
                break
            prevError = error
            derivatives = sigmoidDerivative(planes)
            accumulation = np.divide(np.multiply(xT, np.multiply(np.multiply(derivatives, errors), trainingWeights)), y.shape[0])
            acuumulation = np.add(accumulation, np.divide(np.multiply(self.defaultRegularization, self.W), y.shape[0]))
            velocities = np.add(np.multiply(velocities, 0.2), np.multiply(np.square(accumulation), 0.8))
            self.W = np.subtract(self.W, np.divide(np.multiply(self.defaultTrainingsRate, accumulation), np.sqrt(np.add(velocities, 0.1))))
            
            if self.trainingErrorTracking:
                self.trackedError.add(error/y.shape[0])
            
            #sensitive blah
            trainingWeights[sensitive] = convexLoss(scores[sensitive] - y[sensitive], convexity[0]) * \
                                         mislabelBernoulliMean[0] + convexLoss(y[sensitive] - scores[sensitive],
                                                                                    convexity[0]) * (
                                                     1 - mislabelBernoulliMean[0])
            trainingWeights[nonSensitive] = convexLoss(scores[nonSensitive] - y[nonSensitive], convexity[1]) * (
                        1 - mislabelBernoulliMean[1]) + convexLoss(y[nonSensitive] - scores[nonSensitive],
                                                                        convexity[1]) * (mislabelBernoulliMean[1])
            
            trainingWeightsNext = trainingWeightsNext/np.sum(trainingWeightsNext) * trainingWeightsNext.shape[0]
            trainingWeights = trainingWeights + trainingRate * np.subtract(trainingWeightsNext, trainingWeights)
        
                    
    def predict(self, x):
        ones = np.ones([x.shape[0], 1])
        np.append(x, ones) #nicht ones?
        planes = np.multiply(x, self.w)
        return sigmoid(planes)
    
    def enableTrainingErrorTracking(self):
        self.trainingErrorTracking = true
        
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidDerivate(x):
    return exp(-x) / np.power((1 + np.exp(-x)), 2)
        
def convexLoss(x, beta):
    x = x*beta
    y = np.exp(x)
    return y