from dataclasses import dataclass
import numpy as np



class SimpleLogisticClassifier():
    def __init__(self, defaultConvergence=0.001, defaultTrainingRate=0.1, defaultRegularization=0,
             defaultMaxIterations=10000, trainingErrorTracking=False):
        self.defaultConvergence = defaultConvergence
        self.defaultTrainingRate = defaultTrainingRate
        self.defaultRegularization = defaultRegularization
        self.defaultMaxIterations = defaultMaxIterations
        self.trainingErrorTracking = trainingErrorTracking
        self.trackedError = []
        self.W = None

    def train(self, x, y, trainingWeights=None, previousW=None, regularization=None, trainingRate=None, maxItterations=None):
        ones = np.ones((np.size(x, 0), 1)) # shape [3520, 1]
        x = np.append(x, ones, axis=1) # shape [3520, 7], size in MATLAB is [3521 7]. 1 Dim mehr, wegen des Headers, den Forscher fälschlicherweise für das Training verwenden

        # nargins
        trainingWeights = np.ones((np.size(y, 0), 1)) if trainingWeights is None else trainingWeights
        previousW = np.zeros((np.size(x, 1), 1)) if previousW is None else previousW
        regularization = self.defaultRegularization if regularization is None else regularization
        trainingRate = self.defaultTrainingRate if trainingRate is None else trainingRate
        maxItterations = self.defaultMaxIterations if maxItterations is None else maxItterations

        self.W = previousW
        xT = x.conj().transpose() # conj() ist wahrscheinlich unnötig da wir nur reelzahlige Elemente in x haben
        prevError = 1
        velocities = np.ones((np.size(x, 1), 1))

        for i in range(0, maxItterations):
            planes = np.matmul(x, self.W)
            errors = sigmoid(planes) - y

            error = np.linalg.norm(errors)
            if (abs(error - prevError) < self.defaultConvergence):
                break
            prevError = error
            derivatives = sigmoidDerivative(planes)
            accumulation = np.matmul(xT, (derivatives * errors * trainingWeights)) / y.shape[0] + regularization * self.W / y.shape[0] # * is only for ndarray type element-wise, care! shape is [7, 1]
            velocities = velocities * 0.2 + 0.8 * np.power(accumulation, 2) # shape is [7, 1]
            self.W = self.W - trainingRate * accumulation / np.sqrt(velocities + 0.1)
            if self.trainingErrorTracking:
                self.trackedError.append(error / y.shape[0])
                
    def predict(self, x):
        ones = np.ones((np.size(x, 0), 1))  # shape [3520, 1]
        x = np.append(x, ones, axis=1)

        planes = np.matmul(x, self.W)
        return sigmoid(planes)
    
    def enableTrainingErrorTracking(self):
        self.trainingErrorTracking = True
        
        
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidDerivative(x):
    return np.exp(-x) / np.power((1 + np.exp(-x)), 2)
