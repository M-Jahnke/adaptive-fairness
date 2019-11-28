from dataclasses import dataclass
import numpy as np



def SimpleLogisticClassifier():
    def init(self, defaultConvergence=0.001, defaultTrainingRate=0.1, defaultRegularization=0,
             defaultMaxIterations=10000, trainingErrorTracking=False):
        self.defaultConvergence = defaultConvergence
        self.defaultTrainingRate = defaultTrainingRate
        self.defaultRegularization = defaultRegularization
        self.defaultMaxIterations = defaultMaxIterations
        self.trainingErrorTracking = trainingErrorTracking
        self.trackedError = []
        self.W = None

    def train(self, x, y, trainingWeights=None, previousW=None):
        ones = np.ones([x.shape[0], 1])
        np.append(x, ones)
        previousW = np.zeros([x.shape[0], 2]) if previousW is None else previousW
        self.W = previousW
        xT = np.transpose(x)
        prevError = 1
        velocities = np.ones([x.shape[0], 2])
        trainingWeights = np.ones(y.shape[0], 1) if trainingWeights is None else trainingWeights

        for i in range(0, self.maxIterations):
            planes = x * self.W
            errors = sigmoid(planes) - y
            error = np.linalg.norm(errors)
            if (abs(error - prevError) < self.defaultConvergence):
                break
            prevError = error
            derivatives = sigmoidDerivative(planes)

            accumulation = xT * (derivatives * errors * trainingWeights) / y.shape[0] + self.defaultRegularization * self.W / y.shape[0]
            velocities = velocities * 0.2 + 0.8 * np.power(accumulation, 2)
            self.W = self.W - self.trainingRate * accumulation / np.sqrt(velocities + 0.1)

            if self.trainingErrorTracking:
                self.trackedError.add(error / y.shape[0])
                
    def predict(self, x):
        ones = np.ones([x.shape[0], 1])
        np.append(x, ones)
        planes = x * self.W
        return sigmoid(planes)
    
    def enableTrainingErrorTracking(self):
        self.trainingErrorTracking = True
        
        
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidDerivative(x):
    return np.exp(-x) / np.power((1 + np.exp(-x)), 2)
