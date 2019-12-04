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

    def train(self, x, y, trainingWeights=None, previousW=None):
        ones = np.ones((x.shape[0], 1))    #gibt Fehler

        #print("shape of ones: ", ones.shape)
        #print("shape of x: ", x.shape)
        #print("shape of y: ", y.shape)

        x = np.append(x, ones, axis=1)

        #print("shape of x after append: ", x.shape)

        previousW = np.zeros([x.shape[1], 1]) if previousW is None else previousW
        self.W = previousW
        xT = np.transpose(x)
        prevError = 1
        velocities = np.ones([x.shape[1], 1])

        #print("shape of velocities: ", velocities.shape)

        trainingWeights = np.ones(y.shape[0], 1) if trainingWeights is None else trainingWeights

        for i in range(0, self.defaultMaxIterations):
            planes = np.matmul(x, self.W)
            errors = sigmoid(planes) - y

            #print("shape of errors: ", errors.shape)

            error = np.linalg.norm(errors)
            if (abs(error - prevError) < self.defaultConvergence):
                break
            prevError = error
            derivatives = sigmoidDerivative(planes)
            '''
            #print("shape of derivative: ", derivatives.shape)
            #print("shape of errors: ", errors.shape)
            #print("shape of trainingWs: ", trainingWeights.shape)
            #print("shape of W: ", self.W.shape)
            #print("shape of xT: ", xT.shape)
            
            accumulation = derivatives * errors * trainingWeights
            #print("shape accumulation before x: ", accumulation.shape)
            accumulation = np.matmul(xT, accumulation) # sollte xT sein np.matmul(xT, accumulation)
            #print("shape of accumulation: ", accumulation.shape)
            accumulation = accumulation / y.shape[0]
            #print("shape of accumulation after first div by y shape: ", accumulation.shape)
            accumulation = accumulation + self.defaultRegularization * self.W
            #print("shape of accumulation after addition with regularization: ", accumulation.shape)
            accumulation = accumulation / y.shape[0]
            '''
            accumulation = np.matmul(xT, (derivatives * errors * trainingWeights)) / y.shape[0] + self.defaultRegularization * self.W / y.shape[0]
            velocities = velocities * 0.2 + 0.8 * np.power(accumulation, 2)
            self.W = self.W - self.defaultTrainingRate * accumulation / np.sqrt(velocities + 0.1)
            print(f"weights updated in iteration {i}")
            if self.trainingErrorTracking:
                self.trackedError.append(error / y.shape[0])
                
    def predict(self, x):
        #print("shape of W: ", self.W.shape)
        ones = np.ones((x.shape[0], 1))
        #print("shape of ones: ", ones.shape)
        #print("x before append", x.shape)
        x = np.append(x, ones, axis=1)
        #print("shape of x after append", x.shape)
        planes = np.matmul(x, self.W)
        return sigmoid(planes)
    
    def enableTrainingErrorTracking(self):
        self.trainingErrorTracking = True
        
        
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidDerivative(x):
    return np.exp(-x) / np.power((1 + np.exp(-x)), 2)
