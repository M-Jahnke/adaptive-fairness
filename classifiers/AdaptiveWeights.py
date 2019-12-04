from dataclasses import dataclass
import numpy as np
import nlopt
from classifiers.HeuristicDirect import HeuristicDirect


@dataclass
class Options:
    ep: float = 1e-4
    maxevals: int = 20
    maxits: int = 10
    maxdeep: int = 100
    testflag: int = 0
    showits: int = 1
    globalmin: int = 0
    tol: float = 0.01
    impcons: int = 0


class AdaptiveWeights:  # < handle

    def __init__(self, model, heuristicTraining=False):
        self.model = model
        self.maxItterations = 4
        self.estimatorType = 0
        self.continueFromPreviousWeights = False
        self.heuristicTraining = heuristicTraining

    def train(self, x, y, sensitive, objectiveFunction):
        options = Options(testflag=0, showits=0, maxits=10, maxevals=320, maxdeep=200)

        directLoss = lambda params, grad: -objectiveFunction(
            self.trainModel(x, y, sensitive, params, objectiveFunction), x, y, sensitive)

        if (self.heuristicTraining):
            self.bestParams = HeuristicDirect(directLoss, options)
        else:
            # problem = Problem(directLoss)
            # [_, self.bestParams] = classifiers.Direct(problem, [[0, 1], [0, 1], [0, 3], [0, 3]], options)
            bounds = np.array([[0, 1], [0, 1], [0, 3], [0, 3]])

            opt = nlopt.opt(nlopt.GN_DIRECT, 4)
            opt.set_maxeval(options.maxevals)
            opt.set_lower_bounds(bounds[:, 0])
            opt.set_upper_bounds(bounds[:, 1])
            opt.set_min_objective(directLoss)

            r_start = [np.random.uniform(low=bounds[i][0], high=bounds[i][1]) for i in
                       range(0, np.size(bounds, axis=0))]
            self.bestParams = opt.optimize(r_start)

        self.trainModel(x, y, sensitive, self.bestParams, objectiveFunction)

    def trainModel(self, x, y, sensitive, parameters, objectiveFunction, showConvergence=False):
        mislabelBernoulliMean = [parameters[0], parameters[1]]
        convexity = [parameters[2], parameters[3]]

        convergence = []
        objective = []
        nonSensitive = np.logical_not(sensitive)

        if (sum(y[sensitive]) / sum(sensitive) < sum(y[nonSensitive]) / sum(nonSensitive)):
            tmp = sensitive
            sensitive = nonSensitive
            nonSensitive = tmp

        trainingWeights = np.ones((np.size(y, 0), 1))
        repeatContinue = 1
        itteration = 0
        prevObjective = float('inf')
        while (itteration < self.maxItterations and repeatContinue > 0.01):
            itteration = itteration + 1
            prevWeights = trainingWeights
            self.model.train(x, y, trainingWeights)                             # train weights Gewichte der einzelnen Instancen
            scores = self.model.predict(x)
            trainingWeights[sensitive] = self.convexLoss(scores[sensitive] - y[sensitive], convexity[0]) * \
                                         mislabelBernoulliMean[0] + self.convexLoss(y[sensitive] - scores[sensitive],
                                                                                    convexity[0]) * (
                                                 1 - mislabelBernoulliMean[0])
            trainingWeights[nonSensitive] = self.convexLoss(scores[nonSensitive] - y[nonSensitive], convexity[1]) * (
                    1 - mislabelBernoulliMean[1]) + self.convexLoss(y[nonSensitive] - scores[nonSensitive],
                                                                    convexity[1]) * (mislabelBernoulliMean[1])

            trainingWeights = trainingWeights / sum(trainingWeights) * np.size(trainingWeights, 0)
            repeatContinue = np.linalg.norm(trainingWeights - prevWeights)

            objective = objectiveFunction(self, x, y, sensitive)
            if (objective < prevObjective and itteration > self.maxItterations - 2):
                trainingWeights = prevWeights
                self.model.train(x, y, trainingWeights)
                break

            prevObjective = objective
            if (isinstance(showConvergence, list)):
                convergence = [convergence,
                               np.sqrt(sum(np.power((trainingWeights - prevWeights), 2)) / np.size(trainingWeights, 0))]
                objective = [objective, objective]

        # fprintf('finished within %d itterations for tradeoff %f\n', itteration, tradeoff);
        print('We did it! (╯°□°）╯︵ ┻━┻')

        '''
        if (iscell(showConvergence))
            figure(1);
            hold
            on
            plot(1: length(convergence), convergence, showConvergence
            {1});
            xlabel('Iteration')
            ylabel('Root Mean Square of Weight Edits')
            figure(2);
            hold
            on
            plot(1: obj.maxItterations, [objective
                                         ones(1, obj.maxItterations - length(objective)) * objective(end)], showConvergence
            {1});
            xlabel('Iteration')
            ylabel('Objective')
        end
        '''

    def predict(self, x):
        return self.model.predict(x)

    def convexLoss(self, p, beta=1):
        if self.estimatorType == 0:
            return np.exp(p * beta)
        else:
            raise Exception('Invalid CULEP estimator type')
