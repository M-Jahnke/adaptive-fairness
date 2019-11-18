from dataclasses import dataclass
import classifiers

@dataclass
class Problem:
    f: ...



class AdaptiveWeights: #< handle

        def __init__(self, model, heuristicTraining=False):
            self.model = model
            self.maxItterations = 4
            self.estimatorType = 0
            self.continueFromPreviousWeights = False
            self.heuristicTraining = heuristicTraining
        
        def train(self, obj, x, y, sensitive, objectiveFunction):
            self.testflag  = 0 # don't know global min
            self.showits   = 0 # show iterations
            self.maxits    = 10 # max number of iterations
            self.maxevals  = 320 # max function evaluations
            self.maxdeep   = 200 # max rect divisions
            
            directLoss = -objectiveFunction(self.trainModel(x, y, sensitive, params, objectiveFunction), x, y, sensitive)
            
            if(self.heuristicTraining):
                self.bestParams = classifiers.HeuristicDirect(directLoss, options)
            else:
                Problem.f = directLoss # ?
                [_,self.bestParams] = classifiers.Direct(Problem, [0 1;0 1;0 3;0 3], options)

            self.trainModel(x, y, sensitive, self.bestParams, objectiveFunction)
        end
        
        def trainModel(obj, x, y, sensitive, parameters, objectiveFunction, showConvergence=False):
            if nargin<7
                showConvergence = false;
                
            mislabelBernoulliMean(1) = parameters(1);
            mislabelBernoulliMean(2) = parameters(2);
            convexity(1) = parameters(3);
            convexity(2) = parameters(4);
            convergence = [];
            objective = [];
            nonSensitive = !sensitive;

            if(sum(y(sensitive))/sum(sensitive)<sum(y(nonSensitive))/sum(nonSensitive))
                tmp = sensitive;
                sensitive = nonSensitive;
                nonSensitive = tmp;
            end

            trainingWeights = ones(length(y),1);
            repeatContinue = 1;
            itteration = 0;
            prevObjective = Inf;
            while(itteration<obj.maxItterations && repeatContinue>0.01)
                itteration = itteration+1;
                prevWeights = trainingWeights;
                obj.model.train(x, y, trainingWeights);
                scores = obj.model.predict(x);
                trainingWeights(sensitive) ...
                    = obj.convexLoss(scores(sensitive)-y(sensitive),convexity(1))*mislabelBernoulliMean(1) ... 
                    + obj.convexLoss(y(sensitive)-scores(sensitive),convexity(1))*(1-mislabelBernoulliMean(1));
                trainingWeights(nonSensitive) ...
                    = obj.convexLoss(scores(nonSensitive)-y(nonSensitive),convexity(2))*(1-mislabelBernoulliMean(2)) ... 
                    + obj.convexLoss(y(nonSensitive)-scores(nonSensitive),convexity(2))*(mislabelBernoulliMean(2));

                trainingWeights = trainingWeights/sum(trainingWeights)*length(trainingWeights);
                repeatContinue = norm(trainingWeights-prevWeights);
                
                objective = objectiveFunction(obj, x, y, sensitive);
                if(objective<prevObjective && itteration>obj.maxItterations-2)
                    trainingWeights = prevWeights;
                    obj.model.train(x, y, trainingWeights);
                    break;
                end
                prevObjective = objective;
                if(iscell(showConvergence))
                    convergence = [convergence sqrt(sum((trainingWeights-prevWeights).^2)/length(trainingWeights))];
                    objective = [objective objective];
                end
            end
            %fprintf('finished within %d itterations for tradeoff %f\n', itteration, tradeoff);
            if(iscell(showConvergence))
                figure(1);
                hold on
                plot(1:length(convergence),convergence,showConvergence{1});
                xlabel('Iteration')
                ylabel('Root Mean Square of Weight Edits')
                figure(2);
                hold on
                plot(1:obj.maxItterations,[objective ones(1,obj.maxItterations-length(objective))*objective(end)],showConvergence{1});
                xlabel('Iteration')
                ylabel('Objective')
            end
        end
        
        function y = predict(obj, x)
            y = obj.model.predict(x);
        end
        
        function L = convexLoss(obj, p, beta)
            if(nargin<3)
                beta = 1;
            end
            if(obj.estimatorType==0)
                L = exp(p*beta);
            else
                error('Invalid CULEP estimator type');
            end
        end
    end
end
