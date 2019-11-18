def obtainMetrics(classifier, x, y, sensitive, objectiveWeights = zeros(5,1)):
    decisionThreshold = 0.5
    nonSensitive = not sensitive
    
    # obtain classification scores
    scores = classifier.predict(x)
    
    positiveClassification = (scores > decisionThreshold)
    positive = (y > decisionThreshold)
    
    correctClassifcation = 1 - xor(positiveClassification,positive) 
    accuracy = sum(correctClassifcation) / len(y)
    
    #FPR parity
    DFPR = sum(correctClassifcation(sensitive)==0 & positive(sensitive)==0)/sum(positive(sensitive)==0) - sum(correctClassifcation(nonSensitive)==0 & positive(nonSensitive)==0)/sum(positive(nonSensitive)==0)
    #FNR parity
    DFNR = sum(positiveClassification(sensitive)==0 & positive(sensitive)==1)/sum(positive(sensitive)==1) - sum(positiveClassification(nonSensitive)==0 & positive(nonSensitive)==1)/sum(positive(nonSensitive)==1)
         
    #pRule
    pRule = min(sum(positiveClassification(sensitive))/sum(positiveClassification(nonSensitive)) * sum(nonSensitive)/sum(sensitive), sum(positiveClassification(nonSensitive))/sum(positiveClassification(sensitive)) * sum(sensitive)/sum(nonSensitive))

    
    # AUC evaluation
    if(objectiveWeights(2)!=0):
        [_, _, _, AUC] = perfcurve(y,scores,1)
    else:
        AUC = 0
    
    objective = objectiveWeights(1)*accuracy + objectiveWeights(2)*AUC + objectiveWeights(3)*pRule + objectiveWeights(4)*abs(DFPR) + objectiveWeights(5)*abs(DFNR)
    
    return [objective, accuracy, AUC, pRule, DFPR, DFNR]
