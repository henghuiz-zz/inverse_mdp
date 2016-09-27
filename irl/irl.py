import numpy as np
from numpy.matlib import repmat
from scipy.optimize import minimize

def NegativeLogLikelihood(theta, SampleFeature, SampleControl):
    numAction = SampleFeature.shape[1]
    numFea = SampleFeature.shape[2]
    numSample = SampleControl.shape[0]
    theta = theta[0:numFea]
    StateActionProb = np.dot(SampleFeature, theta)
    StateActionProb = np.exp(StateActionProb)
    sumStateActionProb = np.sum(StateActionProb, 1, keepdims=True)
    StateActionProb = StateActionProb / repmat(sumStateActionProb, 1, numAction)
    Prob = StateActionProb[range(numSample), SampleControl.astype("int")]
    NLL = -np.sum(np.log(Prob))
    return (NLL / (numSample + 0.0))


def GradientNegativeLogLikelihood(theta, SampleFeature, SampleControl):
    numAction = SampleFeature.shape[1]
    numFea = SampleFeature.shape[2]
    numSample = SampleControl.shape[0]
    theta = theta[0:numFea]
    StateActionProb = np.dot(SampleFeature, theta)
    StateActionProb = np.exp(StateActionProb)
    sumStateActionProb = np.sum(StateActionProb, 1, keepdims=True)
    StateActionProb = StateActionProb / repmat(sumStateActionProb, 1, numAction)

    DNLL = np.zeros(2 * numFea)
    for idControl in range(numAction):
        leftm = (StateActionProb[:, idControl] - (SampleControl == idControl))
        rightm = SampleFeature[:, idControl, :]
        DNLL[0:numFea] = DNLL[0:numFea] + np.dot(leftm, rightm)
    return (DNLL / (numSample + 0.0))

def GradientNegativeLogLikelihoodUnc(theta, SampleFeature, SampleControl):
    numAction = SampleFeature.shape[1]
    numFea = SampleFeature.shape[2]
    numSample = SampleControl.shape[0]
    theta = theta[0:numFea]
    StateActionProb = np.dot(SampleFeature, theta)
    StateActionProb = np.exp(StateActionProb)
    sumStateActionProb = np.sum(StateActionProb, 1, keepdims=True)
    StateActionProb = StateActionProb / repmat(sumStateActionProb, 1, numAction)

    DNLL = np.zeros((numFea))
    for idControl in range(numAction):
        leftm = (StateActionProb[:, idControl] - (SampleControl == idControl))
        rightm = SampleFeature[:, idControl, :]
        DNLL = DNLL + np.dot(leftm, rightm)
    return (DNLL / (numSample + 0.0))

def LogsticRegressionWithConstrain(FeaturesTrain,ControlTrain,FeaturesCV,ControlCV,PossibleC,showInfo=True):
    # Generate matrices for linear constrants
    numFea = FeaturesTrain.shape[2]
    G = np.zeros((2 * numFea + 1, 2 * numFea))
    h = np.zeros(2 * numFea + 1)
    h[2 * numFea] = 1

    for id in range(numFea):
        index = [id, id + numFea]
        G[2 * id, index] = [1, -1]
        G[2 * id + 1, index] = [-1, -1]

    G[2 * numFea, numFea:2 * numFea + 1] = 1

    bestTheta = None
    bestNLL = None

    if showInfo:
        print('')

    thisTheta = np.ones((2 * numFea, 1))
    for i in range(len(PossibleC)):
        C = PossibleC[i]
        TrainObjectFun = lambda theta: NegativeLogLikelihood(theta, FeaturesTrain, ControlTrain)
        TrainObjectFunGrad = lambda theta: GradientNegativeLogLikelihood(theta, FeaturesTrain, ControlTrain)

        cons = ({'type': 'ineq', 'fun': lambda x: -(np.dot(G, x) - C * h)})
        thisTheta = minimize(TrainObjectFun, thisTheta.copy(), jac=TrainObjectFunGrad, constraints=cons,
                             tol=1e-12)

        CVObjectFun = lambda theta: NegativeLogLikelihood(theta, FeaturesCV, ControlCV)

        thisTheta = np.array(thisTheta["x"])
        thisNLL = CVObjectFun(thisTheta)

        if bestTheta is None:
            bestTheta = thisTheta[0:numFea]
            bestNLL = thisNLL
            if showInfo:
                print("\r C=", C, "NLL=", bestNLL, end='', flush=True)
        elif bestNLL > thisNLL:
            bestTheta = thisTheta[0:numFea]
            bestNLL = thisNLL
            if showInfo:
                print("\r C=", C, "NLL=", bestNLL, end='', flush=True)
    if showInfo:
        print('')
    return bestTheta

def LogsticRegressionWithoutConstrain(FeaturesTrain,ControlTrain,C):
    numFea = FeaturesTrain.shape[2]
    G = np.zeros((2 * numFea, numFea))
    h = np.zeros(2 * numFea)

    for i in range(numFea):
        G[i,i] = 1
        h[i] = -1
        G[i+numFea,i] = -1
        h[i+numFea] = -1

    cons = ({'type': 'ineq', 'fun': lambda x: (np.dot(G, x) - C * h)})

    TrainObjectFun = lambda theta: NegativeLogLikelihood(theta, FeaturesTrain, ControlTrain)
    TrainObjectFunGrad = lambda theta: GradientNegativeLogLikelihoodUnc(theta, FeaturesTrain, ControlTrain)
    thisTheta = minimize(TrainObjectFun, np.ones((numFea, 1)), jac=TrainObjectFunGrad, constraints=cons, tol=1e-10)
    return thisTheta["x"]