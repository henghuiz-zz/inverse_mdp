import numpy as np
from numpy.matlib import repmat
from scipy.optimize import minimize


def negative_log_likelihood(theta, SampleFeature, SampleControl):
    numAction = SampleFeature.shape[1]
    numFea = SampleFeature.shape[2]
    numSample = SampleControl.shape[0]
    theta = theta[0:numFea]
    StateActionProb = np.dot(SampleFeature, theta)
    StateActionProb = np.exp(np.clip(StateActionProb, -10, 10))
    sumStateActionProb = np.sum(StateActionProb, 1, keepdims=True)
    StateActionProb = StateActionProb / repmat(sumStateActionProb, 1, numAction)
    Prob = StateActionProb[range(numSample), SampleControl.astype("int")]
    NLL = -np.sum(np.log(Prob))
    return (NLL / (numSample + 0.0))


def gradient_negative_log_likelihood(theta, SampleFeature, SampleControl):
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


def gradient_negative_log_likelihood_unc(theta, SampleFeature, SampleControl):
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


def logstic_regression_with_constrain(features_train, control_train, FeaturesCV, ControlCV, PossibleC, show_info=True):
    # Generate matrices for linear constrants
    numFea = features_train.shape[2]
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

    if show_info:
        print('')

    thisTheta = np.ones((2 * numFea, 1)) / numFea / 2
    for i in range(len(PossibleC)):
        C = PossibleC[i]
        TrainObjectFun = lambda theta: negative_log_likelihood(theta, features_train, control_train)
        TrainObjectFunGrad = lambda theta: gradient_negative_log_likelihood(theta, features_train, control_train)

        cons = ({'type': 'ineq', 'fun': lambda x: -(np.dot(G, x) - C * h)})
        thisTheta = minimize(TrainObjectFun, thisTheta.copy(), jac=TrainObjectFunGrad, constraints=cons,
                             tol=1e-12)

        CVObjectFun = lambda theta: negative_log_likelihood(theta, FeaturesCV, ControlCV)

        thisTheta = np.array(thisTheta["x"])
        thisNLL = CVObjectFun(thisTheta)

        if bestTheta is None:
            bestTheta = thisTheta[0:numFea]
            bestNLL = thisNLL
            if show_info:
                print("\r C=", C, "NLL=", bestNLL, end='', flush=True)
        elif bestNLL > thisNLL:
            bestTheta = thisTheta[0:numFea]
            bestNLL = thisNLL
            if show_info:
                print("\r C=", C, "NLL=", bestNLL, end='', flush=True)
    if show_info:
        print('')
    return bestTheta


def logstic_regression_without_constrain(features_train, control_train, PossibleC, show_info=True):
    num_fea = features_train.shape[2]
    G = np.zeros((2 * num_fea, num_fea))
    h = np.zeros(2 * num_fea)

    for i in range(num_fea):
        G[i, i] = 1
        h[i] = -1
        G[i + num_fea, i] = -1
        h[i + num_fea] = -1


    # bestTheta = None
    # bestNLL = None

    C = 1000
    cons = ({'type': 'ineq', 'fun': lambda x: (np.dot(G, x) - C * h)})
    TrainObjectFun = lambda theta: negative_log_likelihood(theta, features_train, control_train)
    TrainObjectFunGrad = lambda theta: gradient_negative_log_likelihood_unc(theta, features_train, control_train)
    thisTheta = minimize(TrainObjectFun, np.ones((num_fea, 1))/num_fea, jac=TrainObjectFunGrad, constraints=cons, tol=1e-10)
    thisTheta = np.array(thisTheta["x"])
    bestTheta = thisTheta[0:num_fea]
    # for C in PossibleC:
    #     cons = ({'type': 'ineq', 'fun': lambda x: (np.dot(G, x) - C * h)})
    #     TrainObjectFun = lambda theta: negative_log_likelihood(theta, features_train, control_train)
    #     TrainObjectFunGrad = lambda theta: gradient_negative_log_likelihood_unc(theta, features_train, control_train)
    #     thisTheta = minimize(TrainObjectFun, np.ones((num_fea, 1)), jac=TrainObjectFunGrad, constraints=cons, tol=1e-10)
    #
    #     CVObjectFun = lambda theta: negative_log_likelihood(theta, FeaturesCV, ControlCV)
    #
    #     thisTheta = np.array(thisTheta["x"])
    #     thisNLL = CVObjectFun(thisTheta)
    #
    #     if bestTheta is None:
    #         bestTheta = thisTheta[0:num_fea]
    #         bestNLL = thisNLL
    #         if show_info:
    #             print("\r C=", C, "NLL=", bestNLL, end='', flush=True)
    #     elif bestNLL > thisNLL:
    #         bestTheta = thisTheta[0:num_fea]
    #         bestNLL = thisNLL
    #         if show_info:
    #             print("\r C=", C, "NLL=", bestNLL, end='', flush=True)

    return bestTheta
