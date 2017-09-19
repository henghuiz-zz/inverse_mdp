import sys
sys.path.append('../../')

import irl.irl as IRL
from numpy.matlib import repmat
from multiprocessing import Pool, TimeoutError
import numpy as np
from EnvSetting import *



def CalculateRSPAverageReward(theta,Features,TransProb,RewardsVector):
    # Calculate the probability that agent take action
    numState  = TransProb.shape[0]
    numAction = TransProb.shape[2]
    StateActionProb = np.dot(Features, theta)
    StateActionProb = np.exp(StateActionProb)
    sumStateActionProb = np.sum(StateActionProb, 1, keepdims=True)
    StateActionProb = StateActionProb / repmat(sumStateActionProb, 1, numAction)

    TransProbUnderTheta = np.zeros((numState, numState))
    for x in range(numState):
        for y in range(numState):
            TransProbUnderTheta[x][y] += np.dot(TransProb[x, y, :], StateActionProb[y, :])

    vecs = np.ones((numState, 1))
    vecs /= numState
    for i in range(500):
        vecs = np.matmul(TransProbUnderTheta, vecs)

    vecs /= np.sum(vecs)
    allreward = np.dot(vecs.T, RewardsVector)
    return allreward[0,0]

def CalculateAverageReward(ActionDataSample,TransProb,RewardsVector):
    # Calculate the probability that agent take action
    numState  = TransProb.shape[0]

    TranProbUnderAction = np.zeros((numState, numState))
    for i in range(numState):
        stateid = ActionDataSample[i, 0]
        action = ActionDataSample[i, 1]
        TranProbUnderAction[:,stateid] = TransProb[:,stateid,action]

    vecs = np.ones((numState, 1))
    vecs /= numState
    for i in range(500):
        vecs = np.matmul(TranProbUnderAction, vecs)

    vecs /= np.sum(vecs)
    allreward = np.dot(vecs.T, RewardsVector)
    return allreward[0,0]

def multi_run_wrapper(args):
   return oneRoundEvaluation(*args)

def oneRoundEvaluation(numState,FeaturesSample,ActionSample,Features,TransProb,RewardsVector):
    ind = np.arange(numState, dtype=int)
    np.random.shuffle(ind)
    ThisFeaturesSample = FeaturesSample[ind, :, :]
    ThisActionSample = ActionSample[ind]

    PossibleNumTry = range(50,501,50)
    lenNT = len(PossibleNumTry)

    RewardCon = np.zeros(lenNT)
    RewardUnc = np.zeros(lenNT)

    for idNT in range(lenNT):
        numTrain = PossibleNumTry[idNT]
        numTrain = int(numTrain)
        numTry = int(numTrain + numTrain/2)
        TrainFeaturesSample = ThisFeaturesSample[0:numTrain, :, :]
        TrainActionSample = ThisActionSample[0:numTrain]
        CVFeaturesSample = ThisFeaturesSample[numTrain:numTry, :, :]
        CVActionSample = ThisActionSample[numTrain:numTry]
        PossibleC = np.arange(10, 200, 4)

        thisTheta = IRL.logstic_regression_with_constrain(TrainFeaturesSample, TrainActionSample, CVFeaturesSample,
                                                          CVActionSample, PossibleC, showInfo=False)
        RewardCon[idNT] = CalculateRSPAverageReward(thisTheta, Features, TransProb, RewardsVector)

        thisTheta = IRL.LogsticRegressionWithoutConstrain(TrainFeaturesSample, TrainActionSample, 10+(1.0*idNT/lenNT*(30-10)))
        RewardUnc[idNT] = CalculateRSPAverageReward(thisTheta, Features, TransProb, RewardsVector)
    return RewardCon, RewardUnc

if __name__ == '__main__':
    # Define the environment


    # Load the datasample
    import scipy.io
    filename = '../../data/EndLessGridWorldOcta/Samples.mat'
    LoadedData = scipy.io.loadmat(filename)
    TransProb = LoadedData["TransProb"]
    VIDataSample = LoadedData["VIDataSample"]
    Features = LoadedData["Features"]
    RewardsVector = LoadedData["RewardsVector"]
    GreedyDataSample = LoadedData["GreedyDataSample"]


    VIReward = CalculateAverageReward(VIDataSample,TransProb,RewardsVector)
    print("VIReward",VIReward)
    GreedyReward = CalculateAverageReward(GreedyDataSample, TransProb, RewardsVector)
    print("GreedyReward", GreedyReward)

    # Transform to Feature space
    FeaturesSample = np.zeros((numState, 2, numFea))
    ActionSample   = np.zeros(numState)
    for i in range(numState):
        stateid = VIDataSample[i,0]
        action = VIDataSample[i,1]
        ActionSample[i] = action
        for u in range(2):
            featureitem = Features[stateid,u,:]
            FeaturesSample[i][u][:] = featureitem

    # # Try unconstrained regression first
    BestTheta = IRL.LogsticRegressionWithoutConstrain(FeaturesSample,ActionSample,20)
    print(BestTheta)
    BestReward = CalculateRSPAverageReward(BestTheta,Features,TransProb,RewardsVector)
    print(BestReward)
    print("Done")
    pool = Pool(processes=8)
    numTrail = 100
    # Prepare arguement
    arg = []
    for i in range(numTrail):
        arg.append((numState,FeaturesSample,ActionSample,Features,TransProb,RewardsVector))

    Result = pool.map(multi_run_wrapper, arg)

    RewardCon,RewardUnc = zip(*Result)
    #RewardCon, RewardUnc = oneRoundEvaluation(numState,FeaturesSample,ActionSample,Features,TransProb,RewardsVector)

    print(np.mean(RewardCon,axis=0))
    print("")
    print(np.mean(RewardUnc,axis=0))
    RewardCon = np.array(RewardCon)
    RewardUnc = np.array(RewardUnc)

    filename = '../../data/EndLessGridWorldOcta/Regression.mat'
    scipy.io.savemat(filename, {"RewardCon": RewardCon,
                                "RewardUnc":RewardUnc,
                               "VIReward": VIReward,
                               "GreedyReward": GreedyReward,
                               "BestReward": BestReward})
