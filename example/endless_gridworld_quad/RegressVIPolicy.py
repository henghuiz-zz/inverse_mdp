# This script is aim to regressed the policy generate by VI using a parametric model
import sys

sys.path.append('../../')
import scipy.io
import irl as IRL
import numpy as np

np.set_printoptions(precision=2, suppress=True)
from numpy.matlib import repmat


def calculate_rsp_average_reward(theta, Features, TransProb, RewardsVector):
    # Calculate the probability that agent take action
    numState = TransProb.shape[0]
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
    for i in range(1000):
        vecs = np.matmul(TransProbUnderTheta, vecs)

    vecs /= np.sum(vecs)
    allreward = np.dot(vecs.T, RewardsVector)
    return allreward[0, 0]


def calculate_average_reward(ActionDataSample, TransProb, RewardsVector):
    # Calculate the probability that agent take action
    numState = TransProb.shape[0]

    TranProbUnderAction = np.zeros((numState, numState))
    for i in range(numState):
        stateid = ActionDataSample[i, 0]
        action = ActionDataSample[i, 1]
        TranProbUnderAction[:, stateid] = TransProb[:, stateid, action]

    vecs = np.ones((numState, 1))
    vecs /= numState
    for i in range(1000):
        vecs = np.matmul(TranProbUnderAction, vecs)

    vecs /= np.sum(vecs)
    allreward = np.dot(vecs.T, RewardsVector)
    return allreward[0, 0]


if __name__ == '__main__':
    # Prepare the enviromnent
    from EnvSetting import *

    loaded_data = scipy.io.loadmat('../../data/EndLessGridWorldQuad/SampleFromVI.mat')
    vi_data_sample = loaded_data['VIDataSample']
    TransProb = loaded_data["TransProb"]
    Features = loaded_data["Features"]
    RewardsVector = loaded_data["RewardsVector"]
    GreedyDataSample = loaded_data['GreedyDataSample']

    # Transform to Feature space
    FeaturesSample = np.zeros((num_state, 4, num_feature))
    ActionSample = np.zeros((1, num_state))
    for i in range(num_state):
        stateid = vi_data_sample[i, 0]
        x, y = ind2sub(stateid)
        action = vi_data_sample[i, 1]
        ActionSample[0, i] = action
        for u in range(4):
            featureitem = Features[stateid, u, :]
            FeaturesSample[i][u][:] = featureitem

    VIReward = calculate_average_reward(vi_data_sample, TransProb, RewardsVector)
    Greedy = calculate_average_reward(GreedyDataSample, TransProb, RewardsVector)

    print(VIReward)
    print(Greedy)

    PossibleSampleRatio = np.arange(0.35, 1.0, 0.05)
    PossibleC = range(1, 202, 10)
    lenRate = len(PossibleSampleRatio)
    RewardvsRate = np.zeros(lenRate)
    UncRewardvsRate = np.zeros(lenRate)
    Trails = 10

    bestReward = -1000
    bestTheta = None

    ConReward = np.zeros((lenRate, Trails))
    UncReward = np.zeros((lenRate, Trails))

    for trailID in range(Trails):
        ind = np.arange(num_state, dtype=int)
        np.random.shuffle(ind)
        ThisFeaturesSample = FeaturesSample[ind, :, :]
        ThisActionSample = ActionSample[0, ind]
        for rateID in range(lenRate):
            SampleRatio = PossibleSampleRatio[rateID]
            numTry = np.ceil(num_state * SampleRatio)
            numTry = int(numTry)
            numTrain = numTry - 20
            TrainFeaturesSample = ThisFeaturesSample[0:numTrain, :, :]
            TrainActionSample = ThisActionSample[0:numTrain]
            CVFeaturesSample = ThisFeaturesSample[numTrain:numTry, :, :]
            CVActionSample = ThisActionSample[numTrain:numTry]
            thisTheta = IRL.logstic_regression_with_constrain(TrainFeaturesSample, TrainActionSample, CVFeaturesSample,
                                                              CVActionSample, PossibleC, showInfo=False)

            thisReward = calculate_rsp_average_reward(thisTheta, Features, TransProb, RewardsVector)
            print("\r", trailID, rateID, thisReward, end='', flush=True)
            if thisReward > bestReward:
                bestReward = thisReward
                bestTheta = thisTheta.copy()
            ConReward[rateID, trailID] = thisReward
            RewardvsRate[rateID] += thisReward

            thisTheta = IRL.LogsticRegressionWithoutConstrain(TrainFeaturesSample, TrainActionSample, 3)
            thisReward = calculate_rsp_average_reward(thisTheta, Features, TransProb, RewardsVector)
            if thisReward > bestReward:
                bestReward = thisReward
                bestTheta = thisTheta.copy()
            UncReward[rateID, trailID] = thisReward
            UncRewardvsRate[rateID] += thisReward

    RewardvsRate /= Trails
    print(RewardvsRate)

    filename = '../../data/EndLessGridWorldQuad/RegressResult2.mat'
    scipy.io.savemat(filename, {"bestTheta": bestTheta,
                                "VIReward": VIReward,
                                "Greedy": Greedy,
                                "PossibleSampleRatio": PossibleSampleRatio,
                                "ConReward": ConReward,
                                "UncReward": UncReward})
