import numpy as np
from EnvSetting import *

def GenerateTranProbandFeature(gridSize,mdp,numState):
    # generate transition probability and feature
    TransProb = np.zeros((numState,numState,2))
    Features = np.zeros((numState, 2, numFea))
    RewardsVector = np.zeros((numState,1))

    for x in range(gridSize[0]):
        for y in range(gridSize[1]):
            for theta in range(8):
                fromid = sub2ind((x,y,theta))
                RewardsVector[fromid,0] = mdp.grid[x][y]
                for u in range(2):
                    nextstateprob = mdp.getTransitionStatesAndProbs((x, y,theta), controlName[u])
                    for item in nextstateprob:
                        toid = sub2ind(item[0])
                        TransProb[toid][fromid][u] = item[1]

                    featureitem = mdp.getActorFeature((x, y, theta), controlName[u])
                    Features[fromid][u][:] = featureitem

    return TransProb, Features, RewardsVector

def FindVIDataSamples(iterNum,discount,mdp,numState):
    import irl.agent.VIAgent as valueIterationAgents
    a = valueIterationAgents.ValueIterationAgent(mdp, discount, iterNum)

    VIDataSample = np.zeros((numState, 2), dtype=int)
    for x in range(gridSize[0]):
        for y in range(gridSize[1]):
            for theta in range(8):
                thisAction = actiondict[a.getAction((x, y, theta))]
                thisid = sub2ind((x, y, theta))
                VIDataSample[thisid, 0] = thisid
                VIDataSample[thisid, 1] = thisAction
    return VIDataSample

def FindGreedyPolicy(mdp,numState):
    # Greedy policy
    GreedyDataSample = np.zeros((numState,2),dtype=int)
    for x in range(gridSize[0]):
        for y in range(gridSize[1]):
            for theta in range(8):
                thisid = sub2ind((x, y, theta))

                maximunRew = -100
                nextaction = None
                for a in range(2):
                    action = controlName[a]
                    stateProb = mdp.getTransitionStatesAndProbs((x, y, theta), action)
                    allnextState, Prob = zip(*stateProb)
                    nextState = allnextState[np.argmax(Prob)]
                    rew = mdp.grid[nextState[0]][nextState[1]]

                    if rew > maximunRew:
                        nextaction = a
                        maximunRew = rew

                GreedyDataSample[thisid, 0] = thisid
                GreedyDataSample[thisid, 1] = nextaction

    return GreedyDataSample

if __name__ == '__main__':

    #
    TransProb, Features, RewardsVector = GenerateTranProbandFeature(gridSize, mdp, numState)

    print('')
    print("---Transition probability and feature generation complete---")
    print('')

    # VI to learn best control
    iterNum = 500
    discount = 0.9
    VIDataSample = FindVIDataSamples(iterNum,discount,mdp,numState)

    print('')
    print("---Value Iteration Completed---")
    print('')

    # Find greedy policy
    GreedyDataSample = FindGreedyPolicy(mdp,numState)
    print('')
    print("---Greedy Policy Estiablish---")
    print('')

    # Save DataSample to files
    import scipy.io
    import os
    filename = '../../data/EndLessGridWorldOcta/Samples.mat'
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    scipy.io.savemat(filename, {"TransProb": TransProb,
                                "Features":Features,
                                "Rewards":Rewards,
                                "VIDataSample":VIDataSample,
                                "GreedyDataSample":GreedyDataSample,
                                "RewardsVector":RewardsVector})

    print('')
    print("---Data save complete---")
    print('')