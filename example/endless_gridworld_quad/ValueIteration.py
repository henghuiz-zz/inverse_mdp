import sys
import cv2
sys.path.append('../../')

if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(precision=2,suppress=True)
    iterNum = 1000
    discount = 0.9

    # Define the environment
    from EnvSetting import *

    TransProb = np.zeros((numState, numState, 4))
    Features = np.zeros((numState, 4, numFea))
    RewardsVector = np.zeros((numState, 1))

    for x in range(gridSize[0]):
        for y in range(gridSize[1]):
            fromid = sub2ind((x, y))
            RewardsVector[fromid, 0] = mdp.grid[x][y]
            for u in range(4):
                nextstateprob = mdp.getTransitionStatesAndProbs((x, y), controlName[u])
                for item in nextstateprob:
                    toid = sub2ind(item[0])
                    TransProb[toid][fromid][u] = item[1]

                featureitem = mdp.getActorFeature((x, y), controlName[u])
                Features[fromid][u][:] = featureitem

    print('')
    print("---Transition probability and feature generation complete---")
    print('')

    from irl.agent.VIAgent import ValueIterationAgent
    print('')
    a = ValueIterationAgent(mdp, discount, iterNum)

    Value = np.zeros([gridSize[0], gridSize[1]])
    VIAction = np.zeros([gridSize[0], gridSize[1]])
    VIDataSample = np.zeros((numState,2),dtype=int)

    for x in range(gridSize[0]):
       for y in range(gridSize[1]):
           Value[x][y] = a.values[(x,y)]
           thisAction = actiondict[a.getAction((x,y))]
           VIAction[x][y] = thisAction
           DataIndex = sub2ind((x,y))
           VIDataSample[DataIndex, 0] = DataIndex
           VIDataSample[DataIndex, 1] = thisAction

    # Visualize action
    import pygame
    import irl.graphunit.gird_world_display as Display

    display = Display.GridWorldWindows(gridSize[0], gridSize[1], unitsize=50, colored=False, action=VIAction)
    display.grid = Value
    display.Update()

    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    image_data = cv2.flip(image_data, 1)
    image_data = np.rot90(image_data)
    cv2.imwrite('../../data/EndLessGridWorldQuad/VI.png', image_data)

    print('')
    print("---Value itration complete---")
    print('')

    GreedyDataSample = np.zeros((numState,2),dtype=int)
    GreedyAction = np.zeros([gridSize[0], gridSize[1]],dtype=int)

    for x in range(gridSize[0]):
        for y in range(gridSize[1]):
                thisid = sub2ind((x, y))
                maximunRew = -100
                nextaction = None
                for a in range(4):
                    action = controlName[a]
                    stateProb = mdp.getTransitionStatesAndProbs((x, y), action)
                    allnextState, Prob = zip(*stateProb)
                    nextState = allnextState[np.argmax(Prob)]
                    rew = mdp.grid[nextState[0]][nextState[1]]

                    if rew > maximunRew:
                        nextaction = a
                        maximunRew = rew

                GreedyDataSample[thisid, 0] = thisid
                GreedyDataSample[thisid, 1] = nextaction
                GreedyAction[x,y] = nextaction

    display.action=GreedyAction
    display.grid = rewards
    display.Update()

    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    image_data = cv2.flip(image_data, 1)
    image_data = np.rot90(image_data)
    cv2.imwrite('../../data/EndLessGridWorldQuad/Greedy.png', image_data)

    print('')
    print("---Greedy Policy Estiablish---")
    print('')

    # Save DataSample to files
    import scipy.io
    import os
    filename = '../../data/EndLessGridWorldQuad/SampleFromVI.mat'
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
       os.makedirs(directory)

    scipy.io.savemat(filename, {"TransProb": TransProb,
                               "Features": Features,
                               "VIDataSample": VIDataSample,
                               "RewardsVector": RewardsVector,
                               "GreedyDataSample": GreedyDataSample})

    while True:
       for event in pygame.event.get():
           if event.type == pygame.QUIT:
               display.Quit()

