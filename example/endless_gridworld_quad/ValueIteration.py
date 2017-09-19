import sys
import cv2
sys.path.append('../../')

if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(precision=2,suppress=True)
    iter_num = 5000
    discount = 0.9

    # Define the environment
    from EnvSetting import *

    trans_prob = np.zeros((num_state, num_state, 4))
    features = np.zeros((num_state, 4, num_feature))
    RewardsVector = np.zeros((num_state, 1))

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            fromid = sub2ind((x, y))
            RewardsVector[fromid, 0] = mdp.grid[x][y]
            for u in range(4):
                nextstateprob = mdp.getTransitionStatesAndProbs((x, y), control_name[u])
                for item in nextstateprob:
                    toid = sub2ind(item[0])
                    trans_prob[toid][fromid][u] = item[1]

                featureitem = mdp.getActorFeature((x, y), control_name[u])
                features[fromid][u][:] = featureitem

    print('')
    print("---Transition probability and feature generation complete---")
    print('')

    from irl.agent.value_iteration_agent import ValueIterationAgent
    print('')
    a = ValueIterationAgent(mdp, discount, iter_num)

    Value = np.zeros([grid_size[0], grid_size[1]])
    VIAction = np.zeros([grid_size[0], grid_size[1]])
    VIDataSample = np.zeros((num_state, 2), dtype=int)

    for x in range(grid_size[0]):
       for y in range(grid_size[1]):
           Value[x][y] = a.values[(x,y)]
           thisAction = action_dict[a.getAction((x, y))]
           VIAction[x][y] = thisAction
           DataIndex = sub2ind((x,y))
           VIDataSample[DataIndex, 0] = DataIndex
           VIDataSample[DataIndex, 1] = thisAction

    print('')
    print(VIAction)

    # Visualize action
    # import pygame
    # import irl.graphunit.gird_world_display as Display
    #
    # display = Display.GridWorldWindows(grid_size[0], grid_size[1], unitsize=50, colored=False, action=VIAction)
    # display.grid = Value
    # display.Update()
    #
    # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    # image_data = cv2.flip(image_data, 1)
    # image_data = np.rot90(image_data)
    # cv2.imwrite('../../data/EndLessGridWorldQuad/VI.png', image_data)

    print('')
    print("---Value itration complete---")
    print('')

    GreedyDataSample = np.zeros((num_state, 2), dtype=int)
    GreedyAction = np.zeros([grid_size[0], grid_size[1]], dtype=int)

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
                thisid = sub2ind((x, y))
                maximunRew = -100
                nextaction = None
                for a in range(4):
                    action = control_name[a]
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

    # display.action=GreedyAction
    # display.grid = rewards
    # display.Update()
    #
    # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    # image_data = cv2.flip(image_data, 1)
    # image_data = np.rot90(image_data)
    # cv2.imwrite('../../data/EndLessGridWorldQuad/Greedy.png', image_data)

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

    scipy.io.savemat(filename, {"TransProb": trans_prob,
                               "Features": features,
                               "VIDataSample": VIDataSample,
                               "RewardsVector": RewardsVector,
                               "GreedyDataSample": GreedyDataSample})
    #
    # while True:
    #    for event in pygame.event.get():
    #        if event.type == pygame.QUIT:
    #            display.Quit()

