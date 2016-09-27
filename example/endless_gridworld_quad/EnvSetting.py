from irl.environment.endlessgridworld import *
import numpy as np

def gaussian(distance, peak, std_deviation):
    return peak * np.exp(-(distance * distance) /
                         (2 * std_deviation * std_deviation)) /\
           (std_deviation * np.sqrt(2 * np.pi))

gridSize = (13, 13)
noise = 0.2

actiondict = {'north': 0,
              'east': 1,
              'south': 2,
              'west': 3}

controlName = ['north', 'east', 'south', 'west']

A = range(1, 13, 2)
lenA = len(A)

source = np.zeros((lenA * lenA, 2))
source_rewards = np.zeros(lenA * lenA)

for idx in range(lenA):
    for idy in range(lenA):
        newid = idx * lenA + idy
        source[newid, 0] = A[idx]
        source[newid, 1] = A[idy]

        if (A[idx] in (1, 11)) and (A[idy] in (1, 11)):
            if A[idx] == A[idy]:
                source_rewards[newid] = 10
            else:
                source_rewards[newid] = 1

rewards = np.zeros([gridSize[0], gridSize[1]])

for x in range(gridSize[0]):
    for y in range(gridSize[1]):
        for s in range(len(source)):
            distance = np.linalg.norm(source[s] - [x, y])
            rewards[x][y] += gaussian(distance, source_rewards[s], 1)

mdp = EndlessGridworld(rewards)
mdp.setNoise(noise)
mdp.setSource(source)
env = EndlessGridworldEnvironment(mdp)

numState = gridSize[0] * gridSize[1]
numFea = mdp.numberActorFea
sub2ind = lambda theta : np.ravel_multi_index([[theta[0]], [theta[1]]], (gridSize[0], gridSize[1]))[0]
ind2sub = lambda theta : np.unravel_index(theta, (gridSize[0], gridSize[1]))