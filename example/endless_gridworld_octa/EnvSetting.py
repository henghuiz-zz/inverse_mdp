from irl.environment.endlessgridworldocta import *
import numpy as np

def gaussian(distance, peak, std_deviation):
    return peak * np.exp(-(distance * distance) / (2 * std_deviation * std_deviation)) / ( std_deviation * np.sqrt(2 * np.pi))

gridSize = (13, 13)
noise = 0.2
controlName = ['left','right']
actiondict = {'left': 0,
              'right': 1}

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
                source_rewards[newid] = 20
            else:
                source_rewards[newid] = 1

Rewards = np.zeros([gridSize[0], gridSize[1]])

for x in range(gridSize[0]):
    for y in range(gridSize[1]):
        for s in range(len(source)):
            distance = np.linalg.norm(source[s] - [x, y])
            Rewards[x][y] += gaussian(distance, source_rewards[s], 2.5)

mdp = EndlessGridworldOcta(Rewards)
mdp.setNoise(noise)
mdp.setSource(source)
env = EndlessGridworldEnvironment(mdp)

numState = gridSize[0] * gridSize[1] * 8
numFea = mdp.numberActorFea
sub2ind = lambda theta : np.ravel_multi_index([[theta[0]], [theta[1]], [theta[2]]], (gridSize[0], gridSize[1], 8))[0]
ind2sub = lambda theta : np.unravel_index(theta, (gridSize[0], gridSize[1], 8))