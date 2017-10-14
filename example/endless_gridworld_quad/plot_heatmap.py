import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def gaussian(distance, peak, std_deviation):
    return peak * np.exp(-(distance * distance) /
                         (2 * std_deviation * std_deviation)) / \
           (std_deviation * np.sqrt(2 * np.pi))


grid_size = (21, 21)
noise = 0.2

action_dict = {'north': 0,
              'east': 1,
              'south': 2,
              'west': 3}

control_name = ['north', 'east', 'south', 'west']

A = range(1, 21, 2)
lenA = len(A)

source = np.zeros((lenA * lenA, 2))
source_rewards = np.zeros(lenA * lenA)

for idx in range(lenA):
    for idy in range(lenA):
        newid = idx * lenA + idy
        source[newid, 0] = A[idx]
        source[newid, 1] = A[idy]

        if (A[idx] in (3, 17)) and (A[idy] in (3, 17)):
            if A[idx] == A[idy]:
                source_rewards[newid] = 10
            else:
                source_rewards[newid] = 1

rewards = np.zeros([grid_size[0], grid_size[1]])

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        for s in range(len(source)):
            distance = np.linalg.norm(source[s] - [x, y])
            rewards[x][y] += gaussian(distance, source_rewards[s], 1)


fig, ax = plt.subplots(figsize=(7, 6))

cax = ax.imshow(rewards, interpolation='None', cmap='coolwarm')
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.xticks([0, 5, 10, 15 ,20], fontsize=16)
plt.yticks([0, 5, 10, 15 ,20], fontsize=16)
cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3])
cbar.ax.tick_params(labelsize=16)

plt.savefig('../../data/EndLessGridWorldQuad/reward_heatmap.pdf')
plt.show()
