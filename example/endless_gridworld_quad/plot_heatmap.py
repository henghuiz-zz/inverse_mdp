import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from irl.environment.endless_grid_world import gaussian

grid_size = (21, 21)
num_state = grid_size[0] * grid_size[1]
num_action = 4

waypoint_4_policy = []
for x in range(0, 21, 2):
    for y in range(0, 21, 2):
        waypoint_4_policy.append((x, y))

num_feature = len(waypoint_4_policy)

rewards = np.zeros(grid_size)
reward_points = [[3, 3], [17, 17], [3, 17], [17, 3]]
reward_amounts = [1, 1, 10, 10]

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        for rp, ra in zip(reward_points, reward_amounts):
            distance = np.linalg.norm([x - rp[0], y - rp[1]])
            rewards[x][y] += gaussian(distance, ra, 1)


fig, ax = plt.subplots(figsize=(7, 6))

cax = ax.imshow(rewards, interpolation='None', cmap='coolwarm')
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.xticks([0, 5, 10, 15 ,20], fontsize=16)
plt.yticks([0, 5, 10, 15 ,20], fontsize=16)
cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3])
cbar.ax.tick_params(labelsize=16)

# plt.savefig('../../data/EndLessGridWorldQuad/reward_heatmap.pdf')
plt.show()
