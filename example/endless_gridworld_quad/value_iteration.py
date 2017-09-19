import sys
import numpy as np

sys.path.append('../..')
from irl.environment.endless_grid_world import gaussian, EndlessGridWorld
from irl.agent.value_iteration_agent import ValueIterationAgent

grid_size = (11, 11)

waypoint_4_policy = []
for x in range(0, 11, 2):
    for y in range(0, 11, 2):
        waypoint_4_policy.append((x, y))

rewards = np.zeros(grid_size)

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        distance = np.linalg.norm([x, y])
        rewards[x][y] += gaussian(distance, 20, 5)

grid_world_mdp = EndlessGridWorld(rewards, waypoint_4_policy)

iterNum = 1000
discount = 0.9
a = ValueIterationAgent(grid_world_mdp, discount, iterNum)

action_name = 'NEWS'

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        state_instance = (x, y)
        action_instance = a.get_action(state_instance)
        print(action_name[action_instance], end='')
    print('')

#
# num_state = grid_size[0] * grid_size[1]
# num_action = len(id_2_action)
#
# num_policy_feature = len(waypoint_4_policy)
# num_transition_feature = len(waypoint_4_transition)
#
# sub_2_ind = lambda theta: np.ravel_multi_index([[theta[0]], [theta[1]]], (grid_size[0], grid_size[1]))[0]
# ind_2_sub = lambda theta: np.unravel_index(theta, (grid_size[0], grid_size[1]))