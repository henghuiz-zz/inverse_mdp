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
reward_points = [[1, 1], [9, 9], [1, 9], [9, 1]]
reward_amounts = [10, 10, 1, 1]

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        for rp, ra in zip(reward_points, reward_amounts):
            distance = np.linalg.norm([x-rp[0], y-rp[1]])
            rewards[x][y] += gaussian(distance, ra, 5)

grid_world_mdp = EndlessGridWorld(rewards, waypoint_4_policy)

# finding the optimal control using value iteration
iterNum = 2000
discount = 0.99
a = ValueIterationAgent(grid_world_mdp, discount, iterNum)

action_name = 'NEWS'
print('')

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        state_instance = (x, y)
        action_instance = a.get_action(state_instance)
        print(action_name[action_instance], end='')
    print('')

# finding a sub-optimal solution using greedy search

greedy_action = np.zeros(grid_size)

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        reward_for_state = []
        for a in range(4):
            reward_for_action = 0
            next_states = grid_world_mdp.get_transition_states_and_prob((x, y), a)
            for next_state in next_states:
                reward_for_action += grid_world_mdp.get_reward((x, y), a, tuple(next_state))
            reward_for_state += reward_for_action
        greedy_action[x, y] = np.argmax(reward_for_state)

print(greedy_action)

#
# num_state = grid_size[0] * grid_size[1]
# num_action = len(id_2_action)
#
# num_policy_feature = len(waypoint_4_policy)
# num_transition_feature = len(waypoint_4_transition)
#
# sub_2_ind = lambda theta: np.ravel_multi_index([[theta[0]], [theta[1]]], (grid_size[0], grid_size[1]))[0]
# ind_2_sub = lambda theta: np.unravel_index(theta, (grid_size[0], grid_size[1]))