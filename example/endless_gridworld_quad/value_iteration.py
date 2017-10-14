import sys
import numpy as np
import scipy.io
import time


sys.path.append('../..')
from irl.environment.endless_grid_world import gaussian, EndlessGridWorld
from irl.agent.value_iteration_agent import ValueIterationAgent

grid_size = (21, 21)
num_state = grid_size[0] * grid_size[1]
num_action = 4

# build mapping for subscript to index
sub2ind = lambda theta: np.ravel_multi_index([[theta[0]], [theta[1]]], (grid_size[0], grid_size[1]))[0]
ind2sub = lambda theta: np.unravel_index(theta, (grid_size[0], grid_size[1]))

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

grid_world_mdp = EndlessGridWorld(rewards, waypoint_4_policy)

# # finding the optimal control using value iteration
iterNum = 2000
discount = 0.99
a = ValueIterationAgent(grid_world_mdp, discount, iterNum)

action_name = 'NEWS'
print('')

vi_data_sample = np.zeros((num_state, 2), dtype=int)
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        state_instance = (x, y)
        action_instance = a.get_action(state_instance)

        data_index = sub2ind((x, y))
        vi_data_sample[data_index, 0] = data_index
        vi_data_sample[data_index, 1] = action_instance

        print(action_name[action_instance], end='')
    print('')

# finding a sub-optimal solution using greedy search

greedy_action = np.zeros(grid_size)

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        reward_for_state = []
        for a in range(num_action):
            reward_for_action = 0
            next_states = grid_world_mdp.get_transition_states_and_prob((x, y), a)
            for next_state in next_states:
                reward_for_action += next_state[1] * grid_world_mdp.get_reward((x, y), a, tuple(next_state[0]))
            reward_for_state.append(reward_for_action)
        greedy_action[x, y] = np.argmax(reward_for_state)

action_name = 'NEWS'
print('')

start_time = time.time()
greedy_data_sample = np.zeros((num_state, 2), dtype=int)
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        action_instance = int(greedy_action[x, y])
        print(action_name[action_instance], end='')

        data_index = sub2ind((x, y))
        greedy_data_sample[data_index, 0] = data_index
        greedy_data_sample[data_index, 1] = action_instance
    print('')
elapsed_time = time.time() - start_time
print(elapsed_time)

trans_prob = np.zeros((num_state, num_state, num_action))
features = np.zeros((num_state, 4, num_feature))
rewards_vector = np.zeros((num_state, 1))

for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        from_id = sub2ind((x, y))
        rewards_vector[from_id, 0] = grid_world_mdp.reward[x][y]
        for u in range(4):
            next_state_prob = grid_world_mdp.get_transition_states_and_prob((x, y), u)
            for item in next_state_prob:
                to_id = sub2ind(item[0])
                trans_prob[to_id][from_id][u] = item[1]

            feature_item = grid_world_mdp.get_actor_feature((x, y), u)
            features[from_id][u][:] = feature_item

filename = '../../data/EndLessGridWorldQuad/sample_from_vi.mat'
scipy.io.savemat(filename, {"trans_prob": trans_prob,
                            "features": features,
                            "vi_data_sample": vi_data_sample,
                            "rewards_vector": rewards_vector,
                            "greedy_data_sample": greedy_data_sample})
