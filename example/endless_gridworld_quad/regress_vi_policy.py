# This script is aim to regressed the policy generate by VI using a parametric model
import sys
import time
import pickle
import argparse

sys.path.append('../../')
import scipy.io
import irl as IRL
import numpy as np

np.set_printoptions(precision=2, suppress=True)
from numpy.matlib import repmat


def calculate_rsp_average_reward(theta, features, trans_prob, rewards_vector):
    # Calculate the probability that agent take action
    num_state = trans_prob.shape[0]
    num_action = trans_prob.shape[2]
    state_action_prob = np.dot(features, theta)
    state_action_prob = np.exp(state_action_prob)
    sum_state_action_prob = np.sum(state_action_prob, 1, keepdims=True)
    state_action_prob = state_action_prob / repmat(sum_state_action_prob, 1, num_action)

    trans_prob_under_theta = np.zeros((num_state, num_state))
    for x in range(num_state):
        for y in range(num_state):
            trans_prob_under_theta[x][y] += np.dot(trans_prob[x, y, :], state_action_prob[y, :])

    vecs = np.ones((num_state, 1))
    vecs /= num_state
    for i in range(1000):
        vecs = np.matmul(trans_prob_under_theta, vecs)

    vecs /= np.sum(vecs)
    allreward = np.dot(vecs.T, rewards_vector)
    return allreward[0, 0]


def calculate_average_reward(action_data_sample, trans_prob, rewards_vector):
    # Calculate the probability that agent take action
    num_state = trans_prob.shape[0]

    tran_prob_under_action = np.zeros((num_state, num_state))
    for i in range(num_state):
        stateid = action_data_sample[i, 0]
        action = action_data_sample[i, 1]
        tran_prob_under_action[:, stateid] = trans_prob[:, stateid, action]

    vecs = np.ones((num_state, 1))
    vecs /= num_state
    for i in range(1000):
        vecs = np.matmul(tran_prob_under_action, vecs)

    vecs /= np.sum(vecs)
    all_reward = np.dot(vecs.T, rewards_vector)
    return all_reward[0, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_id', type=int, default=0, help='an integer for the accumulator')
    args = parser.parse_args()
    save_id = args.save_id

    # Prepare the enviromnent
    loaded_data = scipy.io.loadmat('../../data/endless_grid_quad/sample_from_vi.mat')
    vi_data_sample = loaded_data['vi_data_sample']
    trans_prob = loaded_data["trans_prob"]
    features = loaded_data["features"]
    rewards_vector = loaded_data["rewards_vector"]
    greedy_data_sample = loaded_data['greedy_data_sample']

    num_state = trans_prob.shape[0]
    num_feature = features.shape[2]

    grid_size = int(np.sqrt(num_state))

    # build mapping for subscript to index
    sub2ind = lambda theta: np.ravel_multi_index([[theta[0]], [theta[1]]], (grid_size, grid_size))[0]
    ind2sub = lambda theta: np.unravel_index(theta, (grid_size, grid_size))

    # transform to feature space
    features_sample = np.zeros((num_state, 4, num_feature))
    action_sample = np.zeros((1, num_state))
    for i in range(num_state):
        stateid = vi_data_sample[i, 0]
        x, y = ind2sub(stateid)
        action = vi_data_sample[i, 1]
        action_sample[0, i] = action
        for u in range(4):
            featureitem = features[stateid, u, :]
            features_sample[i][u][:] = featureitem

    vi_reward = calculate_average_reward(vi_data_sample, trans_prob, rewards_vector)
    greedy_reward = calculate_average_reward(greedy_data_sample, trans_prob, rewards_vector)

    print(vi_reward)
    print(greedy_reward)

    # TODO: change this back
    possible_sample_ratio = np.arange(0.1, 1.0, 0.05)
    possible_c = range(1, 101, 5)
    num_rate = len(possible_sample_ratio)
    l1_reward_mean = np.zeros(num_rate)
    unc_reward_mean = np.zeros(num_rate)
    # TODO: change this back
    num_trial = 2

    best_reward = -1000
    best_theta = None

    l1_reward = np.zeros((num_rate, num_trial))
    unc_reward = np.zeros((num_rate, num_trial))

    l1_time = np.zeros(num_rate)
    unc_time = np.zeros(num_rate)

    l1_theta_list = [[] for _ in range(num_rate)]
    unc_theta_list = [[] for _ in range(num_rate)]

    for trail_id in range(num_trial):
        ind = np.arange(num_state, dtype=int)
        np.random.shuffle(ind)
        this_features_sample = features_sample[ind, :, :]
        this_action_sample = action_sample[0, ind]
        for rate_id in range(num_rate):
            sample_ratio = possible_sample_ratio[rate_id]
            num_train_all = int(np.ceil(num_state * sample_ratio))
            num_train = num_train_all - 20

            train_features_sample = this_features_sample[0:num_train, :, :]
            train_action_sample = this_action_sample[0:num_train]
            validate_features_sample = this_features_sample[num_train:num_train_all, :, :]
            validate_action_sample = this_action_sample[num_train:num_train_all]

            start_time = time.time()
            l1_theta = IRL.logstic_regression_with_constrain(
                train_features_sample, train_action_sample,
                validate_features_sample, validate_action_sample,
                possible_c, show_info=False)
            elapsed_time = time.time() - start_time
            l1_time[rate_id] += elapsed_time
            l1_theta_list[rate_id].append(l1_theta)

            l1_reward_ins = calculate_rsp_average_reward(l1_theta, features, trans_prob, rewards_vector)
            print("\r", trail_id, rate_id, l1_reward_ins, end='', flush=True)
            if l1_reward_ins > best_reward:
                best_reward = l1_reward_ins
                best_theta = l1_theta.copy()
            l1_reward[rate_id, trail_id] = l1_reward_ins
            l1_reward_mean[rate_id] += l1_reward_ins

            start_time = time.time()
            unc_theta = IRL.logstic_regression_without_constrain(
                this_features_sample[0:num_train_all, :, :], this_action_sample[0:num_train_all],
                possible_c, show_info=False)
            elapsed_time = time.time() - start_time
            unc_time[rate_id] += elapsed_time
            unc_theta_list[rate_id].append(unc_theta)

            unc_reward_ins = calculate_rsp_average_reward(unc_theta, features, trans_prob, rewards_vector)
            if unc_reward_ins > best_reward:
                best_reward = unc_reward_ins
                best_theta = unc_theta.copy()
            unc_reward[rate_id, trail_id] = unc_reward_ins
            unc_reward_mean[rate_id] += unc_reward_ins

    l1_reward_mean /= num_trial
    unc_reward_mean /= num_trial
    print('')
    print(l1_reward_mean)
    print(unc_reward_mean)

    saver_dict = {'l1_reward': l1_reward, 'unc_reward': unc_reward,
                  'l1_time': l1_time / num_trial, 'unc_time': unc_time / num_trial,
                  'l1_theta_list': l1_theta_list, 'unc_theta_list':unc_theta_list,
                  'possible_sample_ratio': possible_sample_ratio,
                  'greedy_reward': greedy_reward, 'vi_reward': vi_reward
                  }

    filename = '../../data/endless_grid_quad/regress_result' + str(save_id) + '.p'
    pickle.dump(saver_dict, open(filename, 'wb'))
