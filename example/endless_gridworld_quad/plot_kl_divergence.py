import numpy as np
import pickle
import sys
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

from glob import glob
from numpy.matlib import repmat

matplotlib.rcParams.update({'font.size': 14})
sys.path.append('../../')


def calculate_stationary_prob(action_data_sample, trans_prob):
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
    return vecs


def calculate_state_action_prob(theta, features, trans_prob):
    # Calculate the probability that agent take action
    num_state = trans_prob.shape[0]
    num_action = trans_prob.shape[2]
    state_action_prob = np.dot(features, theta)
    state_action_prob = np.exp(state_action_prob)
    sum_state_action_prob = np.sum(state_action_prob, 1, keepdims=True)
    state_action_prob = state_action_prob / repmat(sum_state_action_prob, 1, num_action)

    return state_action_prob


def main():
    loaded_data = scipy.io.loadmat('../../data/endless_grid_quad/sample_from_vi.mat')
    vi_data_sample = loaded_data['vi_data_sample']
    trans_prob = loaded_data["trans_prob"]
    features = loaded_data["features"]

    stationary_vector = calculate_stationary_prob(vi_data_sample, trans_prob)

    # load thetas
    load_prefix = '../../data/endless_grid_quad/reg*'
    all_files = glob(load_prefix)

    l1_theta = []
    unc_theta = []

    for file_ins in all_files:
        load_dict = pickle.load(open(file_ins, 'rb'))
        l1_theta.append(load_dict['l1_theta_list'])
        unc_theta.append(load_dict['unc_theta_list'])

    l1_theta = list(zip(*l1_theta))
    l1_theta = [sum(item, []) for item in l1_theta]
    unc_theta = list(zip(*unc_theta))
    unc_theta = [sum(item, []) for item in unc_theta]

    possible_sample_ratio = load_dict['possible_sample_ratio']

    # NOTE: remember to change this
    num_state = 21 * 21
    num_sample = np.ceil(num_state * possible_sample_ratio)

    # choose one theta as example

    nll_l1_theta = []
    for theta_samples in l1_theta:
        all_nll = []
        for theta_ins in theta_samples:
            state_action_prob = calculate_state_action_prob(theta_ins, features, trans_prob)
            likelihood = state_action_prob[range(len(stationary_vector)), vi_data_sample[:, 1]]
            nll = np.dot(-np.log(likelihood), stationary_vector)[0]
            all_nll.append(nll)
        nll_l1_theta.append(np.mean(all_nll))

    nll_unc_theta = []
    for theta_samples in unc_theta:
        all_nll = []
        for theta_ins in theta_samples:
            state_action_prob = calculate_state_action_prob(theta_ins, features, trans_prob)
            likelihood = state_action_prob[range(len(stationary_vector)), vi_data_sample[:, 1]]
            nll = np.dot(-np.log(likelihood), stationary_vector)[0]
            all_nll.append(nll)
        nll_unc_theta.append(np.mean(all_nll))

    plt.figure(figsize=(7, 4))

    plt.plot(num_sample, nll_l1_theta, '-o',
             lw=3, label='$\ell_1.$-regularized policy')
    plt.plot(num_sample, nll_unc_theta, '-*',
             lw=3, label='unregularized policy')

    # plt.axis([np.min(num_sample), np.max(num_sample), 0, 3])

    plt.xlabel("Number of samples", fontsize=18)
    plt.ylabel("Negative Log-Likelihood", fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig('../../data/endless_grid_quad/endless_grid_world_nll.pdf')

    plt.show()


if __name__ == '__main__':
    main()

