import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

matplotlib.rcParams.update({'font.size': 14})
np.set_printoptions(precision=2, suppress=True)

# load and merge data
load_prefix = '../../data/endless_grid_quad/reg*'

all_files = glob(load_prefix)
l1_reward = []
unc_reward = []

l1_theta = []
unc_theta = []

l1_time = []
unc_time = []


for file_ins in all_files:
    load_dict = pickle.load(open(file_ins, 'rb'))
    l1_reward.append(load_dict['l1_reward'])
    unc_reward.append(load_dict['unc_reward'])
    l1_theta.append(load_dict['l1_theta_list'])
    unc_theta.append(load_dict['unc_theta_list'])
    l1_time.append(load_dict['l1_time'])
    unc_time.append(load_dict['unc_time'])

possible_sample_ratio = load_dict['possible_sample_ratio']
greedy_reward = load_dict['greedy_reward']
vi_reward = load_dict['vi_reward']

# NOTE: remember to change this
num_state = 21 * 21
num_sample = np.ceil(num_state * possible_sample_ratio)

l1_theta = list(zip(*l1_theta))
l1_theta = [sum(item, []) for item in l1_theta]
unc_theta = list(zip(*unc_theta))
unc_theta = [sum(item, []) for item in unc_theta]


l1_mean_std = []

for theta_list in l1_theta:
    ave_cov = []
    for theta in theta_list:
        theta = np.abs(theta)
        theta_max = np.max(theta)

        num_10_pre = np.mean(theta < 0.1 * theta_max)

        ave_cov.append(num_10_pre)

    l1_mean_std.append(np.mean(ave_cov))


unc_mean_std = []

for theta_list in unc_theta:
    ave_cov = []
    for theta in theta_list:
        theta = np.abs(theta)
        theta_max = np.max(theta)

        num_10_pre = np.mean(theta < 0.1 * theta_max)

        ave_cov.append(num_10_pre)

    unc_mean_std.append(np.mean(ave_cov))

print('Average sparsity for l1 regularized policy:', np.mean(l1_mean_std))
print('Average sparsity for unregularized policy:', np.mean(unc_mean_std))

print('Average computation time for l1 regularized policy:', np.mean(l1_time))
print('Average computation time for unregularized policy:', np.mean(unc_time))

l1_reward = np.concatenate(l1_reward, axis=1)
l1_reward = np.sort(l1_reward, axis=1)
mean_l1_reward = np.mean(l1_reward, axis=1)

unc_reward = np.concatenate(unc_reward, axis=1)
unc_reward = np.sort(unc_reward, axis=1)
mean_unc_reward = np.mean(unc_reward, axis=1)

plt.figure(figsize=(7, 6))

plt.fill_between(num_sample, l1_reward[:, 90 - 1], l1_reward[:, 10 - 1], facecolor='blue', alpha=0.3)
plt.fill_between(num_sample, unc_reward[:, 90 - 1], unc_reward[:, 10 - 1], facecolor='yellow', alpha=0.3)

plt.plot(num_sample, mean_l1_reward, '-o',
         lw=2, color='blue', label='$\ell_1.$-regularized policy')
plt.plot(num_sample, mean_unc_reward, '-o',
         lw=2, color='yellow', label='unregularized policy')
plt.plot([np.min(num_sample), np.max(num_sample)],
         [vi_reward, vi_reward],
         lw=2, color='red', label='target policy')
plt.plot([np.min(num_sample), np.max(num_sample)],
         [greedy_reward, greedy_reward],
         lw=2, color='green', label='greedy policy')

plt.axis([np.min(num_sample), np.max(num_sample), 0, 3])

plt.xlabel("Number of samples", fontsize=18)
plt.ylabel("Average reward", fontsize=18)
plt.legend(loc='lower right', fontsize=14)
plt.savefig('../../data/endless_grid_quad/endless_grid_quad_compare.pdf')

plt.show()