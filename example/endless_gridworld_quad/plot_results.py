import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 12})


# load and merge data
load_prefix = '../../data/EndLessGridWorldQuad/reg*'

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
# NOTE: remember to change this
num_state = 21 * 21
num_sample = np.ceil(num_state * possible_sample_ratio)


# l1_reward = np.concatenate(l1_reward, axis=1)
# mean_l1_reward = np.mean(l1_reward, axis=1)
#
# unc_reward = np.concatenate(unc_reward, axis=1)
# mean_unc_reward = np.mean(unc_reward, axis=1)

# plt.plot(mean_l1_reward)
# plt.plot(mean_unc_reward)
# plt.show()

l1_theta = list(zip(*l1_theta))
l1_theta = [sum(item, []) for item in l1_theta]
unc_theta = list(zip(*unc_theta))
unc_theta = [sum(item, []) for item in unc_theta]

l1_mean_std = []

for theta_list in l1_theta:
    ave_cov = []
    for theta in theta_list:
        theta /= np.linalg.norm(theta)
        ave_cov.append(np.std(theta))

    l1_mean_std.append(np.mean(ave_cov))


unc_mean_std = []

for theta_list in unc_theta:
    ave_cov = []
    for theta in theta_list:
        theta /= np.linalg.norm(theta)
        ave_cov.append(np.std(theta))

    unc_mean_std.append(np.mean(ave_cov))

print(l1_mean_std)
print(unc_mean_std)
#
# plt.plot(num_sample, l1_mean_std, '-*', label=r"$\ell_1$-regularized policy", linewidth=2)
# plt.plot(num_sample, unc_mean_std, label=r"unregularized policy", linewidth=2)
# plt.xlabel('Number of samples')
# plt.ylabel(r"Standard deviation of normalized $\hat \theta$.")
# plt.legend()
# plt.savefig('../../data/EndLessGridWorldQuad/sparsity.pdf')
# plt.show()

# l1_time = np.array(l1_time)
# l1_time = np.mean(l1_time, axis=0)
#
# unc_time = np.array(unc_time)
# unc_time = np.mean(unc_time, axis=0)

# plt.plot(num_sample, l1_time, '-*', label=r"$\ell_1$-regularized policy", linewidth=2)
# plt.plot(num_sample, unc_time, label=r"unregularized policy", linewidth=2)
# plt.xlabel('Number of samples')
# plt.ylabel("Computation time")
# plt.legend()
# # plt.savefig('../../data/EndLessGridWorldQuad/sparsity.pdf')
# plt.show()

# print(num_sample)
# print(l1_time)
# print(unc_time)
