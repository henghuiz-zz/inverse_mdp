import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io
from EnvSetting import num_state

filename = '../../data/EndLessGridWorldQuad/RegressResult2.mat'
LOADFILE = scipy.io.loadmat(filename)

PossibleSampleRatio = np.arange(0.1, 1.0, 0.05)
NumSample = np.ceil(num_state * PossibleSampleRatio)

VIReward = LOADFILE['VIReward']
Greedy = LOADFILE['Greedy']
ConReward = LOADFILE['ConReward']
UncReward = LOADFILE['UncReward']

ConReward = np.sort(ConReward, axis=1)

meanCon = np.mean(ConReward, axis=1)
maxCon = np.max(ConReward, axis=1)
minCon = np.min(ConReward, axis=1)

UncReward = np.sort(UncReward, axis=1)

meanUnc = np.mean(UncReward, axis=1)

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

plt.figure(figsize=(7, 6))

plt.plot(NumSample, meanCon, '-o', lw=2, color='blue', label='$\ell_1.$-regularized policy')
plt.plot(NumSample, meanUnc, '-o', lw=2, color='yellow', label='unregularized policy')
plt.plot([np.min(NumSample), np.max(NumSample)], [VIReward[0], VIReward[0]], lw=2, color='red', label='target policy')
plt.plot([np.min(NumSample), np.max(NumSample)], [Greedy[0], Greedy[0]], lw=2, color='green', label='greedy policy')
plt.axis([np.min(NumSample), np.max(NumSample), 0, 3])
plt.xlabel("Number of samples", fontsize=18)
plt.ylabel("Average reward", fontsize=18)
plt.fill_between(NumSample, ConReward[:, 90 - 1], ConReward[:, 10 - 1], facecolor='blue', alpha=0.3)
plt.fill_between(NumSample, UncReward[:, 90 - 1], UncReward[:, 10 - 1], facecolor='yellow', alpha=0.3)
plt.legend(loc='lower right', fontsize=18)
plt.savefig('../../data/EndLessGridWorldQuad/EndLessGridWorldQuadCmp.pdf')
# plt.savefig('../../data/EndLessGridWorldQuad/EndLessGridWorldQuadCmp.eps',rasterized=True)

plt.show()
