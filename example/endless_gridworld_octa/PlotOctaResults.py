import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io
#from EnvSetting import numState

filename = '../../data/EndLessGridWorldOcta/Regression.mat'
LOADFILE = scipy.io.loadmat(filename)

NumSample = range(50,501,50)

VIReward = LOADFILE['VIReward']
Greedy   = LOADFILE['GreedyReward']
ConReward = LOADFILE['RewardCon']
UncReward = LOADFILE['RewardUnc']

ConReward = np.sort(ConReward, axis=0)

meanCon = np.mean(ConReward,axis=0)
maxCon  = np.max(ConReward, axis=0)
minCon  = np.min(ConReward, axis=0)

UncReward = np.sort(UncReward, axis=0)

meanUnc = np.mean(UncReward,axis=0)

plt.plot(NumSample,meanCon,'-o',lw=2,color='blue',label='$\ell_1$-regularized policy')
plt.plot(NumSample,meanUnc,'-o',lw=2,color='yellow',label='unregularized policy')
plt.plot([np.min(NumSample), np.max(NumSample)],[VIReward[0],VIReward[0]],lw=2,color='red',label='target policy')
plt.plot([np.min(NumSample), np.max(NumSample)],[Greedy[0],Greedy[0]],lw=2,color='green',label='greedy policy')
plt.axis([np.min(NumSample), np.max(NumSample),2,3])
plt.xlabel("Number of samples",fontsize=18)
plt.ylabel("Average reward",fontsize=18)
plt.fill_between(NumSample, ConReward[90-1,:], ConReward[10-1,:], facecolor='blue', alpha=0.3)
plt.fill_between(NumSample, UncReward[90-1,:], UncReward[10-1,:], facecolor='yellow', alpha=0.3)
plt.legend(loc='lower right',fontsize=18)
plt.savefig('../../data/EndLessGridWorldOcta/EndLessGridWorldOctaCmp.pdf')
plt.savefig('../../data/EndLessGridWorldOcta/EndLessGridWorldOctaCmp.eps',rasterized=True)
plt.show()
