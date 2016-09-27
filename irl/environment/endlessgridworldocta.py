# gridworld.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random
from irl import MarkovDecisionProcess
import irl.environment as environment
import numpy as np
from scipy.stats import norm


def circleadd(x, y, mod=8):
    return np.mod(x+y, mod)


def gaussian(distance, peak, std_deviation):
    return peak * np.exp(-(distance * distance) / (2 * std_deviation * std_deviation)) / ( std_deviation * np.sqrt(2 * np.pi))


class EndlessGridworldOcta(MarkovDecisionProcess):
    """
      Gridworld
    """
    def __init__(self, reward):
        # layout
        self.grid = reward
        height, width = np.shape(reward)
        self.height = height
        self.width = width
        self.initState = (random.randint(0, self.width-1),
                          random.randint(0, self.height-1),
                          random.randint(0, 7))
        # parameters
        self.noise = 0.2
        self.sources = []
        self.numberActorFea = 0
        self.numberCriticFea = 0

    def setNoise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise

    def setSource(self, source):
        self.sources = source
        self.numberActorFea = len(source)
        self.numberCriticFea = len(source)

    def getPossibleActions(self, state):
        """
        Returns list of valid actions for 'state'.

        Note that you can request moves into walls and
        that "exit" states transition to the terminal
        state under the special action "done".
        """
        return ('left', 'right')

    def getStates(self):
        """
        Return list of all states.
        """
        # The true terminal state.
        states = []
        for x in range(self.width):
            for y in range(self.height):
                for j in range(8):
                    state = (x, y, j)
                    states.append(state)
        return states

    def getReward(self, state, action, nextState):
        """
        Get reward for state, action, nextState transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        x, y, heading = nextState
        cell = self.grid[x][y]
        return cell

    def getStartState(self):
        self.initState = (random.randint(0, self.width - 1),
                          random.randint(0, self.height - 1),
                          random.randint(0, 7))
        return self.initState

    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """

        if action not in self.getPossibleActions(state):
            raise RuntimeError("Illegal action!")

        x, y, heading = state

        successors = []

        NState = (self.__isAllowed(y, x - 1) and (x - 1, y)) or (x, y)
        NEState = (self.__isAllowed(y + 1, x - 1) and (x - 1, y + 1)) or (x, y)
        EState = (self.__isAllowed(y + 1, x) and (x, y + 1)) or (x, y)
        SEState = (self.__isAllowed(y + 1, x + 1) and (x + 1, y + 1)) or (x, y)
        SState = (self.__isAllowed(y, x + 1) and (x + 1, y)) or (x, y)
        SWState = (self.__isAllowed(y - 1, x + 1) and (x + 1, y - 1)) or (x, y)
        WState = (self.__isAllowed(y - 1, x) and (x, y - 1)) or (x, y)
        NWEState = (self.__isAllowed(y - 1, x - 1) and (x - 1, y - 1)) or (x, y)

        PossibleNextState = [NState, NEState,
                             EState, SEState,
                             SState, SWState,
                             WState, NWEState]

        actionbais = (action == 'left' and -1) or 1

        mainheading = circleadd(heading, actionbais)

        PossibleHeadings = [circleadd(mainheading, -1),
                            mainheading, circleadd(mainheading, 1)]

        for nextheading in PossibleHeadings:
            nextState = PossibleNextState[nextheading]
            nextState = nextState+(nextheading,)
            possibility = (nextheading == mainheading and 1-self.noise) or self.noise/2
            successors.append((nextState, possibility))

        return successors

    def __isAllowed(self, y, x):
        if y < 0 or y >= self.height:
            return False
        if x < 0 or x >= self.width:
            return False
        return True

    def findDistanceToOneSource(self, state, sourceid):
        dis = np.linalg.norm(np.array(state[0:2]) - self.sources[sourceid])
        return gaussian(dis, 10, 2.5)

    def findDistanceToAllSource(self, state):
        Dis = np.zeros(self.numberCriticFea)
        for i in range(len(self.sources)):
            Dis[i] = self.findDistanceToOneSource(state, i)
        return Dis

    def findAllFeature(self, state):
        Fea = self.findDistanceToAllSource(state)
        return Fea

    def getActorFeature(self, state, action):
        Fea = np.zeros(self.numberCriticFea)
        for nextState, Prob in self.getTransitionStatesAndProbs(state, action):
            if Prob is not None:
                subFea = self.findAllFeature(nextState)
                Fea += subFea * Prob
        # print (state,action,Fea['reward'])
        return Fea

    def getCriticFeature(self, state):
        return self.findAllFeature(state)

class EndlessGridworldEnvironment(environment.Environment):

    def __init__(self, gridWorld):
        self.gridWorld = gridWorld
        self.reset()

    def getCurrentState(self):
        return self.state

    def getPossibleActions(self, state):
        return self.gridWorld.getPossibleActions(state)

    def doAction(self, action):
        state = self.getCurrentState()
        (nextState, reward) = self.getRandomNextState(state, action)
        self.state = nextState
        return (nextState, reward)

    def getRandomNextState(self, state, action, randObj=None):
        rand = -1.0
        if randObj is None:
            rand = random.random()
        else:
            rand = randObj.random()
        reward_sum = 0.0
        successors = self.gridWorld.getTransitionStatesAndProbs(state, action)
        for nextState, prob in successors:
            reward_sum += prob
            if reward_sum > 1.0:
                raise RuntimeError('Total transition probability more than one; sample failure.')
            if rand < reward_sum:
                reward = self.gridWorld.getReward(state, action, nextState)
                return (nextState, reward)
        raise RuntimeError('Total transition probability less than one; sample failure.')

    def reset(self):
        self.state = self.gridWorld.getStartState()

if __name__ == '__main__':
    gridSize = (9, 9)

    source = np.array(np.mat('2,2;2,8;8,2;8,8'))
    source -= 1
    source_rewards = np.array([1, -1, -1, 1])

    rewards = np.zeros([gridSize[0], gridSize[1]])
    for i in range(len(source)):
        rewards[source[i][0]][source[i][1]] = source_rewards[i]

    mdp = EndlessGridworldOcta(rewards)
    mdp.setNoise(0.2)
    mdp.setSource(source)

    env = EndlessGridworldEnvironment(mdp)

    import irl.graphunit.gird_world_display as Display
    display = Display.GridWorldWindows(9,9,unitsize=50)
    display.grid = rewards
    display.speed=10

    imgs = []

    for i in range(10):
        env.doAction('right')
        print(env.getCurrentState())
        display.Update(list(env.getCurrentState()))
