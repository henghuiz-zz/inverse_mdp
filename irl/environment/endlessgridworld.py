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
import collections as util
import numpy as np

def gaussian(distance, peak, std_deviation):
    return peak * np.exp(-(distance * distance) / (2 * std_deviation * std_deviation)) / (
    std_deviation * np.sqrt(2 * np.pi))

class EndlessGridworld(MarkovDecisionProcess):
    """
      Gridworld
    """
    def __init__(self, reward):
        # layout
        self.grid = makeEndlessGrid(reward)
        self.initState = (random.randint(0,self.grid.width-1),
                          random.randint(0, self.grid.height-1))

        # parameters
        self.noise = 0.2
        self.sources = []
        self.numberActorFea=0
        self.numberCriticFea=0

    def setNoise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise

    def setSource(self,source):
        self.sources = source
        self.numberActorFea = len(source)#+1
        self.numberCriticFea = len(source)#+1

    def findDistanceToOneSource(self,state,sourceid):
        dis=np.linalg.norm(np.array(state)-self.sources[sourceid])
        return gaussian(dis, 10, 1)

    def findDistanceToAllSource(self,state):
        Dis=np.zeros(self.numberCriticFea)
        for i in range(len(self.sources)):
            Dis[i] = self.findDistanceToOneSource(state,i)
        return Dis

    def findAllFeature(self,state):
        Fea=self.findDistanceToAllSource(state)
        return Fea

    def getActorFeature(self,state,action):
        Fea = np.zeros(self.numberCriticFea)
        for nextState,Prob in self.getTransitionStatesAndProbs(state,action):
            if Prob is not None:
                subFea = self.findAllFeature(nextState)
                Fea += subFea*Prob
        return Fea

    def getCriticFeature(self,state):
        return self.findAllFeature(state)

    def getPossibleActions(self, state):
        """
        Returns list of valid actions for 'state'.

        Note that you can request moves into walls and
        that "exit" states transition to the terminal
        state under the special action "done".
        """
        return ('north','west','south','east')

    def getStates(self):
        """
        Return list of all states.
        """
        # The true terminal state.
        states = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                state = (x,y)
                states.append(state)
        return states

    def getReward(self, state, action, nextState):
        """
        Get reward for state, action, nextState transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        x,y = nextState
        cell = self.grid[x][y]
        return cell

    def getStartState(self):
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

        x, y = state

        if type(self.grid[x][y]) == int or type(self.grid[x][y]) == float:
            termState = self.grid.terminalState
            return [(termState, 1.0)]

        successors = []

        eastState = (self.__isAllowed(y + 1, x) and (x, y + 1)) or state
        northState = (self.__isAllowed(y, x - 1) and (x - 1, y)) or state
        westState = (self.__isAllowed(y - 1, x) and (x, y - 1)) or state
        southState = (self.__isAllowed(y, x + 1) and (x + 1, y)) or state


        #northState = (self.__isAllowed(y+1,x) and (x,y+1)) or state
        #westState = (self.__isAllowed(y,x-1) and (x-1,y)) or state
        #southState = (self.__isAllowed(y-1,x) and (x,y-1)) or state
        #eastState = (self.__isAllowed(y,x+1) and (x+1,y)) or state

        if action == 'north' or action == 'south':
            if action == 'north':
                successors.append((northState,1-self.noise))
            else:
                successors.append((southState,1-self.noise))

            massLeft = self.noise
            successors.append((westState,massLeft/2.0))
            successors.append((eastState,massLeft/2.0))

        if action == 'west' or action == 'east':
            if action == 'west':
                successors.append((westState,1-self.noise))
            else:
                successors.append((eastState,1-self.noise))

            massLeft = self.noise
            successors.append((northState,massLeft/2.0))
            successors.append((southState,massLeft/2.0))

        successors = self.__aggregate(successors)

        return successors

    def __aggregate(self, statesAndProbs):
        counter = util.Counter()
        for state, prob in statesAndProbs:
            counter[state] += prob
        newStatesAndProbs = []
        for state, prob in counter.items():
            newStatesAndProbs.append((state, prob))
        return newStatesAndProbs

    def __isAllowed(self, y, x):
        if y < 0 or y >= self.grid.height: return False
        if x < 0 or x >= self.grid.width: return False
        return self.grid[x][y] != '#'


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
        sum = 0.0
        successors = self.gridWorld.getTransitionStatesAndProbs(state, action)
        for nextState, prob in successors:
            sum += prob
            if sum > 1.0:
                raise RuntimeError('Total transition probability more than one; sample failure.')
            if rand < sum:
                reward = self.gridWorld.getReward(state, action, nextState)
                return (nextState, reward)
        raise RuntimeError('Total transition probability less than one; sample failure.')

    def reset(self):
        self.state = self.gridWorld.getStartState()

class Grid:
    """
    A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented appropriately.
    """
    def __init__(self, width, height, initialValue=' '):
        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def _getLegacyText(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._getLegacyText())


def makeEndlessGrid(reward):
    width, height = reward.shape
    grid = Grid(width, height)
    for x in range(width):
        for y in range(height):
            grid[x][y] = reward[x][y]
    return grid

if __name__ == '__main__':
    import pygame
    import irl.graphunit.gird_world_display as Display
    display = Display.GridWorldWindows(3,3, unitsize=50, colored=False)
    reward = np.zeros((3,3))
    mdp = EndlessGridworld(reward)
    mdp.setNoise(0)

    env = EndlessGridworldEnvironment(mdp)
    env.state=(1,1)

    env.doAction('west')
    display.Update(list(env.state))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                display.Quit()
