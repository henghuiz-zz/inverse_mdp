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

from __future__ import absolute_import


import random
import collections as util
import numpy as np


def gaussian(distance, peak, std_deviation):
    return peak * np.exp(-(distance * distance) / (2 * std_deviation * std_deviation)) /\
           (std_deviation * np.sqrt(2 * np.pi))


def find_two_point_similarity(point_one, point_two, peak=10, std=1.75):
    """
    Find the similarity between two points using Gaussian function

    :param point_one: Point 1. Should be 2-D
    :param point_two: Point 2. Should be 2-D
    :param peak: The peak of the Gaussian function
    :param std: The standard deviation of the Gaussian function

    :return: the similarity between two points
    """
    distance = np.linalg.norm(np.array(point_one) - np.array(point_two))
    return gaussian(distance, peak, std)


class EndlessGridWorld:
    """
      Gridworld
    """

    def __init__(self, reward, waypoint_policy, noise=0.2):
        """
        Make a initial reward world
        :param reward: Reward mapping for the states
        """

        # layout
        self.reward = reward
        grid_dims = reward.shape
        self.width = grid_dims[0]
        self.height = grid_dims[1]

        self.init_state = (random.randint(0, self.width - 1),
                           random.randint(0, self.height - 1))

        # parameters
        self.waypoint_policy = waypoint_policy
        self.noise = noise

    def find_policy_features(self, state):
        all_distances = [
            find_two_point_similarity(state, waypoint)
            for waypoint in self.waypoint_policy
        ]
        return np.array(all_distances)

    def get_actor_feature(self, state, action):
        Fea = np.zeros(len(self.waypoint_policy))
        for next_state, prob in self.get_transition_states_and_prob(state, action):
            if prob is not None:
                subFea = self.find_policy_features(next_state)
                Fea += subFea * prob
        return Fea

    def get_possible_actions(self, state):
        """
        Returns list of valid actions for 'state'.
        """
        return [0, 1, 2, 3]

    def get_states(self):
        """
        Return list of all states.
        """
        # The true terminal state.
        states = []
        for x in range(self.width):
            for y in range(self.height):
                state = (x, y)
                states.append(state)
        return states

    def get_reward(self, state, action, next_state):
        """
        Get reward for state, action, next_state transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        x, y = next_state
        cell = self.reward[x][y]
        return cell

    def get_start_state(self):
        return self.init_state

    def get_transition_states_and_prob(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """

        if action not in self.get_possible_actions(state):
            raise RuntimeError("Illegal action!")

        x, y = state

        successors = util.Counter()

        east_state = (self.__is_allowed(x, y + 1) and (x, y + 1)) or state
        north_state = (self.__is_allowed(x - 1, y) and (x - 1, y)) or state
        west_state = (self.__is_allowed(x, y - 1) and (x, y - 1)) or state
        south_state = (self.__is_allowed(x + 1, y) and (x + 1, y)) or state

        next_possible_state = [
            state, east_state, west_state, north_state, south_state]
        prob_distribution = []

        if action == 0:  # go north
            prob_distribution = [
                self.noise/4, self.noise/4, self.noise/4, 1- self.noise, self.noise/4]
        if action == 1:  # go east
            prob_distribution = [
                self.noise/4, 1-self.noise, self.noise/4, self.noise/4, self.noise/4]
        if action == 2:  # go west
            prob_distribution = [
                self.noise/4, self.noise/4, 1-self.noise, self.noise/4, self.noise/4]
        if action == 3:  # go south
            prob_distribution = [
                self.noise/4, self.noise/4, self.noise/4, self.noise/4, 1- self.noise]

        for next_state, prob in zip(next_possible_state, prob_distribution):
            successors[next_state] += prob

        # To list
        successors = [(k, v) for k, v in successors.items()]

        return successors

    def is_terminal(self, state):
        return False

    def __is_allowed(self, x, y):
        if y < 0 or y >= self.height:
            return False
        if x < 0 or x >= self.width:
            return False
        return True
