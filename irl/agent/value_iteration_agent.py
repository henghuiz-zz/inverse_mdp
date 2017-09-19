# valueIterationAgents.py
# -----------------------
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

from __future__ import print_function
import collections as util


class ValueIterationAgent:
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Value iteration
        for i in range(iterations):
            value_instance = util.Counter()
            for state in self.mdp.get_states():
                v = []
                for action in self.mdp.get_possible_actions(state):
                    v.append(self.compute_q_value_from_values(state, action))
                if v:
                    value_instance[state] = max(v)
            self.values = value_instance
            if iterations > 100:
                print("\r Iteration:", i, end='', flush=True)

    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        "Returns the policy at the state (no exploration)."
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)

    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        V = 0
        for nextState, next_prob in self.mdp.get_transition_states_and_prob(state, action):
            V += next_prob * (self.mdp.get_reward(state, action, nextState) + self.discount * self.get_value(nextState))
        return V

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        all_value = util.Counter()
        for action in self.mdp.get_possible_actions(state):
            all_value[action] = self.compute_q_value_from_values(state, action)
        return all_value.most_common(1)[0][0]
