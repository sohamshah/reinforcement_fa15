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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import time


class AsynchronousValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        i = 0
        while i < self.iterations:
            for state in states:
                if i >= self.iterations:
                    break
                if mdp.isTerminal(state):
                    i += 1
                    continue
                self.values[state] = self.getMaxQValue(state)
                i += 1
        # import pdb; pdb.set_trace()
            #state_index += 1

    def getMaxQValue(self, state):
        max_Q = None
        for a in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, a)
            if q > max_Q:
                max_Q = q
        return max_Q

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0
        next_states = self.mdp.getTransitionStatesAndProbs(state, action)
        for s_prime, prob_sPrime in next_states:
            qvalue += prob_sPrime*(self.mdp.getReward(state) + self.discount*self.values[s_prime])
        return qvalue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possible_actions = self.mdp.getPossibleActions(state)
        if len(possible_actions) == 0:
            return None
        best_action, best_action_sum = None, None
        for a in possible_actions:
            q = self.computeQValueFromValues(state, a)
            if q > best_action_sum:
                best_action_sum, best_action = q, a
        return best_action




    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in states:
            possible_actions = self.mdp.getPossibleActions(state)
            for a in possible_actions:
                predecessor_states = self.mdp.getTransitionStatesAndProbs(state, a)
                for predecessor, _ in predecessor_states:
                    if predecessor in predecessors:
                        predecessors[predecessor].add(state)
                    else:
                        predecessors[predecessor] = set([state])


        queue = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                diff = abs(self.getMaxQValue(state) - self.getValue(state))
                queue.push(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = self.getMaxQValue(s)
                for predecessor in predecessors[s]:
                    diff = abs(self.getMaxQValue(predecessor) - self.getValue(predecessor))
                    if diff > theta:
                        queue.update(predecessor, -diff)
