# policyIterationAgents.py
# ------------------------
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
import numpy as np

from learningAgents import ValueEstimationAgent

class PolicyIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PolicyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs policy iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 20):
        """
          Your policy iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState) #WRONG WRONG WRONG WRONG
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        states = self.mdp.getStates()
        # initialize policy arbitrarily
        self.policy = {}
        for state in states:
            if self.mdp.isTerminal(state):
                self.policy[state] = None
            else:
                self.policy[state] = self.mdp.getPossibleActions(state)[0]
        # initialize policyValues dict
        self.policyValues = {}
        for state in states:
            self.policyValues[state] = 0

        for i in range(self.iterations):
            # step 1: call policy evaluation to get state values under policy, updating self.policyValues
            self.runPolicyEvaluation()
            # step 2: call policy improvement, which updates self.policy
            self.runPolicyImprovement()

    def runPolicyEvaluation(self):
        """ Run policy evaluation to get the state values under self.policy. Should update self.policyValues.
        Implement this by solving a linear system of equations using numpy. """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)
        toSolve = np.zeros((numStates, numStates))

        rightHandSide = np.zeros((numStates, 1))

        for i in range(numStates):

            reward = self.mdp.getReward(states[i])
            rightHandSide[i] = -reward

            currState = states[i]
            if not self.mdp.isTerminal(currState):
                action = self.policy[currState]
                next_states = self.mdp.getTransitionStatesAndProbs(currState, action)
                for j in range(len(next_states)):
                        next_state, prob = next_states[j]
                        stateNum = states.index(next_state)
                        if currState == next_state:
                            toSolve[i][stateNum] = (self.discount*(prob-1))
                        else:
                            toSolve[i][stateNum] = self.discount*prob
        print toSolve
        ans = np.linalg.solve(toSolve, reward)

        for i in range(numStates):
            self.policyValues[states[i]] = ans[i]



    def runPolicyImprovement(self):
        """ Run policy improvement using self.policyValues. Should update self.policy. """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for state in states:
            # if self.mdp.isTerminal(state):
            #     self.policy[state] = None
            max_Utility = None
            best_action = None
            for action in self.mdp.getPossibleActions(state):
                utility = self.computeQValueFromValues(state, action)
                if utility > max_Utility:
                    max_Utility, best_action = utility, action
            self.policy[state] = best_action

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.policyValues.
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0
        next_states = self.mdp.getTransitionStatesAndProbs(state, action)
        for s_prime, prob_sPrime in next_states:
            qvalue += prob_sPrime*(self.mdp.getReward(state) + self.discount*self.policyValues[s_prime])
        return qvalue

    def getValue(self, state):
        return self.policyValues[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]
