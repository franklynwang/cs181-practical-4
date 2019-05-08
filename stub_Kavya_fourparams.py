# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.1
        new_state = state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


class QLearner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        temp = np.zeros((60, 40, 20, 2, 2))
        # temp.fill(50.0)
        self.Q = list(temp)
        self.frameNumber = 0
        self.learningRate = 0.9
        self.epsilon = 0.3
        self.discount = 0.99999999

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.frameNumber = 0
        self.learningRate = 0.9
        self.epsilon = 0.3

    def accessQ(self, state, action):
        return self.Q[state[0]][state[1]][state[2]][state[3]][int(action)]

    def updateQ(self, state, action, val):
        self.Q[state[0]][state[1]][state[2]][state[3]][int(action)] = val

    def setLR(self, lr):
        self.learningRate = lr

    def setQ(self, q):
        self.Q = list(q)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        self.frameNumber += 1
        #if(self.frameNumber < 5000):
        #    self.epsilon = 0.1
        #    self.learningRate = 0.9
        #if(self.frameNumber > 5000):
        #    self.epsilon = 0.01
        #    self.learningRate = 0.7

        self.epsilon = 0.0
        self.learningRate = 0.0

        if (self.frameNumber == 1):
            self.frameOneVelocity = state['monkey']['vel']
            self.last_action = 0

            index1 = int(state['tree']['dist'] // 10)
            #index2 = int((state['monkey']['vel'] + 50) // 5)
            index3 = int(state['monkey']['bot'] // 10)
            index4 = int(state['tree']['bot'] // 10)

            self.last_state = [index1, index3, index4, 0]
            self.last_reward = 0

            return bool(self.last_action)

        if (self.frameNumber == 2):
            currentVelocity = state['monkey']['vel']
            if (abs(self.frameOneVelocity - currentVelocity) > 2):
                self.gravity = 0
            else:
                self.gravity = 1

        index1 = int(state['tree']['dist'] // 10)
        #index2 = int((state['monkey']['vel'] + 50) // 5)
        index3 = int(state['monkey']['bot'] // 10)
        index4 = int(state['tree']['bot'] // 10)

        currentState = [index1, index3, index4, int(self.gravity)]
        print(self.last_state)
        print(self.last_action)
        actionVals = [self.last_reward + self.discount * self.accessQ(currentState, a) for a in range(0, 2)]

        #print("THE Q VALUE was: " + str(self.accessQ(self.last_state, self.last_action)))
        self.updateQ(self.last_state, self.last_action, (1 - self.learningRate) * self.accessQ(self.last_state,
                                                                                               self.last_action) + self.learningRate * max(
            actionVals))

        #print("THE Q VALUE HAS BEEN UPDATED TO: " + str(self.accessQ(self.last_state, self.last_action)))

        if (npr.random() < self.epsilon):
            new_action = npr.randint(0, 1)
        else:
            new_action = actionVals.index(max(actionVals))
        new_state = currentState

        self.last_action = new_action
        self.last_state = new_state
        #print(self.last_reward)

        return bool(self.last_action)

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

    def editQValues(self):
        for ind1 in range(len(self.Q)):
            for ind3 in range(len(self.Q[0])):
                for ind4 in range(len(self.Q[0][0])):
                    for ind5 in range(len(self.Q[0][0][0])):
                        for act in range(len(self.Q[0][0][0][0])):
                                if ((ind3 < 3) and (ind1 < 10)):
                                    self.updateQ([ind1, ind3, ind4, ind5], act,
                                                 min(0.5 * (ind3 - 5) * (ind1 - 10), 0))


def run_games(learner, hist, iters=100, t_len=100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    learner.setQ(np.load('Q_matrix_iteration_editedvalues_15000.npy'))
    #learner.editQValues()

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

        #if (ii % 5000 == 0):
        #    np.save("Q_matrix_iteration_editedvalues_2_" + str(ii), np.array(learner.Q))
        #    print("SAVING!")
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = QLearner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 5000, 1)

    # Save history.
    np.save('hist_testing_editedvalues', np.array(hist))