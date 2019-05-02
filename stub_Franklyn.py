# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''
    qtable = dict()
    #takes in [state[tree][dist], state[monkey][velocity], state[monkey][bot], state[tree][bot], action]
    #q table takes in a state and an action
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.alpha = 0.5;
        self.epsilon = 0.5;
        self.NO_MOVE = -1;
        self.SWING = 0;
        self.JUMP = 1;
        self.GRAVITY = -1; #value of gravity

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.GRAVITY = -1;
        
    def process_state(self, state):
        '''convert state into format of array'''
        res = []
        res.append(state["tree"]["dist"] // 10)
        res.append(state["monkey"]["velocity"])
        res.append(state["monkey"]["bot"] // 10)
        res.append(state["tree"]["bot"] // 10);

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        if self.GRAVITY == -1:
            if self.last_state != None:
                grav = abs(self.last_state['monkey']['vel'] - state['monkey']['vel'])
                if grav < 1.25:
                    if grav > 0.75:
                        self.GRAVITY = 0
                elif grav > 3.75:
                    if grav < 4.25:
                        self.GRAVITY = 1
                else:
                    assert(False)
            self.last_state = state
            self.last_action = False
            return False
        else:
            cur_state = self.process_state(state)
            cur_state.append(self.GRAVITY)
            state_swing = cur_state;
            state_swing.append(0)
            state_jump = cur_state;
            state_jump.append(1)
            val_swing = qtable[state_swing];
            val_jump = qtable[state_jump];
            new_action = self.NO_MOVE;
            if npr.rand() > epsilon:
                #be greedy
                if val_swing > val_jump:
                    new_action = self.SWING;
                else:
                    new_action = self.JUMP;
            else:
                new_action = npr.choice([self.SWING, self.JUMP]);
            new_state = state
            last_pair = self.last_state;
            last_pair.append(self.last_action);
            new_pair = self.new_state;
            new_pair.append(self.new_action);
            Q[last_pair] = Q[last_pair] + alpha * (self.last_reward + Q[new_pair] - Q[last_pair]);
            
            self.last_action = new_action
            self.last_state  = new_state
            #with some probability cut epsilon in half. 
            if npr.rand() < 0.02:
                epsilon /= 2.0;
            return bool(self.last_action)
        
    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 20, 10)

	# Save history.
	np.save('hist',np.array(hist))
