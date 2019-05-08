# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import random
import math
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey

class Learner(object):
	'''
	This agent jumps according to the laws of physics.
	'''

	def __init__(self):
		# Non-fixed
		self.bot_buff = 50
		self.top_buff = 50
		
		# Fixed
		self.impulse = 15
		self.horiz_speed = 25
		self.round = 1
		self.eta = 0.5
		self.epsilon = 0
		self.Qarr = np.zeros([2, 3, 2, 2], dtype=float)
		
		# In absence of information, don't jump
		#self.Qarr[:, :, :, 1] = -1
		
		# Don't fall off bottom
		#self.Qarr[:, 0, :, 0] = -1000
		
		# Don't jump off top
		#self.Qarr[:, 2, :, 1] = -1000
		
		# Don't jump if not needed
		#self.Qarr[:, :, 0, 1] = -100
		
		# Jump if needed
		#self.Qarr[:, :, 1, 1] = 100
		
		self.grav_list = []
		
		# Defaults
		self.last_state	 = None
		self.last_state_vec = None
		self.last_action = None
		self.last_reward = None
		self.grav_memory = None
		
		

	def reset(self):
		#print(self.last_state['score'], np.sum(np.square(self.Qarr)))
		self.round += 1
		
		self.grav_list.append(self.grav_memory)
		print(self.grav_list)
			
		self.last_state	 = None
		self.last_state_vec = None
		self.last_action = None
		self.last_reward = None
		self.grav_memory = None
			
	def y_at_tree_dist_if_jump (self, tree_dist, y_0, grav_high):
		y = y_0
		y += self.impulse * (tree_dist / self.horiz_speed)
		if grav_high == 1:
			y -= 0.5 * 4 * (tree_dist / self.horiz_speed) ** 2
		else:
			y -= 0.5 * 1 * (tree_dist / self.horiz_speed) ** 2
		return y
		
	def y_at_apex_of_jump (self, y_0, grav_high):
		if grav_high == 1:
			y = y_0 + self.impulse ** 2 / (2 * 4)
		else:
			y = y_0 + self.impulse ** 2 / (2 * 1)
		return y
	
	def will_clear_tree (self, tree_dist, y_0, grav_high):
		y_at_tree = self.y_at_tree_dist_if_jump(tree_dist, y_0, grav_high)
		if y_at_tree < 100 and y_0 < 50:
			return 1
		else:
			return 0
			
	def will_jump_off_top (self, y_0, grav_high):
		if self.y_at_apex_of_jump(y_0, grav_high) > self.top_buff:
			return 1
		else:
			return 0
			
	def monkey_bot_bucket(self, m_bot, grav_high):
		if m_bot < self.bot_buff:
			return 0
		elif self.will_jump_off_top (m_bot, grav_high):
			return 2
		else:
			return 1
			
	def create_state_vec(self, state):
		state_vec = []
		state_vec.append(self.grav_memory)
		state_vec.append(self.monkey_bot_bucket(state['monkey']['bot'], self.grav_memory))
		y_0 = state['monkey']['bot'] - state['tree']['bot']
		tree_dist = state['tree']['dist'] - 31
		if tree_dist < 0:
			state_vec.append(0)
		else:
			state_vec.append(self.will_clear_tree(tree_dist, y_0, self.grav_memory))
		return state_vec
		
	def action_callback(self, state):
		# Return False to swing and True to jump.
		
		#Do nothing on frame 1
		if self.last_state == None:
			self.last_state = state
			self.last_action = 0
			self.last_reward = 0
			return False
			
		if self.grav_memory == None:
			grav = self.last_state['monkey']['vel'] - state['monkey']['vel']
			if grav < 2:
				self.grav_memory = 0
			else:
				self.grav_memory = 1
			self.last_state_vec = self.create_state_vec(self.last_state)
			
		#print(self.Qarr)
				
		state_vec = self.create_state_vec(state)
		#print(state)
		#print(state_vec)
		#print(self.last_action)
		
		old_state_index = self.last_state_vec
		
		Qswing = self.Qarr[tuple(state_vec) + (0,)]
		Qjump = self.Qarr[tuple(state_vec) + (1,)]

		if Qswing != Qjump:
			best_action = np.argmax(np.array([Qswing, Qjump]))
		else:
			best_action = 0
		
		# Explore
		if npr.rand() < self.epsilon:
			new_action = random.randint(0, 1)
		else:
			new_action = best_action
		
		old_action_i = tuple(self.last_state_vec) + (self.last_action,)
		new_action_i = tuple(state_vec) + (new_action,)
		best_action_i = tuple(state_vec) + (best_action,)
		
		new_Q = (1-self.eta) * (self.Qarr[old_action_i])
		new_Q += self.eta * (self.last_reward + self.Qarr[best_action_i])
		
		# print(self.Qarr[old_action_i])
		self.Qarr[old_action_i] = new_Q
		# print(self.Qarr[old_action_i])

		self.last_action = new_action
		self.last_state_vec	 = state_vec
		self.last_state = state
		
	
		return (self.last_action > 0)

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''
		
		self.last_reward = reward
		
def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	for ii in range(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,				   # Don't play sounds.
							 text="Epoch %d" % (ii),	   # Display the epoch on screen.
							 tick_length = t_len,		   # Make game ticks super fast.
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
	run_games(agent, hist, 50, 10)

	print(hist)
	# Save history. 
	np.save('hist',np.array(hist))
	
	plt.plot(hist)
	plt.xlabel('epoch')
	plt.xticks([0, 5, 10, 15, 20,25,30,35,40,45,50])
	plt.ylabel('score')
	plt.show()


