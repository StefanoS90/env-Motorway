import gym
import gym_Motorway
import numpy as np
from collections import namedtuple
import collections
import time
import math

import sys 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter, init
from torch.nn import functional as F

from tensorboardX import SummaryWriter

import atari_wrappers
from agent import DQNAgent
import utils

from gym.wrappers.monitor import Monitor

DQN_HYPERPARAMS = {
	'dueling': False,
	'noisy_net': False,
	'double_DQN': False,
	'n_multi_step': 2,
	'buffer_start_size': 100,
	'buffer_capacity': 15000,
	'epsilon_start': 1.0,
	'epsilon_decay': 10**5,
	'epsilon_final': 0.02,
	'learning_rate': 5e-5,
	'gamma': 0.99,
	'n_iter_update_target': 1000
}


BATCH_SIZE = 32
MAX_N_GAMES = 3000
TEST_FREQUENCY = 10
ENV_NAME = "Motorway-v0"
SAVE_VIDEO = True
DEVICE = 'cpu' # or 'cuda'
SUMMARY_WRITER = True

LOG_DIR = 'content/runs'
name = '_'.join([str(k)+'.'+str(v) for k,v in DQN_HYPERPARAMS.items()])
name = 'prv'

if __name__ == '__main__':

	# create the environment
	env = gym.make(ENV_NAME)
	if SAVE_VIDEO:
		# save the video of the games
		env = gym.wrappers.Monitor(env, "main-"+ENV_NAME, force=True)

	obs = env.reset()
	# TensorBoard
	writer = SummaryWriter(log_dir=LOG_DIR+'/'+name + str(time.time())) if SUMMARY_WRITER else None

	print('Hyperparams:', DQN_HYPERPARAMS)

	# create the agent
	agent = DQNAgent(env, device=DEVICE, summary_writer=writer, hyperparameters=DQN_HYPERPARAMS)
	
	
	#test net forward pass
#	import matplotlib.pyplot as plt
#	plt.imshow(obs)
#	as_unrolled = [obs]
#	as_np_array = np.array(as_unrolled)
#	as_np_array = np.transpose(as_np_array,(0,3,1,2))
#	as_tensor   = torch.tensor(as_np_array)
#	as_tensor = as_tensor.float()
#	q_values  = agent.cc.moving_nn(as_tensor)
#	action_test = agent.cc.get_max_action(obs)
	
	n_games = 0
	n_iter = 0

	# Play MAX_N_GAMES games
	while n_games < MAX_N_GAMES:
		
#		print("epsilon = ", agent.epsilon)
		# act greedly
		action = agent.act_eps_greedy(obs)

		# one step on the environment
		new_obs, reward, done, _ = env.step(action)

		# add the environment feedback to the agent
		agent.add_env_feedback(obs, action, new_obs, reward, done)

		# sample and optimize NB: the agent could wait to have enough memories
		agent.sample_and_optimize(BATCH_SIZE)
		
		print("Length of replay buffer", len(agent.replay_buffer))
		print("Minimum buffer length", agent.buffer_start_size)


		obs = new_obs
		if done:
			n_games += 1

			# print info about the agent and reset the stats
			agent.print_info()
			agent.reset_stats()

			#if n_games % TEST_FREQUENCY == 0:
			#	print('Test mean:', utils.test_game(env, agent, 1))

			obs = env.reset()

	writer.close()

#tensorboard --logdir content/runs --host localhost
