#!/usr/bin/env python
# coding: utf-8

# # Experiment 1
# 
# 
# **Overview**: 
# 
# Two virtual networks, one with IDS one with IPS. Hosts running a Google Search for benign and a SYN attack for malicious.
# 
# **Actions**: toggle VN
# 
# **State**: where (which VN) each host is + last N IDS alerts for each host
# 
# **Reward**: XNOR of current VN state and desired state (dumb)

# In[1]:


from bella.ciao import GemelEnv

import time
import os
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[2]:


EPSILON = 0.1
EXPLORATION_DECAY = 0.99
GAMMA = 0.99


# ## DQN Agent

# In[3]:


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class DQNAgent:

    def __init__(self, env, max_eps, period=10):
        self.env = env
        self.max_episodes = max_eps
        self.model = self._create_model()
        self.epsilon = EPSILON
        self.period = period   

    def _create_model(self):
        """
        Builds a neural net model to digest the state
        """
        model = Sequential()
        model.add(Dense(
            20,
            input_shape=self.env.observation_shape(),
            activation="relu"
        ))
        model.add(Dense(20, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model
    
    @staticmethod
    def _to_feature_vector(state):
        return np.concatenate((state[0], state[1].flatten()))

    def train(self):

        histories = []
        
        # train for max_eps episodes
        for episode in range(1, self.max_episodes + 1):

            printProgressBar(episode, self.max_episodes)

            # start at random position
            _, terminal, step = self.env.reset(), False, 0
            
            time.sleep(self.period)
            
            state = self.env.state()
                              
            # flatten state
            state = DQNAgent._to_feature_vector(state)
            
            history = []

            # iterate step-by-step
            while not terminal:
                
                step += 1
                
                # pick action based on policy
                action = self.policy(state)
                
                print()
                print(f"Taking action {action}")

                # run action and get reward
                _, reward, terminal = self.env.step(action)
                
                # instead of using the immediate next state, wait for it to simmer
                time.sleep(self.period)
                state_next_raw = self.env.state()
                              
                # flatten state
                state_next = DQNAgent._to_feature_vector(state_next_raw)
                
                print()
                print(f"Step {step} reward={reward} new_state={state_next_raw}")

                # # this makes sense in an episodic environement
                # # where a terminal state means "losing"
                # if terminal:
                #    reward *= -1

                # compute target Q
                q_target = ( reward + GAMMA * np.amax(self.model.predict([[state_next]])[0]) )                         if not terminal else reward

                # update model
                q_updated = self.model.predict([[state]])[0]
                q_updated[action] = q_target
                self.model.fit([[state]], [[q_updated]], verbose=0)

                # update current state
                state = state_next
                
                # update history
                history.append({
                    "time": step,
                    "action": action,
                    "reward": reward,
                    "state": state_next_raw[0].tolist()
                })

            histories.append(history)
            
            # apply exploration decay
            self.epsilon *= EXPLORATION_DECAY

        return histories

    def policy(self, state):
        if np.random.rand() < self.epsilon:
            print("PERFORMING RANDOM ACTION")
            return np.random.randint(self.env.action_space.n)
        else:
            expected_rewards = self.model.predict([[state]])[0]
            return np.argmax(expected_rewards)

    def test(self):

        state, done = self.env.reset(), False
        total_reward = 0

        while not done:
            exp_rew = self.model.predict([[state]])[0]
            action = np.argmax(exp_rew)
            new_state, reward, done = self.env.step(action)
            total_reward += reward
            self.env.render()
            time.sleep(0.05)
            state = new_state

        # self.env.close()
        print(f"Total reward: {total_reward}")


# ## Running DQN

# In[ ]:


env = GemelEnv(interval=10, max_steps=40)
env.reset()

agent = DQNAgent(env, max_eps=10, period=5)
history = agent.train()

history


# In[ ]:




