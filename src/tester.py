import gym
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import time

from body import FIXED_BODY
from utils import ATOMICOPEN
from evogym import is_connected, get_full_connectivity

class TESTER():

    def __init__(self, args):
        self.args = args
        # read the ind from pkl
        with open(self.args.path_to_ind, 'rb') as f:
            self.ind = pickle.load(f)[0]

    def test(self):
        # check the arguments to decide which test to run
        if self.args.test_on_simple_morph == True:
            self.test_on_simple_morph()
        elif self.args.test_search_space == True:
            self.test_search_space()
        else:
            raise ValueError("No test specified")

    def test_on_simple_morph(self):
        # get the fixed bodies
        fixed_bodies = FIXED_BODY(fixed_body=['biped', 'triped', 'worm', 'block'])
        for b in fixed_bodies.bodies:
            # get the environment
            env = gym.make(self.args.task[0], body=b['structure'], connections=b['connections'])
            # run the environment
            cum_reward = self.run(env, b)
            print(f"cum_reward of {b['name']}: {cum_reward}")
        
    def run(self, env, b):
        # run the environment
        cum_reward = 0
        _ = env.reset()
        for ts in range(self.args.sim_timesteps):
            if self.args.sparse_acting == True:
                if ts % self.args.act_every == 0:
                    observation = self.observe(env, b)
                    action = self.ind.brain.get_action(observation)
            else:
                observation = self.observe(env, b)
                action = self.ind.brain.get_action(observation)
            _, reward, done, _ = env.step(action)
            cum_reward += reward
            if done:
                break
        return cum_reward

    def observe(self, env, b):
        '''
        returns the observation of the environment
        '''
        return { 'ind_structure': b['structure'],
                'ind_bb': self.args.bounding_box,
                'pm_relative_pos': env.get_relative_pos_obs('robot'), 
                'pm_absolute_pos': env.object_pos_at_time(env.get_time(), 'robot'),
                'orientation': env.get_ort_obs('robot'),
                'com_pos': env.get_pos_com_obs('robot'),
                'time': env.get_time(),
                'com_vel': env.get_vel_com_obs('robot'),
                'pm_vel': env.object_vel_at_time(env.get_time(), 'robot') }
