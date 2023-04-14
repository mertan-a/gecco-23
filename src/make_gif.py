import gym
import numpy as np
import matplotlib.pyplot as plt
from pygifsicle import optimize
import imageio
import _pickle as pickle

from population import POPULATION
from evogym.envs import *
from evogym_wrappers import RenderWrapper, ActionSkipWrapper, ActionSpaceCorrectionWrapper, LocalObservationWrapper, GlobalObservationWrapper, LocalActionWrapper, GlobalActionWrapper

class MAKEGIF():

    def __init__(self, args):
        self.kwargs = vars(args)
        # read the ind from pkl
        with open(self.kwargs['path_to_ind'], 'rb') as f:
            # unpickle and check if it is a list
            unpickled = pickle.load(f)
            if isinstance(unpickled, POPULATION) or isinstance(unpickled, list):
                self.ind = unpickled[0]
            else:
                self.ind = unpickled

    def run(self):
        for b in self.ind.body.bodies:
            env = gym.make(self.kwargs['task'], body=b['structure'], connections=get_full_connectivity(b['structure']))
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = RenderWrapper(env, render_mode='img')
            if 'sparse_acting' in self.kwargs and self.kwargs['sparse_acting']:
                env = ActionSkipWrapper(env, skip=self.kwargs['act_every'])
            env = ActionSpaceCorrectionWrapper(env)
            if self.kwargs['controller'] in ['MODULAR']:
                env = LocalObservationWrapper(env, **self.kwargs)
                env = LocalActionWrapper(env, **self.kwargs)
            elif self.kwargs['controller'] in ['STANDARD']:
                env = GlobalObservationWrapper(env, **self.kwargs)
                env = GlobalActionWrapper(env, **self.kwargs)
            else:
                raise ValueError('Unknown controller', self.kwargs['controller'])
            env.seed(17)
            env.action_space.seed(17)
            env.observation_space.seed(17)

            # run the environment
            cum_reward = 0
            observation = env.reset()
            for ts in range(500):
                action = self.ind.brain.get_action(observation)
                observation, reward, done, _ = env.step(action)
                cum_reward += reward
                if type(done) == bool:
                    if done:
                        break
                elif type(done) == np.ndarray:
                    if done.all():
                        break
                else:
                    raise ValueError('Unknown type of done', type(d))
            imageio.mimsave(f"{self.kwargs['output_path']}_{cum_reward}_{b['name']}.gif", env.imgs, duration=(1/50.0))
            try:
                optimize(f"{self.kwargs['output_path']}_{cum_reward}_{b['name']}")
            except:
                pass

