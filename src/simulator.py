import multiprocessing
import gym
import numpy as np

from evogym.envs import *
from evogym import get_full_connectivity

from evogym_wrappers import RenderWrapper, ActionSkipWrapper, ActionSpaceCorrectionWrapper, LocalObservationWrapper, GlobalObservationWrapper, LocalActionWrapper, GlobalActionWrapper, NormalizeObservationWrapper, RewardShapingWrapper

def make_env(body, **kwargs):
    env = gym.make(kwargs['task'], body=body, connections=get_full_connectivity(body))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if 'render' in kwargs:
        env = RenderWrapper(env, render_mode='screen')
    if 'sparse_acting' in kwargs and kwargs['sparse_acting']:
        env = ActionSkipWrapper(env, skip=kwargs['act_every'])
    env = ActionSpaceCorrectionWrapper(env)
    if kwargs['controller'] in ['MODULAR']:
        env = LocalObservationWrapper(env, **kwargs)
        env = LocalActionWrapper(env, **kwargs)
    elif kwargs['controller'] in ['STANDARD']:
        env = GlobalObservationWrapper(env, **kwargs)
        env = GlobalActionWrapper(env, **kwargs)
    else:
        raise ValueError('Unknown controller', kwargs['controller'])
    env.seed(17)
    env.action_space.seed(17)
    env.observation_space.seed(17)
    return env

def get_sim_pairs(population, **kwargs):
    sim_pairs = []
    for idx, ind in enumerate(population):
        for b in ind.body.bodies:
            sim_pairs.append( {'body':b['structure'], 'ind':ind, 'kwargs':kwargs, 'body_name':b['name']} )
    return sim_pairs

def simulate_ind(sim_pair):
    # unpack the simulation pair
    body = sim_pair['body']
    body_name = sim_pair['body_name']
    ind = sim_pair['ind']
    kwargs = sim_pair['kwargs']
    # check if the individual has fitness already assigned (e.g. from previous subprocess run. sometimes process hangs and does not return, all the population is re-submitted to the queue)
    if ind.fitness is not None:
        return ind, body_name, ind.fitness
    # get the env
    env = make_env(body, **kwargs)
    # record keeping
    cum_reward = 0
    # run simulation
    obs = env.reset()
    #print(obs.shape) # (n_obs,) for standard, (nr_active_voxels, n_obs) for modular
    for t in range(500):
        # collect actions
        actions = ind.brain.get_action(obs)
        # step
        obs, r, d, i = env.step(actions)
        # record keeping
        cum_reward += r
        # break if done
        if d:
            break
    return ind, body_name, cum_reward

def simulate_population(population, **kwargs):
    #get the simulator 
    sim_pairs = get_sim_pairs(population, **kwargs)
    # run the simulation
    finished = False
    while not finished:
        with multiprocessing.Pool(processes=len(sim_pairs)) as pool:
            results_f = pool.map_async(simulate_ind, sim_pairs)
            try:
                results = results_f.get(timeout=580)
                finished = True
            except multiprocessing.TimeoutError:
                print('TimeoutError')
                pass
    # assign fitness
    for r in results:
        ind, body_name, cum_reward = r
        for i in population:
            if i.self_id == ind.self_id:
                if i.detailed_fitness is None:
                    i.detailed_fitness = {}
                i.detailed_fitness[body_name] = cum_reward


