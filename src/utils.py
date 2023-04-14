import numpy as np
import gym
import os
import shutil
from copy import deepcopy
import inspect
import random
import re
import xml.etree.ElementTree as ET

# filesystem related
def prepare_rundir(args):

    # experiment related arguments
    if args.saving_path is None:
        run_dir = "experiments/"
    else:
        # if saving path starts with '/', then it is an absolute path
        if args.saving_path[0] == '/':
            args.saving_path = os.path.relpath(args.saving_path)
        run_dir = args.saving_path
        # make sure there is a trailing slash
        if not run_dir.endswith("/"):
            run_dir += "/"
    run_dir += "jt-" + args.job_type
    run_dir += "_t-" + args.task.replace('-', '')
    # ea algorithm
    run_dir += '_ea-' + args.evolutionary_algorithm
    if args.evolutionary_algorithm == 'pareto':
        run_dir += '_opt-'
        if args.optimize_fitness:
            run_dir += 'Fitness'
        if args.optimize_age:
            run_dir += 'Age'
    else:
        raise(ValueError('Unknown evolutionary algorithm'))
    # multifitness_type
    if args.use_fixed_body and len(args.fixed_bodies) > 1:
        run_dir += '_mft-' + args.multifitness_type
    # random ind number
    run_dir += "_nri-" + str(args.nr_random_individual)\
        + "_gen-" + str(args.nr_generations) + "_par-"\
        + str(args.nr_parents)
    # controller
    run_dir += "_controller-" + args.controller
    if args.controller == 'MODULAR':
        run_dir += '_or-' + str(args.observation_range)
    elif args.controller == 'STANDARD':
        ...
    else:
        raise(ValueError('Unknown controller'))
    if args.observe_structure:
        run_dir += '_ostr-' + 'True'
    else:
        run_dir += '_ostr-' + 'False'
    if args.observe_voxel_volume:
        run_dir += '_ovol-' + 'True'
    else:
        run_dir += '_ovol-' + 'False'
    if args.observe_voxel_vel:
        run_dir += '_ovel-' + 'True'
    else:
        run_dir += '_ovel-' + 'False'
    if args.observe_time:
        run_dir += '_ot-' + str(args.observe_time_interval)
    else:
        run_dir += '_ot-False'
    if args.sparse_acting:
        run_dir += '_sa-' + str(args.act_every)
    else:
        run_dir += '_sa-False'
    # softrobot
    run_dir += "_bb-" + str(args.bounding_box[0]) + str(args.bounding_box[1])
    if args.use_fixed_body:
        if args.fixed_bodies:
            run_dir += "_body-" + str(args.fixed_bodies).replace(',', '').replace(' ', '')\
                       .replace('\'', '').replace('[', '').replace(']', '')
        elif args.fixed_body_path:
            run_dir += "_body-" + args.fixed_body_path.split('/')[-1].split('.')[0]
        else:
            raise(ValueError('No fixed body specified'))
    if args.use_pretrained_brain:
        run_dir += "_pb-" + args.pretrained_brain.split('/')[-1].split('.')[0]
    # exp
    if len(args.repetition) > 1:
        raise(ValueError('Only one repetition is supported'))
    run_dir += "_rep-" + str(args.repetition[0])
    if args.id is not None:
        if len(args.id) > 1:
            raise(ValueError('Only one id can be specified'))
        run_dir += "_id-" + args.id[0]

    return run_dir

def get_immediate_subdirectories_of(directory):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all subdirectories
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

def get_files_in(directory, extension=None):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all files
    if extension is None:
        return [name for name in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, name))]
    else:
        return [name for name in os.listdir(directory)
                if (os.path.isfile(os.path.join(directory, name)) and
                    os.path.splitext(name)[1] == extension)]

def create_folder_structure(rundir):
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    if os.path.exists(os.path.join(rundir, 'pickled_population')):
        shutil.rmtree(os.path.join(rundir, 'pickled_population'))
    os.makedirs(os.path.join(rundir, 'pickled_population'))
    if os.path.exists(os.path.join(rundir, 'evolution_summary')):
        shutil.rmtree(os.path.join(rundir, 'evolution_summary'))
    os.makedirs(os.path.join(rundir, 'evolution_summary'))
    if os.path.exists(os.path.join(rundir, 'best_so_far')):
        shutil.rmtree(os.path.join(rundir, 'best_so_far'))
    os.makedirs(os.path.join(rundir, 'best_so_far'))
    if os.path.exists(os.path.join(rundir, 'to_record')):
        shutil.rmtree(os.path.join(rundir, 'to_record'))
    os.makedirs(os.path.join(rundir, 'to_record'))
    if os.path.exists(os.path.join(rundir, 'sub_simulations')):
        shutil.rmtree(os.path.join(rundir, 'sub_simulations'))
    os.makedirs(os.path.join(rundir, 'sub_simulations'))

# pareto front related

def natural_sort(l, reverse):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)

def dominates(ind1, ind2, attribute_name, maximize):
    """Returns True if ind1 dominates ind2 in a shared attribute."""
    if maximize:
        return getattr(ind1, attribute_name) > getattr(ind2, attribute_name)
    else:
        return getattr(ind1, attribute_name) < getattr(ind2, attribute_name)

# processing observations

def get_volumes_from_pos(pos, structure, bounding_box):
    '''
    returns a 2d matrix with volumes of each voxel, or a -1 if the voxel is not in the body
    '''
    structure_corners = get_structure_corners(pos, structure, bounding_box)
    volumes = np.ones( bounding_box ) * -1
    for idx, corners in enumerate(structure_corners):
        if corners == None:
            continue
        x, y = two_d_idx_of(idx, bounding_box)
        volumes[x,y] = polygon_area([x[0] for x in corners], [x[1] for x in corners])
    return volumes

def get_velocities_from_pos(pos, structure, bounding_box):
    '''
    returns a 2d matrix with velocity of each voxel, or a 0 if the voxel is not in the body
    '''
    structure_corners = get_structure_corners(pos, structure, bounding_box)
    velocities = [[0] * bounding_box[0] for _ in range(bounding_box[1])]
    for idx, corners in enumerate(structure_corners):
        if corners == None:
            continue
        x, y = two_d_idx_of(idx, bounding_box)
        velocities[x][y] = voxel_velocity([x[0] for x in corners], [x[1] for x in corners])
    return velocities

def get_structure_corners(observation, structure, bounding_box):
    ''' returns the corner coordinates of each voxel that exists in reading order (left to right, top to bottom)'''
    f_structure = structure.flatten()
    pointer_to_masses = 0
    structure_corners = [None] * bounding_box[0] * bounding_box[1]
    for idx, val in enumerate(f_structure):
        if val == 0:
            continue
        else:
            if pointer_to_masses == 0:
                structure_corners[idx] = [ [observation[0,0], observation[1,0]],
                                      [observation[0,1], observation[1,1]],
                                      [observation[0,2], observation[1,2]],
                                      [observation[0,3], observation[1,3]] ]
                pointer_to_masses += 4
            else:
                # check the 2d location and find out whether this voxel has a neighbor in to its left or up
                x, y = two_d_idx_of(idx, bounding_box)
                left_idx = one_d_idx_of(x, y-1, bounding_box)
                up_idx = one_d_idx_of(x-1, y, bounding_box)
                upright_idx = one_d_idx_of(x-1, y+1, bounding_box)
                upleft_idx = one_d_idx_of(x-1, y-1, bounding_box)
                if y-1 >= 0 and x-1 >= 0 and structure_corners[left_idx] is not None and structure_corners[up_idx] is not None:
                    # both neighbors are occupied, only the bottom right point mass is new
                    structure_corners[idx] = [ structure_corners[up_idx][2],
                                          structure_corners[up_idx][3],
                                          structure_corners[left_idx][3],
                                          [observation[0,pointer_to_masses], observation[1,pointer_to_masses]] ]
                    pointer_to_masses += 1
                elif y-1>=0 and structure_corners[left_idx] is not None and y+1<bounding_box[1] and x-1>=0 and structure_corners[upright_idx] is not None and structure[x,y+1] != 0:
                    # left and up right are occupied, bottom right point mass is new (connected to up right through right neighbor)
                    structure_corners[idx] = [ structure_corners[left_idx][1],
                                          structure_corners[upright_idx][2],
                                          structure_corners[left_idx][3],
                                          [observation[0,pointer_to_masses], observation[1,pointer_to_masses]] ]
                    pointer_to_masses += 1
                elif y-1>=0 and structure_corners[left_idx] is not None:
                    # only the left neighbor is occupied, top right and bottom right point masses are new
                    structure_corners[idx] = [ structure_corners[left_idx][1],
                                          [observation[0,pointer_to_masses], observation[1,pointer_to_masses]],
                                          structure_corners[left_idx][3],
                                          [observation[0,pointer_to_masses+1], observation[1,pointer_to_masses+1]] ]
                    pointer_to_masses += 2
                elif x-1>=0 and structure_corners[up_idx] is not None:
                    # only the up neighbor is occupied, bottom left and bottom right point masses are new
                    structure_corners[idx] = [ structure_corners[up_idx][2],
                                            structure_corners[up_idx][3],
                                            [observation[0,pointer_to_masses], observation[1,pointer_to_masses]],
                                            [observation[0,pointer_to_masses+1], observation[1,pointer_to_masses+1]] ]
                    pointer_to_masses += 2
                elif y+1<bounding_box[1] and x-1>=0 and structure_corners[upright_idx] is not None and structure[x,y+1] != 0:
                    # only the up right neighbor is occupied, top left, bottom left, and bottom right point masses are new (connected to upright through right neighbor)
                    structure_corners[idx] = [ [observation[0,pointer_to_masses], observation[1,pointer_to_masses]],
                                            structure_corners[upright_idx][2],
                                            [observation[0,pointer_to_masses+1], observation[1,pointer_to_masses+1]],
                                            [observation[0,pointer_to_masses+2], observation[1,pointer_to_masses+2]] ]
                    pointer_to_masses += 3
                else:
                    # no neighbors are occupied, all four point masses are new
                    structure_corners[idx] = [ [observation[0,pointer_to_masses], observation[1,pointer_to_masses]],
                                            [observation[0,pointer_to_masses+1], observation[1,pointer_to_masses+1]],
                                            [observation[0,pointer_to_masses+2], observation[1,pointer_to_masses+2]],
                                            [observation[0,pointer_to_masses+3], observation[1,pointer_to_masses+3]] ]
                    pointer_to_masses += 4

    return structure_corners

def polygon_area(x,y):
    ''' Calculates the area of an arbitrary polygon given its vertices in x and y (list) coordinates. assumes the order is wrong'''
    x[0], x[1] = x[1], x[0]
    y[0], y[1] = y[1], y[0]
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

def voxel_velocity(x,y):
    ''' Calculates the velocity of a voxel given its 4 corners' velocities'''
    return ( (x[0]+x[1]+x[2]+x[3])/4.0, (y[0]+y[1]+y[2]+y[3])/4.0 )

def two_d_idx_of(idx, bounding_box):
    '''
    returns 2d index of a 1d index
    '''
    return idx // bounding_box[0], idx % bounding_box[1]

def one_d_idx_of(x, y, bounding_box):
    '''
    returns 1d index of a 2d index
    '''
    return x * bounding_box[0] + y

def get_moore_neighbors(x, y, bounding_box, observation_range):
    '''
    returns the 8 neighbors of a voxel in the structure
    '''
    neighbors = []
    min = observation_range * -1
    max = observation_range + 1
    for i in range(min, max):
        for j in range(min, max):
            if x+i >= 0 and x+i < bounding_box[0] and y+j >= 0 and y+j < bounding_box[1]:
                neighbors.append( (1.0, [x+i, y+j]) )
            else:
                neighbors.append( (-1.0, None) )
    return neighbors

# file locking for concurrent access to the same file
# the solution is taken from https://stackoverflow.com/questions/489861/locking-a-file-in-python
import fcntl
def lock_file(f):
    if f.writable(): fcntl.lockf(f, fcntl.LOCK_EX)
def unlock_file(f):
    if f.writable(): fcntl.lockf(f, fcntl.LOCK_UN)
# Class for ensuring that all file operations are atomic, treat
# initialization like a standard call to 'open' that happens to be atomic.
###### This file opener *must* be used in a "with" block.
class ATOMICOPEN:
    # Open the file with arguments provided by user. Then acquire
    # a lock on that file object (WARNING: Advisory locking).
    def __init__(self, path, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        self.file = open(path,*args, **kwargs)
        # Lock the opened file
        lock_file(self.file)

    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs): return self.file

    # Unlock the file and close the file object.
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):        
        # Flush to make sure all buffered contents are written to file.
        self.file.flush()
        os.fsync(self.file.fileno())
        # Release the lock on the file.
        unlock_file(self.file)
        self.file.close()
        # Handle exceptions that may have come up during execution, by
        # default any exceptions are raised to the user.
        if (exc_type != None): return False
        else:                  return True   




###### gym environment wrappers
# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs), {}

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    The exponential moving average will have variance :math:`(1 - \gamma)^2`.
    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        dones = np.logical_or(terminateds, truncateds)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

