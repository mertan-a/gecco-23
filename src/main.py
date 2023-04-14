import os
import time
from copy import deepcopy
import random
import numpy as np
import multiprocessing
import time

from utils import prepare_rundir
from population import POPULATION
from algorithms import PARETO_OPTIMIZATION
from make_gif import MAKEGIF
from tester import TESTER

import argparse
parser = argparse.ArgumentParser(description='run jobs')
# experiment related arguments
parser.add_argument('-sp', '--saving_path', help='path to save the experiment')
parser.add_argument('-r', '--repetition', type=int,
                    help='repetition number, dont specify this if you want it to be determined automatically', nargs='+')
parser.add_argument('-id', '--id', type=str,
                    help='id of the job, dont specify this if you want no specific id for the jobs', nargs='+')
parser.add_argument('-jt', '--job_type', type=str, default='optimizeBrain',
                    help='job type', choices=['optimizeBrain', 'makeGif', 'cooptimize', 'test'])
# evolutionary algorithm related arguments
parser.add_argument('--evolutionary_algorithm', '-ea', type=str,
                    choices=['pareto'], default='pareto', help='choose the evolutionary algorithm')
parser.add_argument('--optimize_fitness',
                    help='when using pareto optimization, use fitness as an optimization target', action='store_true')
parser.add_argument('--optimize_age',
                    help='when using pareto optimization, use age as an optimization target', action='store_true')
parser.add_argument('--multifitness_type', '-mft', type=str,
                    choices=['min', 'avg', 'max'], default='min', help='choose how to combine the fitnesses')
parser.add_argument('-nrp', '--nr_parents', type=int,
                    default=16, help='number of parents')
parser.add_argument('-nrg', '--nr_generations', type=int,
                    default=1000, help='number of generations')
parser.add_argument('--nr_random_individual', '-nri', type=int, default=1,
                    help='Number of random individuals to insert each generation')
parser.add_argument('--mc_ratio', '-mcr', nargs='+', type=int, 
                    default=[50,50], help='probability of mutation for morp:cont')
# softrobot related arguments
parser.add_argument('--use_fixed_body', '-ufbo',
                    help='use fixed body/ies', action='store_true')
parser.add_argument('--fixed_bodies', '-fbo', nargs='+',
                    help='specify the fixed body/ies', type=str,  choices=['biped', 'worm', 'triped', 'block', 'deneme'])
parser.add_argument('--fixed_body_path', '-fbp',
                    help='specify the path to the individual that contains the body you want', default=None)
parser.add_argument('--bounding_box', '-bb', nargs='+', type=int, default=(5, 5),
                    help='Bounding box dimensions (x,y)') 
parser.add_argument('--use_pretrained_brain', '-uptbr',
                    help='use pretrained brain', action='store_true')
parser.add_argument('--pretrained_brain', '-ptbr',
                    help='specify the path to the pretrained brain\'s pkl', default='None')
# controller
parser.add_argument('--controller', '-ctrl', help='specify the controller',
                    choices=['MODULAR', 'STANDARD'], default='MODULAR')
parser.add_argument('--observation_range', '-or', type=int,
                    default=1, help='observation range')
parser.add_argument('--observe_structure', '-ostr', action='store_true')
parser.add_argument('--observe_voxel_volume', '-ovol', action='store_true')
parser.add_argument('--observe_voxel_vel', '-ovel', action='store_true')
parser.add_argument('--observe_time', '-ot', action='store_true')
parser.add_argument('--observe_time_interval', '-oti', type=int, default=25)
parser.add_argument('--sparse_acting', '-sa', action='store_true')
parser.add_argument('--act_every', '-ae', type=int, default=4)
# task 
parser.add_argument('--task', '-t', help='specify the task',
                    choices=['Walker-v0'], default='Walker-v0')
# testing
parser.add_argument('--path_to_ind', '-pti', help='path to the indivuduals pkl file')
parser.add_argument('--test_on_simple_morph', '-tosm', action='store_true')
parser.add_argument('--search_space_path', '-ssp', help='path to the search space file')
parser.add_argument('--search_space_result_path', '-ssrp', help='path to the results for search space')
parser.add_argument('--output_path', '-op', help='path to the gif file')
parser.add_argument('--gif_every', '-ge', type=int, default=50)

args = parser.parse_args()

def run(args):


    if args.job_type == 'makeGif':
        giffer = MAKEGIF(args)
        giffer.run()
        exit()

    if args.job_type == 'test':
        tester = TESTER(args)
        tester.test()
        exit()

    # run the job directly
    if args.repetition is None:
        args.repetition = [1]
    rundir = prepare_rundir(args)
    args.rundir = rundir
    print('rundir', rundir)

    # if this experiment is currently running or has finished, we don't want to run it again
    if os.path.exists(args.rundir + '/RUNNING'):
        print('Experiment is already running')
        exit()
    if os.path.exists(args.rundir + '/FINISHED'):
        print('Experiment has already finished')
        exit()

    # Initializing the random number generator for reproducibility
    SEED = args.repetition[0]
    random.seed(SEED)
    np.random.seed(SEED)

    # create population
    population = POPULATION(args=args)
    solution = population[0]

    # Setting up the optimization algorithm and runnning
    if args.evolutionary_algorithm == 'pareto':
        pareto_optimization = PARETO_OPTIMIZATION(
            args=args, population=population)
        pareto_optimization.optimize()
    else:
        raise ValueError('unknown evolutionary algorithm')

    # delete running file in any case
    if os.path.isfile(args.rundir + '/RUNNING'):
        os.remove(args.rundir + '/RUNNING')

    # if the job is finished, create a finished file
    print('job finished successfully')
    # write a file to indicate that the job finished successfully
    with open(args.rundir + '/FINISHED', 'w') as f:
        pass

if __name__ == '__main__':

    # sanity checks
    if args.optimize_fitness == False and args.optimize_age == False and args.job_type not in ['makeGif', 'test'] and args.evolutionary_algorithm == 'pareto':
        raise ValueError('you need to specify at least one optimization target')
    if args.job_type == 'cooptimize' and args.use_fixed_body == True:
        raise ValueError('cooptimization is not supported for fixed bodies')
    if args.job_type == 'cooptimize' and args.evolutionary_algorithm != 'pareto':
        raise ValueError('cooptimization is only supported for pareto optimization')

    # run the job
    run(args)





