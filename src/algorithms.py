import os
import shutil
import random
import _pickle as pickle
import math
from copy import deepcopy
import numpy as np
import subprocess as sub
import cma
import time
import string
from datetime import datetime
from itertools import permutations

from utils import get_files_in, create_folder_structure, natural_sort, get_immediate_subdirectories_of
from simulator import simulate_population
from make_gif import MAKEGIF
from population import POPULATION


class EVOLUTIONARY_ALGORITHM():
    def __init__(self, args):

        self.args = args
        self.early_stopping = False
        self.best_fitness = None

    def initialize_optimization(self):
        '''controls whether we are continuing or starting from scratch 
        initialize necessary things
        '''
        from_scratch = False
        if os.path.exists(os.path.join(self.args.rundir, 'pickled_population')):
            pickles = get_files_in(os.path.join(
                self.args.rundir, 'pickled_population'))
            if len(pickles) == 0:
                from_scratch = True
        else:
            from_scratch = True

        if from_scratch:
            print('starting from scratch\n')
            create_folder_structure(self.args.rundir)
            self.starting_generation = 1
            self.current_generation = 0
            print('evaluating initial population\n')
            self.evaluate()
            self.pickle_population()
        else:
            print('continuing from previous run\n')
            pickles = natural_sort(pickles, reverse=False)
            path_to_pickle = os.path.join(
                self.args.rundir, 'pickled_population', pickles[-1])
            self.population = self.load_pickled_population(path=path_to_pickle)
            # extract the generation number from the pickle file name
            self.starting_generation = int(
                pickles[-1].split('_')[-1].split('.')[0]) + 1
        if self.best_fitness is None:
            self.best_fitness = self.population[0].fitness

    def optimize(self):

        self.initialize_optimization()
        # write a file to indicate that the job is running
        with open(self.args.rundir + '/RUNNING', 'w') as f:
            pass

        for gen in range(self.starting_generation, self.args.nr_generations):

            print('GENERATION: {}'.format(gen))
            self.do_one_generation(gen)
            self.record_keeping(gen)
            self.pickle_population()

            if gen % self.args.gif_every == 0 or gen == self.args.nr_generations - 1:
                self.args.path_to_ind = os.path.join(self.args.rundir, 'to_record/best.pkl')
                self.args.output_path = os.path.join(self.args.rundir, f'to_record/{gen}')
                t = MAKEGIF(self.args)
                t.run()

            if self.early_stopping:
                self.pickle_population()
                break

    def do_one_generation(self, gen):

        self.current_generation = gen
        print('PRODUCING OFFSPRINGS')
        self.produce_offsprings()
        print('EVALUATING POPULATION')
        self.evaluate()
        print('SELECTING NEW POPULATION')
        self.select()

    def produce_offsprings(self):
        '''produce offsprings from the current population
        '''
        print('population size: {}'.format(len(self.population)))
        self.population.produce_offsprings()
        print('offsprings produced')
        print('add {} random individuals'.format(self.args.nr_random_individual))
        for i in range(self.args.nr_random_individual):
            self.population.add_individual()
        print('new population size: {}\n'.format(len(self.population)))

    def evaluate(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    def pickle_population(self):
        '''pickle population for later use
        '''
        pickle_dir = os.path.join(self.args.rundir, 'pickled_population')

        pickle_file = os.path.join(pickle_dir, 'generation_{}.pkl'.format(
            self.current_generation))
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.population, f)

    def load_pickled_population(self, path):
        '''load the population from a pickle file
        '''
        with open(path, 'rb') as f:
            population = pickle.load(f)
        return population

    def record_keeping(self, gen):
        '''writes a summary and saves the best individual'''
        # keep a fitness over time txt
        with open(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_over_time.txt'), 'a') as f:
            f.write('{}\n'.format(self.population[0].fitness))
        # write the best individual
        best_ind = self.population[0]
        with open(os.path.join(self.args.rundir, 'to_record', 'best.pkl'), 'wb') as f:
            pickle.dump(best_ind, f)
        # check whether there is an improvement in the best fitness
        if self.best_fitness is None:
            self.best_fitness = best_ind.fitness
        else:
            if best_ind.fitness > self.best_fitness:
                if self.args.id is None or self.args.id[0] != 'reptileInner':
                    self.best_fitness = best_ind.fitness
                else:
                    if best_ind.fitness > self.best_fitness + 0.3: # TODO: magic number
                        self.best_fitness = best_ind.fitness
                        if self.args.reptile_early_stopping and not (self.args.id is None) and self.args.id[0] == 'reptileInner':
                            self.early_stopping = True
                            print('early stopping')


class PARETO_OPTIMIZATION(EVOLUTIONARY_ALGORITHM):
    def __init__(self, args, population):
        EVOLUTIONARY_ALGORITHM.__init__(self, args)
        self.population = population

    def evaluate(self):
        '''evaluate the current population
        '''
        # determine unevaluated individuals
        unevaluated = []
        for ind in self.population:
            if ind.fitness is None:
                unevaluated.append(ind)
        # evaluate the unevaluated individuals
        simulate_population(population=unevaluated, **vars(self.args))
        # update the fitness of the evaluated individuals
        for ind in unevaluated:
            if len(ind.body.bodies) > 1:
                if self.args.multifitness_type == 'min':
                    ind.fitness = min( ind.detailed_fitness.values() )
                elif self.args.multifitness_type == 'max':
                    ind.fitness = max( ind.detailed_fitness.values() )
                elif self.args.multifitness_type == 'avg':
                    ind.fitness = np.mean( list(ind.detailed_fitness.values()) )
                else:
                    raise ValueError('multifitness_type not recognized')
            else:
                ind.fitness = list(ind.detailed_fitness.values())[0]
        print('population evaluated\n')

    def select(self):
        '''select the individuals that will be the next generation
        '''
        self.population.calc_dominance()
        self.population.sort_by_objectives()
        # print the self_id, fitness, pareto_level, parent_id, and age of the individuals
        print('population before selection')
        for ind in self.population:
            print('self_id: {}, fitness: {}, pareto_level: {}, parent_id: {}, age: {}'.format(
                ind.self_id, ind.fitness, ind.pareto_level, ind.parent_id, ind.age))
        # choose population_size number of individuals based on pareto_level
        new_population = []
        done = False
        pareto_level = 0
        while not done:
            this_level = []
            size_left = self.population.args.nr_parents - len(new_population)
            for ind in self.population:
                if len(ind.dominated_by) == pareto_level:
                    this_level += [ind]

            # add best individuals to the new population.
            # add the best pareto levels first until it is not possible to fit them in the new_population
            if len(this_level) > 0:
                # if whole pareto level can fit, add it
                if size_left >= len(this_level):
                    new_population += this_level
                else:  # otherwise, select by sorted ranking within the level
                    new_population += [this_level[0]]
                    while len(new_population) < self.population.args.nr_parents:
                        random_num = random.random()
                        log_level_length = math.log(len(this_level))
                        for i in range(1, len(this_level)):
                            if math.log(i) / log_level_length <= random_num < math.log(i + 1) / log_level_length and \
                                    this_level[i] not in new_population:
                                new_population += [this_level[i]]
                                continue

            pareto_level += 1
            if len(new_population) == self.population.args.nr_parents:
                done = True
        self.population.individuals = new_population
        self.population.update_ages()
        # print the self_id, fitness, parent_id, and age of the individuals
        print('population after selection')
        for ind in self.population:
            print('self_id: {}, fitness: {}, parent_id: {}, age: {}'.format(
                ind.self_id, ind.fitness, ind.parent_id, ind.age))


