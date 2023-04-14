import operator
import numpy as np

from individual import INDIVIDUAL
from body import FIXED_BODY, BASIC_BODY
from brain import MODULAR, STANDARD


class POPULATION(object):
    """A population of individuals"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.individuals = []
        self.non_dominated_size = 0

        while len(self) < self.args.nr_parents:
            self.add_individual()

    def add_individual(self):
        valid = False
        while not valid:
            # body
            if self.args.use_fixed_body:
                if (self.args.fixed_bodies is None or len(self.args.fixed_bodies) == 0) and self.args.fixed_body_path is None:
                    raise ValueError("No fixed bodies specified")
                elif self.args.fixed_bodies is not None and len(self.args.fixed_bodies) > 0:
                    body = FIXED_BODY(fixed_body=self.args.fixed_bodies)
                elif self.args.fixed_body_path is not None:
                    body = FIXED_BODY(fixed_body_path=self.args.fixed_body_path)
                else:
                    raise ValueError("something surely wrong")
            else:
                body = BASIC_BODY(self.args)
            # brain
            if self.args.controller == 'MODULAR':
                brain = MODULAR(args=self.args)
            elif self.args.controller == 'STANDARD':
                brain = STANDARD(args=self.args)
            else:
                raise ValueError("Unknown brain type", self.args.controller)
            ind = INDIVIDUAL(body=body, brain=brain)
            if ind.is_valid():
                self.individuals.append(ind)
                valid = True

    def produce_offsprings(self):
        """Produce offspring from the current population."""
        offspring = []
        for counter, ind in enumerate(self.individuals):
            #### temporary bug fix
            try:
                self.args.mc_ratio
            except:
                self.args.mc_ratio = [50,50]
            offspring.append(ind.produce_offspring(self.args.mc_ratio))
        self.individuals.extend(offspring)

    def calc_dominance(self):
        """Determine which other individuals in the population dominate each individual."""

        # if tied on all objectives, give preference to newer individual
        self.sort(key="age", reverse=False)

        # clear old calculations of dominance
        self.non_dominated_size = 0
        for ind in self:
            ind.dominated_by = []
            ind.pareto_level = 0

        for ind in self:
            for other_ind in self:
                if other_ind.self_id != ind.self_id:
                    if self.dominated_in_multiple_objectives(ind, other_ind) and (ind.self_id not in other_ind.dominated_by):
                        ind.dominated_by += [other_ind.self_id]

            ind.pareto_level = len(ind.dominated_by)  # update the pareto level

            # update the count of non_dominated individuals
            if ind.pareto_level == 0:
                self.non_dominated_size += 1

    def dominated_in_multiple_objectives(self, ind1, ind2):
        """Calculate if ind1 is dominated by ind2 according to all objectives in objective_dict.

        If ind2 is better or equal to ind1 in all objectives, and strictly better than ind1 in at least one objective.

        """
        wins = []  # 1 dominates 2
        if self.args.optimize_fitness:
            wins += [ind1.fitness > ind2.fitness]
        if self.args.optimize_age:
            wins += [ind1.age < ind2.age]
        return not np.any(wins)

    def sort_by_objectives(self):
        """Sorts the population multiple times by each objective, from least important to most important."""
        if self.args.optimize_age:
            self.sort(key="age", reverse=False)
        if self.args.optimize_fitness:
            self.sort(key="fitness", reverse=True)

        self.sort(key="pareto_level", reverse=False)  # min

    def update_ages(self):
        """Increment the age of each individual."""
        for ind in self:
            ind.age += 1

    def sort(self, key, reverse=False):
        """Sort individuals by their attributes.

        Parameters
        ----------
        key : str
            An individual-level attribute.

        reverse : bool
            True sorts from largest to smallest (useful for maximizing an objective).
            False sorts from smallest to largest (useful for minimizing an objective).

        """
        return self.individuals.sort(reverse=reverse, key=operator.attrgetter(key))

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        return iter(self.individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        try:
            return n in self.individuals
        except TypeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        return len(self.individuals)

    def __getitem__(self, n):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.individuals[n]

    def pop(self, index=None):
        """Remove and return item at index (default last)."""
        return self.individuals.pop(index)

    def append(self, individuals):
        """Append a list of new individuals to the end of the population.

        Parameters
        ----------
        individuals : list of/or INDIVIDUAL
            A list of individuals to append or a single INDIVIDUAL to append

        """
        if type(individuals) == list:
            for n in range(len(individuals)):
                if type(individuals[n]) != INDIVIDUAL:
                    raise TypeError("Non-INDIVIDUAL added to the population")
            self.individuals += individuals
        elif type(individuals) == INDIVIDUAL:
            self.individuals += [individuals]
        else:
            raise TypeError("Non-INDIVIDUAL added to the population")

