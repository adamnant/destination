# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:27:48 2017

@author: ag

Defines and execute a simple evolutionary algorithm for optimising a single objective instance of 
destination problem using the DEAP library

Methods for initialisation, crossover and mutation are defined in utils.py.

Solutions are represented with a boolean matrix with a number of columns equal to the number of digging units and a number of 
rows equal to the number of time steps. The interpretation of each column is a series of ore blocks extracted
by digging equipment during a period of discrete time steps (each row). Values in the matrix specify whether 
to stockpile or process the ore at each time step, 0 is interpreted as to process, and a 1 as to stockpile.

"""
import random
import numpy
import ea.utils as utils
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=40)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", utils.evaluate_destinations)
toolbox.register("mate", utils.cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.025)
toolbox.register("select", tools.selTournament, tournsize=3)


def ea_runner(pop_size=100,
              crossover_prob=0.5,
              mutation_prob=0.2,
              num_gen=100,
              elite_archive_size=1,
              rnd_seed=64):
    """
    Run the EA with the parameters set saving statistics
    return population, statistics, best individual(s)
    """
    random.seed(rnd_seed)

    pop = toolbox.population(n=pop_size)

    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(elite_archive_size, similar=numpy.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # run
    algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_gen, stats=stats,
                        halloffame=hof, verbose=True)

    # return p,l,hof
    return pop, stats, hof








