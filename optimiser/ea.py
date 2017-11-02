import utils
import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", utils.MyContainer, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("individual", utils.initIndividual, creator.Individual, size=16) # size is number of stockpile thresholds
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("clone", utils.clone)
toolbox.register("mutate", utils.mutation)
toolbox.register("mate", utils.crossover)
toolbox.register("select",tools.selTournament, tournsize=3)
toolbox.register("evaluate", utils.evaluate)


def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop


############################################
#
#test_ind = toolbox.individual()
#test_ind2 = toolbox.individual()
#print type(test_ind)
#print test_ind

#issubclass(type(test_ind), utils.MyContainer)

#test_ind.fitness.values = utils.evaluate(test_ind)


#toolbox.mutate(test_ind)
#toolbox.mate(test_ind,test_ind2)

#print "attr1 = %s"  % (test_ind.attr1)
#print test_ind.fitness.values
#print  test_ind

#...