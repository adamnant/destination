# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:04:41 2017

@author: ag
"""
import random
import numpy as np

import sim_eval as sim

class MyContainer(np.ndarray):
    # This class does not require the fitness attribute
    # it will be  added later by the creator
    def __init__(self, attributes):
        pass

def initIndividual(ind_class, size):
    # ind_class will receive a class inheriting from MyContainer
    #ind = ind_class(random.random() for _ in range(size))

    ind = ind_class((size/2,2))
    for i in range(size/2):  
        ind[i,0] = random.randint(48,60)
        ind[i,1] = ind[i,0] + random.randint(2,6)         
    
    return ind

def mutation(ind):
    for i in range(ind.size / 2):
        if random.random() < 1.0 / (float(ind.size) / 2.0):
            ind[i,0] = random.randint(48,60)
            ind[i,1] = ind[i,0] + random.randint(2,6)
    return ind,



def crossover(ind1, ind2):
    size = len(ind1)
    
    # always choose an even point
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
        
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    
    return ind1, ind2
    
    
def evaluate(individual):
    
    simulation = sim.Stockpile_sim(num_stockpiles = len(individual) )
    
    simulation.reset()
    simulation.stockyard.set_thresholds_ea(individual)
    fit = simulation.run()
    
    return fit,

def plot_sim(individual):
    
    simulation = sim.Stockpile_sim(num_stockpiles = len(individual) )
    
    simulation.reset()
    simulation.stockyard.set_thresholds_ea(individual)
    simulation.run()
    simulation.plot_summary()
    return simulation


def clone(ind):
    copy = type(ind)((len(ind),2))
    for i in range(len(ind)):
        copy[i,0] = ind[i,0]
        copy[i,1] = ind[i,1]
    copy.fitness.values = ind.fitness.values
    return copy
    
def get_best(pop):
    fit = 101
    ii=0
    for ind in range(len(pop)):
        print pop[ind].fitness.values[0]        
        if fit < pop[ind].fitness.values[0]:
            print pop[ind].fitness.values[0]
            ii = ind
    return ii
    
       
    

# ...
