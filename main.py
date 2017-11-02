# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 23:28:27 2017

@author: ag
"""
#import os
#__path__=[os.path.dirname(os.path.abspath(__file__))]

import ea.destinations_ga as ga
import ea.utils as utils


def main():
    """
    Main class to execute a test and save result data in results folder, also a simple plot is shown for the 
    simulation of the best individual 

    """
    res = ga.ea_runner()
    utils.plot_sim_dest_individual(res[2][0])

if __name__ == "__main__":
    main()