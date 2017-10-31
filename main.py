# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 23:28:27 2017

@author: ag
"""
#import os
#__path__=[os.path.dirname(os.path.abspath(__file__))]

import ea.destinations_ga as ga
#from ea import utils as utils

def main():
    res = ga.ea_runner()
#   utils.plot_sim_dest_individual(res[2][0])
    

if __name__ == "__main__":
    main()