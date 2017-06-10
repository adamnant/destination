# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 23:28:27 2017

@author: ag
"""

import ea.destinations_ga as ea
import ea.utils as utils


def main():
    res = ea.ea_runner()
    utils.plot_sim_dest_individual(res[2][0])
    

if __name__ == "__main__":
    main()