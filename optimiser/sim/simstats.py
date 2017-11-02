# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:50:35 2017

@author: ag
"""

import numpy as np

import os
from datetime import datetime




class SimStats:
    # builds
    # 1 time, 2 target, 3 build index, 4 num blocks in build,  5 block qual,
    # 6 csum block qual, 7 source (pit 0, stockpile 1)

    table = None

    def __init__(self, ntime):
        # self.table = np.zeros((ntime,8))
        self.table = []
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = "./results/%s build.csv" % (self.time)
        self.directory = os.path.dirname(self.file_path)

        try:
            os.stat(self.directory)
        except:
            os.mkdir(self.directory)

    def add_row(self, time, target, bld_ind, build_n, block_q, build_avg, block_src, digger_pile_index):
        self.table.append((time, target, bld_ind, build_n, block_q, build_avg, block_src, digger_pile_index))

    def print_table(self, piles_n, destinations):
        build = np.reshape(self.table, newshape=(len(self.table), 8))
        np.savetxt(self.file_path, build, delimiter=",", fmt='%.2f')

        stockpiles_file_path = "./results/%s stockpiles.csv" % (self.time)

        np.savetxt(stockpiles_file_path, piles_n.transpose(), delimiter=",", fmt='%d')

        destinations_file_path = "./results/%s destinations.csv" % (self.time)

        np.savetxt(destinations_file_path, destinations.transpose(), delimiter=",", fmt='%d')


