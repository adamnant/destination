# -*- coding: utf-8 -*-
"""
Stockyard model

@author: ag
"""

import numpy as np

class Stockyard(object):
    ntime = 0
    stocks = None  # the actual blocks - a list of stacks - one for each stockpile
    stocks_limits = None  # array - for each stockpile [[s1 l,s1 h,s1 m],... ]

    piles = None  # rehandle stockpiles
    stockpile_state = None  # stockpile build model - indicates whether stockyard is in build or destroy mode
    stockpile_capacity = 4

    npiles = 0  # number of rehandle stockpiles
    piles_n = None  # number of blocks on stockpiles at timestep n

    def __init__(self, low, high, num_piles, ntime, s_capacity=3):

        self.ntime = ntime
        self.piles = np.array(range(num_piles))
        self.npiles = len(self.piles)

        self.piles_n = np.zeros((self.npiles,
                                 self.ntime))  # variable to store number of blocks in each stockpile at each timestep in the sim
        self.piles_n[0:self.npiles, 0] = 0  # set starting stockpile inventory

        # list of empty stacks: empty stockpiles
        self.stocks = [[] for i in range(num_piles)]  # a = [[0] * number_cols for i in range(number_rows)]
        # self.example_thresholds()
        self.stockpile_state = [1 for i in range(num_piles)]  # 1 means pile is in build mode
        self.stockpile_capacity = s_capacity

    def set_thresholds_ea(self, thresholds):
        # print "set_thresholds_ea"
        # print thresholds
        # get an ndarray
        # self.npiles = NUM_PILES
        self.stocks_limits = np.zeros((self.npiles, 2))
        # zeroth pile is not used so just set to zero
        self.stocks_limits[0, 0] = 0.0
        self.stocks_limits[0, 1] = 0.0

        for i in range(1, self.npiles):
            self.stocks_limits[i, 0] = thresholds[i - 1, 0]
            self.stocks_limits[i, 1] = thresholds[i - 1, 1]
        self.stocks_limits = self.stocks_limits[self.stocks_limits[:, 0].argsort()]
        # print self.stocks_limits

    def set_example_thresholds(self):
        # self.npiles = NUM_PILES
        self.stocks_limits = np.zeros((self.npiles, 2))
        # pile zero is default catch all pile / dump - not used to reclaim
        self.stocks_limits = np.array([ \
            [0, 0], \
            [48, 51], \
            [51, 54], \
            [54, 57], \
            [57, 60], \
            [61, 63], \
            [64, 66]])
        #
        #                                       [45,51],\
        #                                       [51,54],\
        #                                       [54,57],\
        #                                       [57,60],\
        #                                       [61,63],\
        #                                       [64,66],\
        #                                       [45,51],\
        #                                       [51,54],\
        #                                       [54,57],\
        #                                       [57,60],\
        #                                       [61,63],\
        #                                       [64,66],\
        #                                       [45,51],\
        #                                       [51,54],\
        #                                       [54,57],\
        #                                       [57,60],\
        #                                       [61,63],\
        #                                       [64,66]] )

        # sort by the lower threshold:
        self.stocks_limits = self.stocks_limits[self.stocks_limits[:, 0].argsort()]
        # min = 55, max = 66
        # self.stocks_limits = np.array([60,61,62,63,65])

    def example_even_spaced_stockpile_thresholds(self, low, high, num_piles):
        # thresholds to accept material
        self.stocks_limits = np.zeros((num_piles, 2))

        step = float((high - low) / num_piles)

        s_low = float(low)
        s_high = float(s_low + step)

        for i in range(num_piles):
            self.stocks_limits[i] = [s_low, s_high]
            s_low += step
            s_high += step

    def update_trackers(self, tt):
        self.piles_n[:, tt] = self.piles_n[:, tt - 1]

    def add_block(self, block, tt):
        # get stockpile index
        # pile zero is default catch all pile / dump
        # can set to -1 if not using catch all pile and raise exception or define other behaviour

        assert len(self.stocks) == len(self.stocks_limits), "%s %s" % (len(self.stocks), len(self.stocks_limits))

        pile_index = 0
        for ind in range(1, len(self.stocks)):
            if (block >= self.stocks_limits[ind][0] and block < self.stocks_limits[ind][1]):
                if (self.stockpile_state[ind] > 0):  # is pile is in build state
                    pile_index = ind
                    break

                    #        if pile_index is -1:
                    #            pile_index = 0
                    #             raise Exception (("can't find stockpile for grade %s"), block)
                    #       else:

        self.stocks[pile_index].append(block)
        self.piles_n[pile_index, tt] = self.piles_n[pile_index, tt] + 1

        # update build state
        if (len(self.stocks[pile_index]) >= self.stockpile_capacity):
            self.stockpile_state[ind] = 0  # set pile to reclaim state

        return pile_index

    def available_stocks(self):
        av = []
        # pile zero is currently a catch all pile / dump - not used to reclaim just to track blocks that fall through thresholds as defined in stockpile list
        for i in range(1, len(self.stocks)):
            if len(self.stocks[i]) > 0 and self.stockpile_state[i] == 0:  # are stocks and pile is in reclaim mode
                av.append(i)
        return av

    def get_block(self, target_grade, tt):
        # which stockpiles have material?
        available = self.available_stocks()
        # if no stocks at all just return None
        if not available:
            return None, -1

        # Try to find a block with the lowest level above the target grade ie compare to min threshold
        # otherwise return the highest grade available to try to keep crushers running
        # also note available is assumed in order of lowest to highest grade stockpile

        for pile_ind in available:
            if self.stocks_limits[pile_ind][0] >= target_grade:
                block = self.stocks[pile_ind].pop()
                self.piles_n[pile_ind, tt] = self.piles_n[pile_ind, tt] - 1

                # update state to build if pile has become empty (build destroy logic)
                if len(self.stocks[pile_ind]) == 0:
                    self.stockpile_state[pile_ind] = 1

                return block, pile_ind

        # didn't find a block above the target quality so return highest grade available which is the highest index in available if stockpiles
        # are indexed in order of
        pile_ind = available[len(available) - 1]
        block = self.stocks[pile_ind].pop()
        self.piles_n[pile_ind, tt] = self.piles_n[pile_ind, tt] - 1

        # update state to build mode in case any pile has become empty ( build destroy logic)
        if len(self.stocks[pile_ind]) == 0:
            self.stockpile_state[pile_ind] = 1

        return block, pile_ind
