# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:50:35 2017

@author: ag
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from configuration import config_obj
import simstats
import stockyard


class Simulator(object):
    
    # simulation variables
    
    ndiggers = 0 # number of diggers   
    nblocks = 0 # number of blocks per build
    ntime = 0 # number of time periods in simulation  
    low = 0 # low grade cutoff
    high = 0  # highest grade threshold


    stockyard = None # stockyard object 
    npiles = 0 # number of rehandle stockpiles
    
    digggers = None # block sequence for each digging unit  
    destinations = None # destinations of each block for each digging unit    
    
    crusher_rate = 0 # blocks per time unit for crusher

    build_av = None  # average quality in current build
    build_cnt = None # blocks in current build at a time step 
    build_bks = None # sequence of blocks sent to builds
    build_ind = None # is block in build from stock or dig unit
    nbuild_bks = 0 # number of blocks in all builds
    
    upper = None # absolute lowest grade of Fe for ore - cutoff grade - lower is treated as waste
    lower = None # the highest concentration of iron that is possible in the data 
    
    build_start = None
    
    sim_stats = None
    
    def __init__(self, time_periods=40, starting_inventory_n=0, num_stockpiles=int(str(config_obj.sim.num_stockpiles)), grade_target=55.0):
        """
        Destination problem simulation
        """
        
        self.sim_stats = simstats.SimStats(time_periods)
        
        self.ndiggers = 2 
        self.nblocks = 7
        self.ntime = time_periods  
        
        self.npiles = num_stockpiles
        
        self.low = 45      
        self.high = 60  

        self.starting_inventory_n = starting_inventory_n
        self.target = grade_target
        
        # digger model                
        # mining sequence per dig unit
        self.diggers = self._load_digger_block_seq(self.ntime)
        #print self.diggers
        
        self.destinations = self._set_random_dest(self.ntime)
        #print self.destinations
        
        # crusher model
        # blocks to crush per time period (one crusher)
        self.crusher_rate = 2 

    def reset(self):   
        """
        reset the simulation variables 
        """
        # stockpile model
        # initially use one stockpile for each integer grade in the range

        self.stockyard = stockyard.Stockyard(self.low, self.high, self.npiles,self.ntime)
        self.sim_stats = simstats.SimStats(self.ntime)

        # state variables
        self.build_av = np.zeros((1,self.ntime))
        self.build_cnt = np.zeros((1,self.ntime), dtype=np.int8)
        self.build_bks = np.zeros((1,self.ntime*self.ndiggers), dtype=np.int8)
        self.build_ind = np.zeros((1,self.ntime*self.ndiggers), dtype=np.int8) # % 1 if from build, 0 otherwise
            
        
        self.upper = np.zeros((1,self.ntime))
        self.lower = np.zeros((1,self.ntime))         
        self.build_start = np.zeros((1,1)) # time step when builds were started
        self.nbuild_bks=0 # blocks in builds

    def _load_digger_block_seq(self, ntime, filename = './data/dig_seq.csv'):
        data = np.genfromtxt (filename, delimiter=",")
        
        seq = np.concatenate((\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime))
        ))
        
        
                
        for ii in range(len(seq[0,:])):
            if (ii < len(data)):
                seq[0,ii]=data[ii]
            else:
                break
        
        return seq


    def _set_random_dest(self, ntime):
        assert self.diggers is not None
        block_destinations = np.random.randint(2, size=(self.ndiggers, ntime))
        
        return block_destinations
    
    def set_destinations(self, digger, array):
        #print "<><><> set destinations <><><>"        
        self.destinations[digger,:] = array
        

    def _get_digger_block_seq(self,ntime,values = None):


        return np.concatenate((\
                np.random.rand(1,ntime)*10 + 55,\
                np.ones((1,ntime))*55,\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime))\
                )) 
        
            
    

    def run(self):
        #self.reset()
        tt = 0
        build_n = 0 # blocks in a current build
        build_number = 0 # the index of the build
        
        while (tt < self.ntime):
            #print "time step: "+str (tt)
            
            if (tt > 0):
                self.build_av[:,tt] = self.build_av[:,tt-1]
                self.stockyard.update_trackers(tt)

            
            # for each digger
            for jj in range (0,self.ndiggers):
                
                block = self.diggers[jj,tt]
                
                #print "digger: "+str(jj) +"\t" + "block Q: "+ str(block) 
                
                # if not waste
                if(block >= self.low):
                    
                    # calculate lower and upper threshold for selecting a block for the build
                    if (build_n is 0 or build_n is self.nblocks):
                        self.upper[0,tt] = 100
                        self.lower[0,tt] = 0                
                    else: 
                        self.upper[0,tt] = self.nblocks*self.target - build_n*self.build_av[0,tt] - (self.nblocks - build_n-1)*self.low
                        self.lower[0,tt] = self.nblocks*self.target - build_n*self.build_av[0,tt] - (self.nblocks - build_n-1)*self.high           
                    #print "accept: "+str(self.lower[:,tt] ) +" \t"+str(self.upper[:,tt]) 
                    #print "dest to crusher: "+ str(self.destinations[jj,tt] ==0)
                    
                    
                    # Decide whether to send to a stockpile or to the crusher  
                    # comment out one of the methods (either the business logic thresholds or the bit string logic)
                    if self.destinations[jj,tt] == 0 :                                                                      
                    #if (block >= self.lower[:,tt] and block <= self.upper[:,tt]) : # send to crusher (build)                                  
                        #print "send to crusher"
                        #print "build_n: "+str(build_n)
                                
                        # check if a new build is to be started (build size = 20)
                        if (build_n is self.nblocks):
                            build_n = 0
                            self.build_start = np.append(self.build_start, tt)
                            build_number = build_number +1
        
                        build_n = build_n + 1
                        
                        # calculate the average quality value of the build - ie before a block is added
                        if (build_n is 1): # first block
                            self.build_av[0,tt] = block              
                        else: # blocks 2 - number of blocks (20)
                            current_build_av = self.build_av[0,tt]
                            self.build_av[0,tt] = ((build_n-1)*current_build_av + block)/build_n                
                    
                        self.nbuild_bks = self.nbuild_bks + 1
                        
                        self.build_bks[0,self.nbuild_bks] = block
                        self.build_ind[0,self.nbuild_bks] = 1 # TODO check this shouldn't it be the index of the build?
                        
                        self.build_cnt[0,tt] = self.build_cnt[0,tt] + 1
                        
                        # stats -  direct feed to crusher block sent from dig unit in pit to build  
                        self.sim_stats.add_row(tt, self.target, build_number, build_n, block, self.build_av[0,tt] ,0 ,jj)
                        
                        
        
                    else: # send to stockpile - ROM pad               
                        pile_index = self.stockyard.add_block(block, tt)
                        #print "***"
                        #print "stocks"
                        #print self.stockyard.stocks
                        #print (("Send block to stockpile: %s %s"),block, pile_index)
                        #print "updated stocks"
                        #print self.stockyard.stocks
                        #print "***"
                else:
                    # do nothing - waste
                    pass
                    #print "Send to waste dump"
                
            # Keep crusher feed running from stockpile if necessary - ie keep crusher fully utilised       
            for kk in range(0,self.crusher_rate - self.build_cnt[0, tt]):
                #print "reclaim"
                
                new_grade = (build_n+1) * self.target - build_n * self.build_av[0,tt]
                #print "new grade: "+str(new_grade)
                # take block from stockpile
                grade, pile_ind = self.stockyard.get_block(new_grade, tt)
                #print "available grade: " +str (grade)
                #print self.stockyard.stocks
                
                if (grade is not None):
                                            
                    pbuild = self.build_av[0,tt]
                    self.cbuild = ((build_n)*pbuild + grade)/(build_n+1)
                    
                    #if (abs(cbuild-target) < abs(pbuild-target))
                                                           
                    # check if we need to start a new build
                    if (build_n is self.nblocks):
                        build_n = 0
                        self.build_start = np.append(self.build_start, tt)
                        build_number = build_number +1
                                            
                    build_n = build_n + 1
                    # calculate the average quality value of the build - ie before a block is added
                    if (build_n is 1): # first block
                        self.build_av[0,tt] = grade              
                    else: # blocks 2 - number of blocks (20)
                        current_build_av = self.build_av[0,tt]
                        self.build_av[0,tt] = ((build_n-1)*current_build_av + grade)/build_n
                        
                    self.nbuild_bks = self.nbuild_bks + 1
                    self.build_bks[0,self.nbuild_bks] = grade
                        
                    self.build_cnt[0,tt] = self.build_cnt[0,tt] + 1
                    
                    self.sim_stats.add_row(tt, self.target, build_number, build_n, grade, self.build_av[0,tt] , 1,pile_ind)
                    
            tt = tt + 1
        self.build_start = np.append(self.build_start, self.ntime)
        return self.eval(build_number)
        

    def eval(self,k):

        serror = 0
        for ii in range(1,np.shape(self.build_start)[0]):
            serror += math.pow((self.build_av[0, int(self.build_start[ii])-1 ] - self.target ), 2)
        
        penalty =   self.stockyard.piles_n[0,self.ntime-1] - k
        
        serror += penalty
            
        return serror   
              
        
    
    def plot_summary(self):
        
        self.sim_stats.print_table(self.stockyard.piles_n,self.destinations)
        #print "plot_summary"
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        
        #print np.shape(self.build_start)[0]
        
        for ii in range(1,np.shape(self.build_start)[0]):
                                   
            indices = np.linspace(self.build_start[ii-1] + 1,self.build_start[ii]-1,self.nblocks/2)                                    
            c = np.cumsum(self.build_bks[:,(ii-1)*self.nblocks +1 : (ii*self.nblocks)+1]  )

            
            #print "***"
            #print indices
            #print (ii-1)*self.nblocks + 1
            #print (ii*self.nblocks)+1
            #print c

                        
            c = c[np.arange(1,self.nblocks,2)]
            print (c)
            #print np.arange(2,self.nblocks,2)
            ba = np.divide(c,np.arange(2,self.nblocks,2))
            
            #print np.arange(2,self.nblocks,2)
            #print ba
            
            #print "***"
            
            ax1.plot(indices,self.target*np.ones(np.shape(indices)),color = 'k')
            ax1.plot(indices, ba, color = 'k')
        
        ax2.plot(np.transpose(self.stockyard.piles_n))
        plt.legend(self.stockyard.piles, ncol=4, loc='lower right', 
                   bbox_to_anchor=[1.0, -0.5],borderaxespad=1)
        plt.subplots_adjust(left=0.07, bottom=0.15, right=0.96, top=0.96, wspace=0.17, hspace=0.17)
        fig1.savefig("results/output.png", bbox_inches="tight")
        plt.show()
    
    
    #np.savetxt("stockpiles.csv", piles_n, delimiter=",",fmt='%i')
    
def test():
    s = Simulator()
    s.reset()
    s.stockyard.set_example_thresholds()
    s.set_destinations(1,np.random.randint(2, size=(1, 40)) )
    s.run()
    s.plot_summary()
    return s
    
#import pickle
#def stuff():
#    with open('result2.pickle','wb') as handle:
#        pickle.dump(result[2], handle)
    
