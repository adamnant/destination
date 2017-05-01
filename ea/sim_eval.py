# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:50:35 2017

@author: ag
"""

import numpy as np 
import matplotlib.pyplot as plt


class stockpile_sim(object):     
    
    # simulation variables
    
    ndiggers = 0 # number of diggers   
    nblocks = 0 # number of blocks per build
    ntime = 0 # number of time periods in simulation  
    low = 0 # low grade cutoff
    high = 0  # highest grade threshold
    piles = None # rehandle stockpiles
    npiles = 0 # number of rehandle stockpiles
    piles_n = None # blocks on stockpiles at timestep n    
    digggers = None # block sequence for each digging unit  
    crusher_rate = 0 # blocks per time unit for crusher

    build_av = None
    build_cnt = None
    build_bks = None
    build_ind = None 
    nbuild_bks = 0 # number of blocks in all builds
    
    upper = None
    lower = None
    
    build_start = None
    
    def __init__(self, time_periods=100, starting_inventory_n=10, grade_target=58.0):
        """
        Destination problem simulation
        """               
        self.ndiggers = 5       
        self.nblocks = 20       
        self.ntime = time_periods   
         
        self.low = 55      
        self.high = 62
        
        # initially use one stockpile for each integer grade in the range
        self.piles = np.array(range (self.low, self.high))
        self.npiles = len (self.piles)
        # variable to store stockpile contents info at each timestep in the sim
        self.piles_n = np.zeros((self.npiles,time_periods))
        # set starting stockpile inventory
        self.piles_n[0:self.npiles,0] = starting_inventory_n
        
        # digger model                
        # mining sequence per dig unit
        self.diggers = self._get_digger_block_seq(time_periods)
        
        # crusher model
        # blocks to crush per time period (one crusher)
        self.crusher_rate = 2 
   
        # build model
        # product stockpile grade target %Fe
        self.target = grade_target
        
        self.build_av = np.zeros((1,time_periods))
        self.build_cnt = np.zeros((1,time_periods))
        
        self.build_bks = np.zeros((1,time_periods*self.ndiggers))  
        self.build_ind = np.zeros((1,time_periods*self.ndiggers)) # % 1 if from build, 0 otherwise
        
        
        self.upper = np.zeros((1,time_periods))
        self.lower = np.zeros((1,time_periods)) 
        
        self.build_start = np.zeros((1,1)) # time step when builds were started

        pass

    def _get_digger_block_seq(self,ntime):

        return np.concatenate((
                np.random.rand(1,ntime) * 6 + 53,
                np.random.rand(1,ntime) * 6 + 56,
                np.random.rand(1,ntime) * 6 + 55,
                np.zeros((1,ntime)),
                np.zeros((1,ntime))
                ))     
    

    def run(self):
        tt=0
        build_n = 0 # blocks in a current build
        
        
        
        while (tt < self.ntime):
            print "time step: "+str (tt)
            
            if (tt > 0):
                self.piles_n[:,tt] = self.piles_n[:,tt-1]
                self.build_av[:,tt] = self.build_av[:,tt-1]
            
            # for each digger
            for jj in range (0,self.ndiggers):
                
                block = self.diggers[jj,tt]
                
                print "digger: "+str(jj) +"\t" + "block Q: "+ str(block) 
                
                # if not waste
                if(block >= self.low):
                    
                    # calculate lower and upper threshold for selecting a block for the build
                    # that would allow the target to be met
                    if (build_n is 0 or build_n is self.nblocks):
                        self.upper[0,tt] = 100
                        self.lower[0,tt] = 0                
                    else: 
                        self.upper[0,tt] = self.nblocks*self.target - build_n*self.build_av[0,tt] - (self.nblocks - build_n-1)*self.piles[0]
                        self.lower[0,tt] = self.nblocks*self.target - build_n*self.build_av[0,tt] - (self.nblocks - build_n-1)*self.piles[self.npiles-1]            
                    print "accept: "+str(self.lower[:,tt] ) +" \t"+str(self.upper[:,tt]) 
                     
                    
                    # Decide whether to send to a stockpile or to the crusher            
                    if (block >= self.lower[:,tt] and block <= self.upper[:,tt]) : # send to crusher (build)                                  
                        print "send to crusher"
                        print "build_n: "+str(build_n)                
                                
                        # check if a new build is to be started (build size = 20)
                        if (build_n is self.nblocks):
                            build_n = 0
                            self.build_start = np.append(self.build_start, tt)
        
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
        
                    else: # send to stockpile - ROM pad                
                        pile_index, = np.where(self.piles == np.floor(block))
                        self.piles_n [pile_index[0],tt] = self.piles_n [pile_index[0],tt] + 1
                        print "Send to stockpile: "+str(pile_index[0])+"\t Quality: "+str(self.piles[pile_index[0]] )
                else:
                    print "Send to waste dump"
                
            # Keep crusher feed running from stockpile if necessary - ie keep crusher fully utilised       
            for kk in range(0,self.crusher_rate - self.build_cnt[:,tt]):
                print "reclaim"
                        
                new_grade = (build_n+1)*self.target - build_n*self.build_av[0,tt]
                print "new grade: "+str(new_grade)
                # which stockpiles have material?
                temp_i, = np.where(self.piles_n[:,tt] > 0)
                
                if (np.size(temp_i) > 0):
                    grade = np.min(np.abs(self.piles[temp_i] - new_grade))
                    igrade = np.argmin(np.abs(self.piles[temp_i] - new_grade))
                
                    grade = self.piles[temp_i[igrade]]
                    
                    pbuild = self.build_av[0,tt]
                    self.cbuild = ((build_n)*pbuild + grade)/(build_n+1)
                    
                    #if (abs(cbuild-target) < abs(pbuild-target))
        
                    #take the block from the stock pile     
                    self.piles_n[temp_i[igrade],tt] = self.piles_n[temp_i[igrade],tt] - 1
                               
                    # check if we need to start a new build
                    if (build_n is self.nblocks):
                        build_n = 0
                        self.build_start = np.append(self.build_start, tt)
                                            
                    build_n = build_n + 1
                    # calculate the average quality value of the build - ie before a block is added
                    # TODO this seems to set av too low
                    if (build_n is 1): # first block
                        self.build_av[0,tt] = grade              
                    else: # blocks 2 - number of blocks (20)
                        current_build_av = self.build_av[0,tt]
                        self.build_av[0,tt] = ((build_n-1)*current_build_av + grade)/build_n
                        
                    self.nbuild_bks = self.nbuild_bks + 1
                    self.build_bks[0,self.nbuild_bks] = grade
                        
                    self.build_cnt[0,tt] = self.build_cnt[0,tt] + 1
                        
            tt = tt + 1
        self.build_start = np.append(self.build_start, self.ntime)
        

    #t = np.arange(0.,100,1)
    #plt.plot(t,build_av[0,:],'ro')
    
    def plot_summary(self):
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        
        for ii in range(1,np.shape(self.build_start)[0]-2):
            indices = np.linspace(self.build_start[ii],self.build_start[ii+1]-1,10)
            c = np.cumsum(self.build_bks[:,(ii-1)*self.nblocks+1 : (ii*self.nblocks)+1]  )
            c = c[np.arange(1,20,2)]
            ba = np.divide(c,np.arange(2,self.nblocks+2,2))
            
            ax1.plot(indices,self.target*np.ones(np.shape(indices)),color = 'k')
            ax1.plot(indices, ba, color = 'k')
        
        ax2.plot(np.transpose(self.piles_n))
        plt.legend(self.piles, ncol=4, loc='lower right', 
                   bbox_to_anchor=[1.0, -0.5],borderaxespad=1)
        plt.subplots_adjust(left=0.04, bottom=0.15, right=0.96, top=0.96, wspace=0.17, hspace=0.17)
        fig1.savefig("output.png", bbox_inches="tight")
        plt.show()
    
    
    #np.savetxt("stockpiles.csv", piles_n, delimiter=",",fmt='%i')
    
