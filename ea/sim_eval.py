# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:50:35 2017

@author: ag
"""

import numpy as np 
import math
import matplotlib.pyplot as plt



NUM_PILES = 7

class Stockyard(object):
    ntime = 0    
    
    stocks = None # the actual blocks - a list of stacks - one for each stockpile
    stocks_limits = None # array - for each stockpile [[s1 l,s1 h,s1 m],... ]
    

    piles = None # rehandle stockpiles
    stockpile_state = None # stockpile build model - indicates whether stockyard is in build or destroy mode  
    stockpile_capacity = 4
    
    npiles = 0 # number of rehandle stockpiles
    piles_n = None # number of blocks on stockpiles at timestep n     
    
    def __init__(self,low, high, num_piles,ntime, s_capacity=4):
        
        self.ntime = ntime
        self.piles = np.array(range (num_piles))
        self.npiles = len (self.piles)
        
        self.piles_n = np.zeros((self.npiles,self.ntime)) # variable to store number of blocks in each stockpile at each timestep in the sim
        self.piles_n[0:self.npiles,0] = 0 # set starting stockpile inventory   
        
        # list of empty stacks: empty stockpiles
        self.stocks = [[] for i in range(num_piles)] #a = [[0] * number_cols for i in range(number_rows)]
        #self.example_thresholds()
        self.stockpile_state = [1 for i in range(num_piles)] # 1 means pile is in build mode
        self.stockpile_capacity = s_capacity
        
        

    def set_thresholds_ea(self, thresholds):
        print "set_thresholds_ea" 
        print thresholds
        # get an ndarray
        #self.npiles = NUM_PILES
        self.stocks_limits = np.zeros((self.npiles,2))
        # zeroth pile is not used so just set to zero
        self.stocks_limits[0,0] = 0.0
        self.stocks_limits[0,1] = 0.0
        
        for i in range(1,self.npiles):
            self.stocks_limits[i,0] = thresholds[i-1,0]
            self.stocks_limits[i,1] = thresholds[i-1,1]
        self.stocks_limits = self.stocks_limits[self.stocks_limits[:,0].argsort()] 
        print self.stocks_limits
        



    def set_example_thresholds(self):
        #self.npiles = NUM_PILES
        self.stocks_limits = np.zeros((self.npiles,2))
        # pile zero is default catch all pile / dump - not used to reclaim
        self.stocks_limits = np.array([\
                                       [0,0],\
                                       [48,51],\
                                       [51,54],\
                                       [54,57],\
                                       [57,60],\
                                       [61,63],\
                                       [64,66]])
        # sort by the lower threshold:
        self.stocks_limits = self.stocks_limits[self.stocks_limits[:,0].argsort()]                              
        # min = 55, max = 66
        # self.stocks_limits = np.array([60,61,62,63,65]) 
            
    def example_even_spaced_stockpile_thresholds(self, low, high, num_piles):
        # thresholds to accept material
        self.stocks_limits = np.zeros((num_piles,2))  
        
        step = float((high - low)/num_piles)
             
        s_low = float(low)
        s_high = float(s_low + step)
        
        for i in range(num_piles):
            self.stocks_limits[i] = [s_low,s_high]
            s_low += step
            s_high += step
        

    def update_trackers(self, tt):
        self.piles_n[:,tt] = self.piles_n[:,tt-1]
        
        
    def add_block(self, block, tt):
        # get stockpile index
        # pile zero is default catch all pile / dump 
        # can set to -1 if not using catch all pile and raise exception or define other behaviour

        assert len(self.stocks) == len(self.stocks_limits), "%s %s" % (len(self.stocks) ,len(self.stocks_limits))
        
        pile_index = 0
        for ind in range(1,len(self.stocks)):
            if ( block >= self.stocks_limits[ind][0] and block < self.stocks_limits[ind][1] ) :
                if (self.stockpile_state[ind] > 0): # is pile is in build state
                    pile_index = ind
                    break

#        if pile_index is -1:
#            pile_index = 0            
#             raise Exception (("can't find stockpile for grade %s"), block) 
 #       else:

        self.stocks[pile_index].append(block)
        self.piles_n [pile_index,tt] = self.piles_n [pile_index,tt] + 1
        
        # update build state
        if ( len(self.stocks[pile_index]) >= self.stockpile_capacity ):
            self.stockpile_state[ind] = 0 # set pile to reclaim state
        
        return pile_index

    def available_stocks(self):
        av = []
        # pile zero is currently a catch all pile / dump - not used to reclaim just to track blocks that fall through thresholds as defined in stockpile list
        for i in range(1,len(self.stocks)) :
            if len(self.stocks[i]) > 0 and self.stockpile_state[i] == 0: # are stocks and pile is in reclaim mode
                av.append(i)
        return av
            
    def get_block(self, target_grade, tt):
        # which stockpiles have material?
        available = self.available_stocks()
        # if no stocks at all just return None
        if not available:
            return None
              
        # Try to find a block with the lowest level above the target grade ie compare to min threshold 
        # otherwise return the highest grade available to try to keep crushers running
        # also note available is assumed in order of lowest to highest grade stockpile
              
        for pile_ind in available :
            if self.stocks_limits[pile_ind][0] >= target_grade:
                block = self.stocks[pile_ind].pop() 
                self.piles_n [pile_ind,tt] = self.piles_n [pile_ind,tt] - 1
   
                # update state to build if pile has become empty (build destroy logic)
                if len(self.stocks[pile_ind]) == 0:
                    self.stockpile_state[pile_ind] = 1
                                            
                
                return block 
        
        # didn't find a block above the target quality so return highest grade available which is the highest index in available if stockpiles
        # are indexed in order of 
        pile_ind = available[len(available)-1]
        block = self.stocks[pile_ind].pop() 
        self.piles_n [pile_ind,tt] = self.piles_n [pile_ind,tt] - 1

        # update state to build mode in case any pile has become empty ( build destroy logic)        
        if len(self.stocks[pile_ind]) == 0:
            self.stockpile_state[pile_ind] = 1
        
        return block                 
                


class Stockpile_sim(object):     
    
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

    build_av = None
    build_cnt = None
    build_bks = None
    build_ind = None 
    nbuild_bks = 0 # number of blocks in all builds
    
    upper = None # absolute lowest grade of Fe for ore - cutoff grade - lower is treated as waste
    lower = None # the highest concentration of iron that is possible in the data 
    
    build_start = None
    
    def __init__(self, time_periods=40, starting_inventory_n=0, num_stockpiles=NUM_PILES, grade_target=55.0):
        """
        Destination problem simulation
        """               
        self.ndiggers = 2 
        self.nblocks = 8 
        self.ntime = time_periods  
        
        self.npiles = num_stockpiles
        
        self.low = 45      
        self.high = 60  

        self.starting_inventory_n = starting_inventory_n
        self.target = grade_target
        
        # digger model                
        # mining sequence per dig unit
        self.diggers = self._load_digger_block_seq(self.ntime)
        print self.diggers
        
        self.destinations = self._set_random_dest(self.ntime)
        print self.destinations
        
        # crusher model
        # blocks to crush per time period (one crusher)
        self.crusher_rate = 2 

    def reset(self):   
        """
        reset the simulation variables 
        """
        # stockpile model
        # initially use one stockpile for each integer grade in the range


        self.stockyard = Stockyard(self.low, self.high, self.npiles,self.ntime)

           
        # state variables
    
        self.build_av = np.zeros((1,self.ntime))
        self.build_cnt = np.zeros((1,self.ntime))        
        self.build_bks = np.zeros((1,self.ntime*self.ndiggers))  
        self.build_ind = np.zeros((1,self.ntime*self.ndiggers)) # % 1 if from build, 0 otherwise
        self.upper = np.zeros((1,self.ntime))
        self.lower = np.zeros((1,self.ntime))         
        self.build_start = np.zeros((1,1)) # time step when builds were started
        self.nbuild_bks=0 # blocks in builds
        

    
    def _load_digger_block_seq(self, ntime, filename = 'dig_seq.csv'):

        
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
        print "<><><> set destinations <><><>"        
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
        tt=0
        build_n = 0 # blocks in a current build
        
        while (tt < self.ntime):
            print "time step: "+str (tt)
            
            if (tt > 0):
                self.build_av[:,tt] = self.build_av[:,tt-1]
                self.stockyard.update_trackers(tt)

            
            # for each digger
            for jj in range (0,self.ndiggers):
                
                block = self.diggers[jj,tt]
                
                print "digger: "+str(jj) +"\t" + "block Q: "+ str(block) 
                
                # if not waste
                if(block >= self.low):
                    
                    # calculate lower and upper threshold for selecting a block for the build
                    if (build_n is 0 or build_n is self.nblocks):
                        self.upper[0,tt] = 100
                        self.lower[0,tt] = 0                
                    else: 
                        self.upper[0,tt] = self.nblocks*self.target - build_n*self.build_av[0,tt] - (self.nblocks - build_n-1)*self.low
                        self.lower[0,tt] = self.nblocks*self.target - build_n*self.build_av[0,tt] - (self.nblocks - build_n-1)*self.high           
                    print "accept: "+str(self.lower[:,tt] ) +" \t"+str(self.upper[:,tt]) 
                    print "dest to crusher: "+ str(self.destinations[jj,tt] ==0)
                    
                    
                    # Decide whether to send to a stockpile or to the crusher  
                    # comment out one of the methods (either the business logic thresholds or the bit string logic)
                    if self.destinations[jj,tt] == 0 :                                                                      
                    #if (block >= self.lower[:,tt] and block <= self.upper[:,tt]) : # send to crusher (build)                                  
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
                        pile_index = self.stockyard.add_block(block, tt)
                        print "***"
                        print "stocks"
                        print self.stockyard.stocks
                        print (("Send block to stockpile: %s %s"),block, pile_index)
                        print "updated stocks"
                        print self.stockyard.stocks
                        print "***"
                else:
                    print "Send to waste dump"
                
            # Keep crusher feed running from stockpile if necessary - ie keep crusher fully utilised       
            for kk in range(0,self.crusher_rate - self.build_cnt[:,tt]):
                print "reclaim"
                
                new_grade = (build_n+1) * self.target - build_n * self.build_av[0,tt]
                print "new grade: "+str(new_grade)
                # take block from stockpile
                grade = self.stockyard.get_block(new_grade, tt)
                print "available grade: " +str (grade)
                print self.stockyard.stocks
                
                if (grade is not None):
                                            
                    pbuild = self.build_av[0,tt]
                    self.cbuild = ((build_n)*pbuild + grade)/(build_n+1)
                    
                    #if (abs(cbuild-target) < abs(pbuild-target))
                                                           
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
        return self.eval()
        

    def eval(self):

        serror = 0
        for ii in range(1,np.shape(self.build_start)[0]):
            serror += math.pow((self.build_av[0, int(self.build_start[ii])-1 ] - self.target ), 2)
        
        penalty =   self.stockyard.piles_n[0,self.ntime-1]
        
        serror += penalty
            
        return serror   
  
        
    def plot_summary(self):
        
        print "plot_summary"
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        
        print np.shape(self.build_start)[0]
        
        for ii in range(1,np.shape(self.build_start)[0]):
            indices = np.linspace(self.build_start[ii-1] + 1,self.build_start[ii]-1,self.nblocks/2)            
            
            print indices
            c = np.cumsum(self.build_bks[:,(ii-1)*self.nblocks +1 : (ii*self.nblocks)+1]  )

            print "***"
            print (ii-1)*self.nblocks  + 1
            print (ii*self.nblocks)+1
            print np.cumsum(self.build_bks[:,(ii-1)*self.nblocks +1: (ii*self.nblocks)+1]  )
            print "***"
                        
            c = c[np.arange(1,self.nblocks,2)]
            print c
            print np.arange(2,self.nblocks,2)
            ba = np.divide(c,np.arange(2,self.nblocks,2))
            
            ax1.plot(indices,self.target*np.ones(np.shape(indices)),color = 'k')
            ax1.plot(indices, ba, color = 'k')
        
        ax2.plot(np.transpose(self.stockyard.piles_n))
        plt.legend(self.stockyard.piles, ncol=4, loc='lower right', 
                   bbox_to_anchor=[1.0, -0.5],borderaxespad=1)
        plt.subplots_adjust(left=0.07, bottom=0.15, right=0.96, top=0.96, wspace=0.17, hspace=0.17)
        fig1.savefig("output.png", bbox_inches="tight")
        plt.show()
    
    
    #np.savetxt("stockpiles.csv", piles_n, delimiter=",",fmt='%i')
    
def test():
    s = Stockpile_sim()
    s.reset()
    s.stockyard.set_example_thresholds()
    s.set_destinations(1,np.random.randint(2, size=(1, 40)) )
    s.run()
    s.plot_summary()
    return s
    