# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:22:45 2017

@author: ag
"""
import numpy as np 
import matplotlib.pyplot as plt

def get_digger_block_seq():
   # diggers = 
   # ret = [56+6*rand(1,100); 
   #        52+6*rand(1,100); 
   #        zeros(3,100)];
    return np.concatenate((
            np.random.rand(1,100) * 6 + 52,
            np.random.rand(1,100) * 6 + 56,
            np.zeros(shape=(1,100)), 
            np.zeros(shape=(1,100)), 
            np.zeros(shape=(1,100)) 
            ))



ndiggers = 5
nblocks = 20
# number of time periods
ntime = 100

# crusher model

# blocks to crush per time period (one crusher)
crusher_rate = 2 

# digger model

# digger quality of blocks to mine in each time period ()
diggers = get_digger_block_seq()
# variable to store the diggers activity in each timestep in the sim
dig_var = np.zeros((ndiggers,ntime))


# stockpile model

# thresholds for low and high grade
low = 55
high = 62

# just have one stockpile for each grade
piles = np.array(range (low, high))
npiles = len (piles)
# variable to store stockpile contents info at each timestep in the sim
piles_n = np.zeros((npiles,ntime))
# set value at first timestep to 10
piles_n[0:npiles,0] = 10

# build model
build_n = 0
build_av = np.zeros((1,ntime))
build_cnt = np.zeros((1,ntime))
build_start = np.zeros((1,1))
build_bks = np.zeros((1,ntime*ndiggers))  
build_ind = np.zeros((1,ntime*ndiggers)) # % 1 if from build, 0 otherwise

nbuild_bks = 0

upper = np.zeros((1,ntime))
lower = np.zeros((1,ntime))

target = 58

tt = 0
crush_count = 0
while (tt < ntime):
    print "time step: "+str (tt)
    
    if (tt > 0):
        piles_n[:,tt] = piles_n[:,tt-1]
        build_av[:,tt] = build_av[:,tt-1]
    
    # for each digger
    for jj in range (0,ndiggers):
        
        block = diggers[jj,tt]
        
        print "digger: "+str(jj) +"\t" + "block Q: "+ str(block)
        
        # if not waste
        if(block >= low):
            
            # calculate lower and upper threshold for selecting a block for the build
            # that would allow the target to be met
            if (build_n is 0 or build_n is nblocks):
                upper[:,tt] = 100
                lower[:,tt] = 0
            else: 
                upper[:,tt] = nblocks*target - build_n*build_av[:,tt] - (nblocks - build_n-1)*piles[0]
                lower[:,tt] = nblocks*target - build_n*build_av[:,tt] - (nblocks - build_n-1)*piles[npiles-1]
        
            # Decide whether to send to a stockpile or to the crusher
            if (block >= lower[:,tt] and block <= upper[:,tt]) : # send to crusher (build)                                  
                print "send to crusher"
                crush_count += 1
                # check if a new build is to be started (build size = 20)
                if (build_n is 20):
                    build_n = 0
                    build_start = np.array([build_start[0], tt])
            
                build_n = build_n + 1
            
                # calculate the average quality value of the build - ie before a block is added
                if (build_n is 0): # first block
                    build_av[:,tt] = block
                
                else: # blocks 2 - number of blocks (20)
                    current_build_av = build_av[:,tt]
                    build_av[:,tt] = ((build_n-1)*current_build_av + block)/build_n
            
                nbuild_bks = nbuild_bks + 1
                build_bks[:,nbuild_bks] = block
                build_ind[:,nbuild_bks] = 1 # TODO check this shouldn't it be the index of the build?
                
                build_cnt[:,tt] = build_cnt[:,tt] + 1

            else: # send to stockpile - ROM pad                
                pile_index, = np.where(piles == np.floor(block))
                piles_n [pile_index[0],tt] = piles_n [pile_index[0],tt] + 1
                print "Send to stockpile: "+str(pile_index[0])+"\t Quality: "+str(piles[pile_index[0]] )
        else:
            print "Send to waste dump"
        
    # Keep crusher feed running from stockpile if necessary - ie keep crusher fully utilised
        
    for kk in range(0,crusher_rate - build_cnt[:,tt]):
                
        new_grade = (build_n+1)*target - build_n*build_av[:,tt]
        
        # which stockpiles have material?
        temp_i, = np.where(piles_n[:,tt] > 0)
        
        if (np.size(temp_i) > 0):
            grade = np.min(np.abs(piles[temp_i] - new_grade))
            igrade = np.argmin(np.abs(piles[temp_i] - new_grade))
        
            grade = piles[temp_i[igrade]]
            
            pbuild = build_av[:,tt]
            cbuild = ((build_n)*pbuild + grade)/(build_n+1)
            
            #if (abs(cbuild-target) < abs(pbuild-target))

            #take the block from the stock pile     
            piles_n[temp_i[igrade],tt] = piles_n[temp_i[igrade],tt] - 1
            
            # send the block to the build
            build_n = build_n + 1
            
            # check if we need to start a new build
            if (build_n is 20):
                build_n = 0
                build_start = np.array([build_start[0], tt])
            
            # calculate the average quality value of the build - ie before a block is added
            if (build_n is 0): # first block
                build_av[:,tt] = block
            
            else: # blocks 2 - number of blocks (20)
                current_build_av = build_av[:,tt]
                build_av[:,tt] = ((build_n-1)*current_build_av + block)/build_n
            
            
            nbuild_bks = nbuild_bks + 1
            build_bks[:,nbuild_bks] = grade
           # build_ind[:,nbuild_bks] = 1 # TODO check this shouldn't it be the index of the build?
                
            build_cnt[:,tt] = build_cnt[:,tt] + 1
                       

    tt = tt + 1

print "Crusher count: " + str(crush_count)
#t = np.arange(0.,100,1)
#plt.plot(t,upper[0,:])