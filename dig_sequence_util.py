# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:56:34 2017

@author: ag
"""
import os
import numpy as np
from numpy import genfromtxt
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot

def  above (block1, block2):

    x1 = block1[0]
    y1 = block1[1]
    z1 = block1[2]
    
    x2 = block2[0]
    y2 = block2[1]
    z2 = block2[2]

    ret = z1==z2+5 and np.abs(x1-x2)<=25 and np.abs(y1-y2)<=25
    return ret


def read_block_file(fname):
    my_data = genfromtxt(fname, delimiter=',',skip_header=1)
    return my_data
    

def precedence_graph (blocks):
    """
    :param blocks: array with rows: block name, x-coord, y-coord, z-coord, Fe
    """
    n = len(blocks)
    G = nx.DiGraph()
    
    for ii in range(n):      
        G.add_node(str(blocks[ii,0]) )
    
    for ii in range(n):      
        for jj in range(n):            
            if above (blocks[ii,1:4], blocks[jj,1:4] ) :
                G.add_edge(str(blocks[jj,0]), str(blocks[ii,0]) )
    
    return G
            

def dig_sequence(G, blocks):
    G_sorted = nx.topological_sort(G) 
    d = {}    
    for ii in range(len(blocks)):
        d[str(blocks[ii,0])] = blocks[ii,5]
    
    seq = []
    for el in G_sorted :
        seq= np.append(seq,d[el])
        
    return seq


def get_small_example():
# small_example.csv

    data = read_block_file("small_example.csv")
    blocks = data[:,(0,1,2,3,4,6)]
    G = precedence_graph(blocks)
    #write_dot(G,'file.dot')
    G_sorted = nx.topological_sort(G) 

    digger = dig_sequence(G, blocks)
    np.savetxt("dig_seq.csv",digger, delimiter=",",newline="\n")
    return digger


def test():
    get_small_example()
    data = np.genfromtxt ('dig_seq.csv', delimiter=",")

    ntime = 40
    seq = np.concatenate((\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime))\
                )) 
    print seq
    for ii in range(len(seq[0,:])):
        if (ii < len(data)):
            seq[0,ii]=data[ii]
        else:
            break
        
    print seq[0,:]
    ntime = 10
    seq = np.concatenate((\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime)),\
                np.zeros((1,ntime))\
                )) 

    for ii in range(len(seq[0,:])):
        if (ii < len(data)):
            seq[0,ii]=data[ii]
        else:
            break
    
    print seq[0,:]
    
    print type (data)

######################
#    
#A = block_data_z_x_y();
#
#
#%ii = randperm(n);
#%A = A(ii,:);
#
#A_prec = zeros(n,n);
#
#for ii = 1:n
#    for jj = 1:n
#        A_prec(ii,jj) = above(A(ii,2:4),A(jj,2:4));
#    end
#end
#
#close all
#figure(1)
#hold on 
#
#for ii = 1:n
#    
#    if (A(ii,7) <= 55)
#        col = 0.5*[1 1 1];
#    else
#        col = [1 0 0];
#    end
#    
#    plotcube(edges,[A(ii,2), A(ii,3), A(ii,4)]-edges/2, 0.5, col);
#        
#end
#
#mine_order = graphtopoorder(sparse(A_prec));
#
#for ii = 1:(n-1)
#    
#    index1 = mine_order(ii);
#    index2 = mine_order(ii+1);
#    
#    B = [A(index1,2:4); A(index2,2:4)];
#    
#    plot3(B(:,1), B(:,2), B(:,3),'k');
#    
#end
#
#view(48,24)
#hold off