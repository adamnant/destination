=============
Destination Problem Experiments
=============

Overview
========
These scripts allow running of the destination problem and generation of results.
The current state of the code is one of development.

Pre-requisites
==============
There are several system dependencies:

build-essential python-dev libffi-dev libtool automake autoconf libev-dev libyaml-dev libzmq-dev libqt4-dev libgraphviz-dev


Install and Run
===============

It is suggested to use a virtual environment (virtualenv)

Run python setup.py install

Run python main.py


Running the main function will run the evolutionary algorithm, save results in results directory
labelled with the current date/time and plot a visualization of the result simulating 
the best individual found by the algorithm. To use the selection cone method set 


Simulation and Algorithm
===============

The stockyard and mining simulation that results are based on is in ea/sim_eval.py
The evolutionary algorithm is in ea/destinations_ga.py
Some plotting, operators and other tools are in ea/utils.py
To construct the digging sequence requires adaptation of methods get_small_example and long sequence in 
the dig_sequence_utils.py

Data
===============
The datafile containing the small block model is in small_example.csv: Location in space is by
xyz coords, chemical attribute concentrations are in columns e.g. t_fe and rprod_*_fe.

dig_seq.csv contains a dig sequence from small_example able to be read by the simulation sim_eval.py.

Configuration
=============

Currently configurations are set in the code - should be set to use a config file and suitable comments.