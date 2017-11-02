=============
Destination Problem Experiments
=============

Overview
========
These scripts are some experiments and simulation of the open pit mine material destination problem.

Pre-requisites
==============
There are several system dependencies:

build-essential python-dev libffi-dev libtool automake autoconf libev-dev libyaml-dev libzmq-dev libqt4-dev libgraphviz-dev


Install and Run
===============
It is suggested to use a virtual environment (virtualenv)

Run python setup.py install

Run python main.py

Running the main function will run the evolutionary algorithm, save results in results directory labelled with the current date/time and plot a visualization of the result simulating the best individual found by the algorithm. To use the selection cone method the file ea/sim_eval.py must be edited manually (see notes in config below).


Data
===============
The datafile containing the small block model is in small_example.csv: Location in space is by
xyz coords, chemical attribute concentrations are in columns e.g. t_fe and rprod_*_fe.
dig_seq.csv contains a dig sequence from small_example able to be read by the simulation sim_eval.py.

Configuration
=============

