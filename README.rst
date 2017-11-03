=============
Destination Problem Experiments
=============

Overview
========
This package contains some experiments and simulation of the open pit mine material destination problem.

An evolutionary algorithm for optimising decisions on whether to process or stockpile ore is included as well
as a heuristic rule based method ("selection cone").

Solutions are represented with a boolean matrix with a number of columns equal to the number of digging units and a number of
rows equal to the number of time steps. The interpretation of each column is a series of ore blocks extracted
by digging equipment during a period of discrete time steps (each row). Values in the matrix specify whether
to stockpile or process the ore at each time step, 0 is interpreted as to process, and a 1 as to stockpile.


Pre-requisites
==============
There are several system dependencies:

build-essential python-dev libffi-dev libtool automake autoconf libev-dev libyaml-dev libzmq-dev libqt4-dev libgraphviz-dev


Install and Run
===============
It is suggested to use a virtual environment (virtualenv)

Run python setup.py install

Run python main.py

Running the main function will run the evolutionary algorithm, save results in results directory labelled with the current
date/time and plot a visualization of the result simulating the best individual found by the algorithm.

Output
================

Running main.py should result in a graph showing the progress of the build quality with regard to target of completed builds
(for in the best solution in the EA test) and three files: build.csv, destinations.csv, and stockpiles.csv.

build.csv saves results for build performance on reaching the target and has column headings:
simulation time step,
quality target,
build index,
number of blocks in build,
quality of block added,
average build quality,
block source (stockpile or pit)
digger index (if source is pit) or stockpile index (if source is stockpile)

destinations.csv contains the destinations of all blocks in the digging sequence at each time step. This is a solution to the
problem and the best individual is provided for the EA runs.

stockpiles.csv contents of each of the stockpiles during the simulation

Data
===============
The datafile containing the small block model is in small_example.csv: Location in space is by
xyz coords, chemical attribute concentrations are in columns e.g. t_fe and rprod_*_fe.
dig_seq.csv contains a dig sequence from small_example able to be read by the simulation sim_eval.py.

Configuration
=============

