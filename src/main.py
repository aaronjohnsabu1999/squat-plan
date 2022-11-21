# /***********************************************************
# *                                                         *
# * Copyright (c) 2022                                      *
# *                                                         *
# * Department of Mechanical and Aerospace Engineering      *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Ryan Nemiroff, Brett T. Lopez *
# * Contact: {aaronjs, ryguyn, btlopez}@ucla.edu             *
# *                                                         *
# ***********************************************************/

#!/usr/bin/env python3

# PACKAGE IMPORTS
import numpy as np
from trajopt  import trajopt
from plotter  import plotter
from forester import genRandomForest

# MAIN PROGRAM
if __name__ == '__main__':
  numStep = 150
  totTime = 20
  solverParams = {'numStep': numStep, 'totTime': totTime}
  
  K_f =  0.01
  K_m = 10
  M = 10
  J = np.diag([0.1, 0.1, 0.1])
  P_I = [ 0,  0,  0]
  P_F = [10, 10, 10]
  systemParams = {'K_f': K_f, 'K_m': K_m, 'M': M, 'J': J, 'P_I': P_I, 'P_F': P_F}

  obstacles = genRandomForest(15, [P_I, P_F])
  # obstacles.append(Obstacle('sphere',   {'x': 1, 'y': 1, 'z': 1, 'r': 1}))
  # obstacles.append(Obstacle('sphere',   {'x': 8, 'y': 8, 'z': 8, 'r': 2}))
  # obstacles.append(Obstacle('cylinder', {'x': 5, 'y': 5, 'z': 0, 'r': 2}))
  
  time, pos, vel, mom, F_B = trajopt(solverParams, systemParams, obstacles)
  
  # get additional solution information
  # import json
  # with open(m.path+'//results.json') as f:
  #     results = json.load(f)
  
  plotter(time, pos, vel, mom, F_B, obstacles)