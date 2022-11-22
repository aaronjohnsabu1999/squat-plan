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
  numStep = 100
  totTime = 10
  maxIter = 2000
  solverParams  = {'numStep': numStep, 'totTime': totTime, 'maxIter': maxIter}
  
  M = 10
  J = np.diag([0.1, 0.1, 0.1])
  systemParams  = {'M': M, 'J': J}
  
  P_I = [ 0,  0,  0]
  P_F = [10, 10, 10]
  V_F = [ 0,  0,  0]
  Q_F = [ 1,  0,  0,  0]
  W_F = [ 0,  0,  0]
  boundaryCons  = {'P_I': P_I, 'P_F': P_F, 'V_F': V_F, 'Q_F': Q_F, 'W_F': W_F}
  
  K_f =  0.01
  K_m = 10.00
  K_p =  0.00
  K_v =  0.00
  controlParams = {'K_f': K_f, 'K_m': K_m, 'K_p': K_p, 'K_v': K_v}

  numObs =  0
  minRad =  1.5
  maxRad =  0.5
  obstacles = genRandomForest(numObs, [P_I, P_F], maxRad, minRad, seed = 2)
  
  time, F_B, mom, pos, vel, omg, quat = trajopt(solverParams, systemParams, boundaryCons, controlParams, obstacles)
  
  # get additional solution information
  # import json
  # with open(m.path+'//results.json') as f:
  #     results = json.load(f)
  
  plotter(time, pos, vel, quat, omg, F_B, mom, obstacles)