# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * Department of Mechanical and Aerospace Engineering      *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Ryan Nemiroff, Brett T. Lopez *
# * Contact: {aaronjs, ryguyn, btlopez}@ucla.edu             *
# *                                                         *
# ***********************************************************/

import numpy as np

from trajopt  import trajopt
from plotter  import plotter
from forester import genRandomForest

# MAIN PROGRAM
if __name__ == '__main__':
  numStep = 100
  totTime = 20.0
  maxIter = 1000
  solverParams  = {'numStep': numStep, 'totTime': totTime, 'maxIter': maxIter, 'set_QF': False, 'set_WF': False}
  
  M = 10
  J = np.diag([0.1, 0.1, 0.1])
  systemParams  = {'M': M, 'J': J}
  
  P_I = [5.0,  0.0, 5.0]
  P_F = [5.0, 40.0, 5.0]
  V_I = [ 0,  0,  0]
  V_F = [ 0,  0,  0]
  Q_I = [ 1,  0,  0,  0]
  Q_F = [ 1,  0,  0,  0]
  W_I = [ 0,  0,  0]
  W_F = [ 0,  0,  0]
  boundaryCons  = {'P_I': P_I, 'P_F': P_F, 'V_I': V_I, 'V_F': V_F, 'Q_I': Q_I, 'Q_F': Q_F, 'W_I': W_I, 'W_F': W_F}
  
  K_f =  0.01
  K_m = 20.00
  K_p =  0.00
  K_v =  0.00
  controlParams = {'K_f': K_f, 'K_m': K_m, 'K_p': K_p, 'K_v': K_v}

  numObs =  0
  minRad =  1.0
  maxRad =  0.5
  obstacles = genRandomForest(numObs, [P_I, P_F], maxRad, minRad, seed = 0)
  
  time, F_B, mom, pos, vel, omg, quat = trajopt(solverParams, systemParams, boundaryCons, controlParams, obstacles)
  
  # get additional solution information
  # import json
  # with open(m.path+'//results.json') as f:
  #     results = json.load(f)
  
  ParamsText =    " numStep : " + str(numStep)  + \
                "\n totTime : " + str(totTime)  + \
                "\n M       : " + str(M)        + \
                "\n J       : " + str(J)        + \
                "\n P_I     : " + str(P_I)      + \
                "\n P_F     :"  + str(P_F)      + \
                "\n K_f     : " + str(K_f)      + \
                "\n K_m     : " + str(K_m)      + \
                "\n K_p     : " + str(K_p)      + \
                "\n K_v     : " + str(K_v)      + \
                "\n numObs  : " + str(numObs)   + \
                "\n minRad  : " + str(minRad)   + \
                "\n maxRad  : " + str(maxRad)
  print(ParamsText)

  plotter(time, pos, vel, quat, omg, F_B, mom, obstacles)