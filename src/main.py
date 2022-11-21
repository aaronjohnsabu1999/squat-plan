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
from trajopt import trajopt, Obstacle
from plotter import plotter

# MAIN PROGRAM
if __name__ == '__main__':
  obstacles = []
  obstacles.append(Obstacle('sphere', 1, 1, 1, 1))
  obstacles.append(Obstacle('sphere', 8, 8, 8, 2))
  
  time, pos, vel, mom, F_B = trajopt(obstacles)
  
  # get additional solution information
  # import json
  # with open(m.path+'//results.json') as f:
  #     results = json.load(f)
  
  plotter(time, pos, vel, mom, F_B, obstacles)