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

# PACKAGE IMPORTS
import random
from trajopt import Obstacle

# FUNCTION DEFINITIONS
def genRandomForest(numObs, boundaryPoints, maxRad = 4.0, minRad = 0.5, seed = 0):
  random.seed(seed)
  P_I, P_F = boundaryPoints
  obstacles = []
  while len(obstacles) < numObs:
    for i in range(numObs-len(obstacles)):
      x = random.random() * (P_F[0] - P_I[0]) + P_I[0]
      y = random.random() * (P_F[1] - P_I[1]) + P_I[1]
      z = random.random() * (P_F[2] - P_I[2]) + P_I[2]
      r = random.random() * (maxRad - minRad) + minRad
      obstacles.append(Obstacle('sphere', {'x': x, 'y': y, 'z': z, 'r': r}))
    for obstacle in obstacles:
      for point in boundaryPoints:
        if obstacle.function(point[0], point[1], point[2]) < 0:
          obstacles.remove(obstacle)
          break
  return obstacles