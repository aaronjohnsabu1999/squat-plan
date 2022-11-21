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
random.seed(0)

# FUNCTION DEFINITIONS
def genRandomForest(numObs, excluded_points):
  obstacles = []
  for i in range(numObs):
    x = random.random() * 10
    y = random.random() * 10
    z = random.random() * 10
    r = random.random() * 2
    obstacles.append(Obstacle('sphere', {'x': x, 'y': y, 'z': z, 'r': r}))
    for obstacle in obstacles:
      for point in excluded_points:
        if obstacle.function(point[0], point[1], point[2]) < 0:
          obstacles.remove(obstacle)
          break
  return obstacles