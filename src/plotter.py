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
import numpy             as np
import matplotlib.pyplot as plt

# FUNCTION DEFINITIONS
def plot_sphere(ax, x, y, z, r):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = x + r * np.cos(u) * np.sin(v)
    y = y + r * np.sin(u) * np.sin(v)
    z = z + r * np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

def plotter(time, pos, vel, F_B, obstacles):
    axes = ['x', 'y', 'z']
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, F_B[0], '-', label='F_B', drawstyle='steps-post')
    plt.grid()
    plt.legend(loc='best')
    plt.ylabel('Thrust')
    plt.subplot(2,1,2)
    for i in range(3):
      plt.plot(time, pos[i], '-', label='p_'+axes[i])
    plt.grid()
    plt.ylabel('Position')
    plt.xlabel('Time')
    plt.legend(loc='best')

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(pos[0], pos[1], pos[2])
    for obstacle in obstacles:
      if obstacle.type == 'sphere':
        plot_sphere(ax, obstacle.x, obstacle.y, obstacle.z, obstacle.r)

    plt.show()