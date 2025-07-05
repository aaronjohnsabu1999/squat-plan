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

import numpy             as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# FUNCTION DEFINITIONS
def plot_sphere(ax, x, y, z, r):
  u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
  x = x + r * np.cos(u) * np.sin(v)
  y = y + r * np.sin(u) * np.sin(v)
  z = z + r * np.cos(v)
  ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

def plot_cylinder(ax, x, y, z, r):
  u, v = np.mgrid[0:2 * np.pi:30j, 0:10:30j]
  x = x + r * np.cos(u)
  y = y + r * np.sin(u)
  z = z + v
  ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

def plotter(time, pos, vel, quat, omg, F_B, mom, obstacles):
  axes = ['x', 'y', 'z']
  
  plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot3D(pos[0], pos[1], pos[2], 'red')
  for obstacle in obstacles:
    if obstacle.type == 'sphere':
      plot_sphere  (ax, obstacle.params['x'], obstacle.params['y'], obstacle.params['z'], obstacle.params['r'])
    elif obstacle.type == 'cylinder':
      plot_cylinder(ax, obstacle.params['x'], obstacle.params['y'], obstacle.params['z'], obstacle.params['r'])
  
  plt.figure()
  
  plt.subplot(6,1,1)
  for i in range(3):
    plt.plot(time, pos[i], '-', label='p_'+axes[i])
  plt.grid()
  plt.ylabel('Position')
  plt.xlabel('Time')
  plt.legend(loc='best')

  plt.subplot(6,1,2)
  for i in range(3):
    plt.plot(time, vel[i], '-', label='v_'+axes[i])
  plt.grid()
  plt.ylabel('Velocity')
  plt.xlabel('Time')
  plt.legend(loc='best')

  plt.subplot(6,1,3)
  for i in range(4):
    plt.plot(time, quat[i], '-', label='q_'+str(i))
  q_norm = []
  for t in range(len(quat[0])):
    q_norm.append(np.linalg.norm([quat[i][t] for i in range(4)]))
  plt.plot(time, q_norm, '-.', label='||q||')
  plt.grid()
  plt.ylabel('Rotation (Quaternion)')
  plt.xlabel('Time')
  plt.legend(loc='best')

  plt.subplot(6,1,4)
  for i in range(3):
    plt.plot(time, omg[i], '-', label='w_'+axes[i])
  plt.grid()
  plt.ylabel('Angular Velocity')
  plt.xlabel('Time')
  plt.legend(loc='best')

  plt.subplot(6,1,5)
  plt.plot(time, F_B[0], '-', label='F_B', drawstyle='steps-post')
  plt.grid()
  plt.ylabel('Thrust')
  plt.xlabel('Time')
  plt.legend(loc='best')
  
  plt.subplot(6,1,6)
  for i in range(3):
    plt.plot(time, mom[i], '-', label='M_B'+axes[i])
  plt.grid()
  plt.ylabel('Moment')
  plt.xlabel('Time')
  plt.legend(loc='best')
  
  plt.show()