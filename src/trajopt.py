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
import numpy    as np
from gekko      import GEKKO
from quaternion import quat_mult, quat_conj, quat_rot

# CLASS DEFINITIONS
class Obstacle:
  def __init__(self, type, x, y, z, r):
    self.type = type
    self.x = x
    self.y = y
    self.z = z
    self.r = r

# FUNCTION DEFINITIONS
def trajopt(obstacles):
  m = GEKKO()
  nt = 51
  m.time = np.linspace(0, 10, nt)

  p = [] # linear position
  v = [] # linear velocity
  # a = [] # linear acceleration
  q = [] # angular position
  w = [] # angular velocity

  F_I = [] # inertial forces
  F_B = [] # body forces
  M_B = [] # body moments

  p_f = [] # final linear position
  v_f = [] # final linear velocity
  q_f = [] # final angular position
  w_f = [] # final angular velocity

  M = 10
  J = np.diag([5, 3, 2])
  P_F = [10, 10, 10]
  
  for _ in range(3):
    # initialize variables
    # a.append(m.Var(value=0))
    # a.append(m.Var(value=None, lb=-1, ub=1, fixed_initial=False))
    v.append(m.Var(value=0))
    p.append(m.Var(value=0))
    q.append(m.Var(value=0))
    w.append(m.Var(value=0))

    F_I.append(m.Var(value=0))
    F_B.append(m.Var(value=0, fixed_initial=False))
    M_B.append(m.Var(value=0, fixed_initial=False))

    p_f.append(m.Var()) # should be FV instead of Var but doesn't work for some reason
    v_f.append(m.Var())
    q_f.append(m.Var())
    w_f.append(m.Var())

  q.append(m.Var(value=0))
  q_f.append(m.Var(value=0))

  # end constraints
  m.Equations(p_f[i] == P_F[i] for i in range(0, len(p_f)))
  m.Equations(v_f[i] == 0      for i in range(0, len(v_f)))
  m.Equations(q_f[i] == 0      for i in range(1, len(q_f)))
  m.Equations(w_f[i] == 0      for i in range(0, len(w_f)))

  m.Equation(F_I[0] ==       0)
  m.Equation(F_I[1] ==       0)
  m.Equation(F_I[2] == - M * 9.81)
  m.Equation(F_B[0] ==       0)
  m.Equation(F_B[1] ==       0)

  for i in range(3):
    # dynamics
    m.Equation(p[i].dt()   == v[i])
    m.Equation(M*v[i].dt() == F_I[i] + quat_mult(quat_mult(q, [0, F_B[0], F_B[1], F_B[2]]), quat_conj(q))[i])
    m.Equation(q[i].dt()   == 0.5 * quat_mult(q, [0, w[0], w[1], w[2]])[i])
    m.Equation(np.sum(J[i][j] * w[j].dt() for j in range(len(w))) == M_B[i]
                                                                     - np.cross(w,
                                                                                np.matmul(J,
                                                                                          w))[i])
    
    # set up end constraints
    # m.Connection(p[i], p_f[i], 'end', 'end')
    # m.Connection(v[i], v_f[i], 'end', 'end')
    # m.Connection(q[i], q_f[i], 'end', 'end')
    # m.Connection(w[i], w_f[i], 'end', 'end')
    # m.fix(p[i], pos = len(m.time)-1, val = P_F[i])

  # spherical obstacles
  # for obstacle in obstacles:
  #   if obstacle.type == 'sphere':
  #     m.Equation((p[0] - obstacle.x)**2 + (p[1] - obstacle.y)**2 + (p[2] - obstacle.z)**2  >= obstacle.r**2)
  
  # minimize control input
  m.Minimize(F_B[2]**2 + M_B[2]**2)
  
  m.options.IMODE = 6 # control
  try:
    m.solve()
    success = 1
  except:
    print('Failed to solve')
    success = 0
  
  pos, vel, frc = [], [], []
  if success:
    for i in range(3):
      pos.append([p[i].value[k] for k in range(nt)])
      vel.append([v[i].value[k] for k in range(nt)])
    frc.append([F_B[2].value[k] for k in range(nt)])
  else:
    for i in range(3):
      pos.append([0 for k in range(nt)])
      vel.append([0 for k in range(nt)])
    frc.append([0 for k in range(nt)])
  
  return m.time, pos, vel, frc