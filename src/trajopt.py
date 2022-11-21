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
  nt = 100
  m.time = np.linspace(0, 10, nt)

  p = [] # linear position
  v = [] # linear velocity
  # a = [] # linear acceleration
  q = [] # angular position
  w = [] # angular velocity

  f_I = [] # inertial forces
  f_B = [] # body forces
  m_B = [] # body moments

  p_f = [] # final linear position
  v_f = [] # final linear velocity
  q_f = [] # final angular position
  w_f = [] # final angular velocity

  M = 1
  J = np.diag([1, 1, 1])
  P_I = [ 0,  0,  0]
  P_F = [10, 10, 10]
  F_I = [0, 0, - M * 9.81]
  
  q.append(m.Var(value=1))
  q_f.append(m.Var(value=1))
  
  for i in range(3):
    # initialize variables
    # a.append(m.Var(value=0))
    # a.append(m.Var(value=None, lb=-1, ub=1, fixed_initial=False))
    v.append(m.Var(value=0))
    p.append(m.Var(value=P_I[i], fixed_initial=True))
    q.append(m.Var(value=0))
    w.append(m.Var(value=0))

    f_I.append(m.FV(value=F_I[i]))
    if i < 2:
      f_B.append(m.Var(value=0))
    else:
      f_B.append(m.Var(value=0, fixed_initial=False))
    m_B.append(m.Var(value=0, fixed_initial=False))

    p_f.append(m.Var()) # should be FV instead of Var but doesn't work for some reason
    v_f.append(m.Var())
    q_f.append(m.Var())
    w_f.append(m.Var())

  # end constraints
  m.Equations(p_f[i] == P_F[i] for i in range(0, len(p_f)))
  m.Equations(v_f[i] == 0      for i in range(0, len(v_f)))
  m.Equations(q_f[i] == 0      for i in range(1, len(q_f)))
  m.Equations(w_f[i] == 0      for i in range(0, len(w_f)))

  # m.Equation(f_B[0] == 0)
  # m.Equation(f_B[1] == 0)

  eqs = []
  for i in range(3):
    # dynamics
    eqs.append(p[i].dt()   == v[i])
    eqs.append(M*v[i].dt() == f_I[i] + quat_mult(quat_mult(q, [0, f_B[0], f_B[1], f_B[2]]), quat_conj(q))[i+1])
    eqs.append(q[i].dt()   == 0.5 * quat_mult(q, [0, w[0], w[1], w[2]])[i])
    eqs.append(np.sum([J[i][j] * w[j].dt() for j in range(len(w))]) == m_B[i]
                                                                     - np.cross(w,
                                                                                np.matmul(J,
                                                                                          w))[i])
      
  # spherical obstacles
  # for obstacle in obstacles:
  #   if obstacle.type == 'sphere':
  #     m.Equation((p[0] - obstacle.x)**2 + (p[1] - obstacle.y)**2 + (p[2] - obstacle.z)**2  >= obstacle.r**2)
  eqs = m.Equations(eqs)

  # set up end constraints
  for i in range(3):
    m.Connection(p[i], p_f[i], 'end', 'end')
    # m.Connection(v[i], v_f[i], 'end', 'end')
    # m.Connection(q[i], q_f[i], 'end', 'end')
    # m.Connection(w[i], w_f[i], 'end', 'end')
    # m.fix(p[i], pos = len(m.time)-1, val = P_F[i])


  # minimize control input
  m.Minimize(f_B[2]**2 + m_B[2]**2)
  
  m.options.IMODE  = 6 # control
  m.options.SOLVER = 3 # IPOPT
  m.options.MAX_ITER = 250
  try:
    m.solve()
    success = 1
  except:
    print('Failed to solve')
    success = 0
  
  pos, vel, mom, force = [], [], [], []
  if success:
    for i in range(3):
      pos.append([p  [i].value[k] for k in range(nt)])
      vel.append([v  [i].value[k] for k in range(nt)])
      mom.append([m_B[i].value[k] for k in range(nt)])
    force.append([f_B[2].value[k] for k in range(nt)])
  else:
    for i in range(3):
      pos.append([0 for k in range(nt)])
      vel.append([0 for k in range(nt)])
      mom.append([0 for k in range(nt)])
    force.append([0 for k in range(nt)])
  
  return m.time, pos, vel, mom, force