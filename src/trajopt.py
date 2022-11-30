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
from quaternion import quat_mult, quat_conj

# CLASS DEFINITIONS
class Obstacle:
  def __init__(self, type, params):
    self.type = type
    self.params = params
    
  def function(self, x, y, z):
    if self.type == 'sphere':
      return (x - self.params['x'])**2 + (y - self.params['y'])**2 + (z - self.params['z'])**2 - self.params['r']**2
    elif self.type == 'cylinder':
      return (x - self.params['x'])**2 + (y - self.params['y'])**2 - self.params['r']**2

# FUNCTION DEFINITIONS
def trajopt(solverParams, systemParams, boundaryCons, controlParams, obstacles):
  numStep = solverParams['numStep']
  totTime = solverParams['totTime']
  maxIter = solverParams['maxIter']

  M   = systemParams['M']
  J   = systemParams['J']

  P_I = boundaryCons['P_I']
  P_F = boundaryCons['P_F']
  V_I = boundaryCons['V_I']
  V_F = boundaryCons['V_F']
  Q_I = boundaryCons['Q_I']
  Q_F = boundaryCons['Q_F']
  W_I = boundaryCons['W_I']
  W_F = boundaryCons['W_F']
  F_I = [0, 0, - M * 9.81]
  
  m = GEKKO(remote = True)
  m.time = np.linspace(0, totTime, numStep)

  p = [] # linear position
  v = [] # linear velocity
  q = [] # angular position
  w = [] # angular velocity

  f_I = [] # inertial forces
  f_B = [] # body forces
  m_B = [] # body moments

  p_f = [] # final linear position
  v_f = [] # final linear velocity
  q_f = [] # final angular position
  w_f = [] # final angular velocity

  q.append(m.Var(value=1))
  q_f.append(m.Var(value=1))
  
  for i in range(3):
    # initialize variables
    v.append(m.Var(value=V_I[i]))
    p.append(m.Var(value=P_I[i], fixed_initial=True))
    q.append(m.Var(value=Q_I[i]))
    w.append(m.Var(value=W_I[i]))

    f_I.append(m.FV(value=F_I[i]))
    if i < 2:
      f_B.append(m.Var(value=0))
    else:
      f_B.append(m.Var(value=0, lb = -100, ub = 100, fixed_initial=False))
    m_B.append(m.Var(value=0, lb = -200, ub = 200, fixed_initial=False))

    p_f.append(m.Var())
    v_f.append(m.Var())
    q_f.append(m.Var())
    w_f.append(m.Var())

  p[2].lower = 0
  
  # end constraints
  m.Equations(p_f[i] == P_F[i] for i in range(0, len(p_f)))
  m.Equations(v_f[i] == V_F[i] for i in range(0, len(v_f)))
  m.Equations(q_f[i] == Q_F[i] for i in range(0, len(q_f)))
  m.Equations(w_f[i] == W_F[i] for i in range(0, len(w_f)))

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
    eqs.append(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 == 1)
  
  obstacle_check = True
  if obstacle_check:
    for obstacle in obstacles:
      eqs.append(obstacle.function(p[0], p[1], p[2]) >= 0)
  eqs = m.Equations(eqs)

  # set up end constraints
  for i in range(3):
    m.Connection(p[i], p_f[i], 'end', 'end')
    m.Connection(v[i], v_f[i], 'end', 'end')
    if solverParams['set_QF']:
      m.Connection(q[i+1], q_f[i+1], 'end', 'end')
    if solverParams['set_WF']:
      m.Connection(w[i], w_f[i], 'end', 'end')

  # minimize control input
  K_f = controlParams['K_f']
  K_m = controlParams['K_m']
  K_p = controlParams['K_p']
  K_v = controlParams['K_v']
  m.Minimize(K_f*f_B[2]**2
           + K_m*np.sum([(m_B[i]     )**2 for i in range(len(m_B))])
           + K_p*np.sum([(p[i]-P_F[i])**2 for i in range(len(p  ))])
           + K_v*np.sum([(v[i]-V_F[i])**2 for i in range(len(v  ))]))
  
  m.options.IMODE  = 6 # control
  m.options.SOLVER = 3 # IPOPT
  m.options.MAX_ITER = maxIter
  # m.options.OTOL = 1e-4
  # m.options.RTOL = 1e-4
  try:
    m.solve(disp = True)
    success = 1
  except:
    print('Failed to solve')
    success = 0
  
  force, mom, pos, vel, omg, quat = [], [], [], [], [], []
  if success:
    force.append( [f_B[2].value[k] for k in range(numStep)])
    for i in range(3):
      mom.append( [m_B[i].value[k] for k in range(numStep)])
      pos.append( [p  [i].value[k] for k in range(numStep)])
      vel.append( [v  [i].value[k] for k in range(numStep)])
      omg.append( [w  [i].value[k] for k in range(numStep)])
    for i in range(4):
      quat.append([q  [i].value[k] for k in range(numStep)])
    
  else:
    force.append( [0 for k in range(numStep)])
    for i in range(3):
      mom.append( [0 for k in range(numStep)])
      pos.append( [0 for k in range(numStep)])
      vel.append( [0 for k in range(numStep)])
      omg.append( [0 for k in range(numStep)])
    for i in range(4):
      quat.append([0 for k in range(numStep)])
  
  return m.time, force, mom, pos, vel, omg, quat