#!/usr/bin/env python3

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_sphere(ax, x, y, z, r):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = x + r * np.cos(u) * np.sin(v)
    y = y + r * np.sin(u) * np.sin(v)
    z = z + r * np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

def quat_mult(q_1, q_2):
    w_1, x_1, y_1, z_1 = q_1
    w_2, x_2, y_2, z_2 = q_2
    w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
    x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
    y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
    z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2
    return [w, x, y, z]

def quat_conj(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

def quat_rot(q, v):
    q_conj = quat_conj(q)
    q_v = [0, v[0], v[1], v[2]]
    q_v_rot = quat_mult(q, quat_mult(q_v, q_conj))
    return [q_v_rot[1], q_v_rot[2], q_v_rot[3]]

m = GEKKO()
nt = 51
m.time = np.linspace(0, 10, nt)

p = [] # linear position
v = [] # linear velocity
a = [] # linear acceleration
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

# 3D
for _ in range(3):
    # initialize variables
    # a.append(m.Var(value=0))
    a.append(m.Var(value=None, lb=-1, ub=1, fixed_initial=False))
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

for i in range(3):
    # dynamics
    m.Equation(p[i].dt() == v[i])
    m.Equation(v[i].dt() == a[i])
    m.Equation(M*a[i] == F_I[i] + quat_mult(quat_mult(q, [0, F_B[0], F_B[1], F_B[2]]), quat_conj(q))[i])
    m.Equation(q[i].dt() == 0.5 * quat_mult(q, [0, w[0], w[1], w[2]])[i])
    m.Equation(np.sum(J[i][j] * w[j].dt() for j in range(len(w))) == - np.cross(w,
                                                                                np.multiply(J,
                                                                                            w))[i]
                                                                        + M_B[i])
    
    # set up end constraints
    m.Connection(p[i], p_f[i], 'end', 'end')
    m.Connection(v[i], v_f[i], 'end', 'end')
    m.Connection(q[i], q_f[i], 'end', 'end')
    m.Connection(w[i], w_f[i], 'end', 'end')

# end constraints
P_F = [10, 10, 10]
m.Equations(p_f[i] == P_F[i] for i in range(0, len(p_f)))
m.Equations(v_f[i] == 0      for i in range(0, len(v_f)))
m.Equations(q_f[i] == 0      for i in range(1, len(q_f)))
m.Equations(w_f[i] == 0      for i in range(0, len(w_f)))

m.Equation(F_I[0] ==       0)
m.Equation(F_I[1] ==       0)
m.Equation(F_I[2] == - M * 9.81)
m.Equation(F_B[0] ==       0)
m.Equation(F_B[1] ==       0)

# spherical obstacles
spheres = [
    (1, 1, 1, 1), # x, y, z, r
    (8, 8, 8, 2)
]
for s in spheres:
    m.Equation((p[0]-s[0])**2 + (p[1]-s[1])**2 + (p[2]-s[2])**2 >= s[3]**2)

# minimize control input
m.Minimize(F_B[2]**2 + M_B[2]**2)

m.options.IMODE = 6 # control
m.solve()

# get additional solution information
# import json
# with open(m.path+'//results.json') as f:
#     results = json.load(f)

plt.figure()
plt.subplot(2,1,1)
plt.plot(m.time, a[0].value[1:] + [a[0].value[-1]], '-', label='a_x', drawstyle='steps-post')
plt.plot(m.time, a[1].value[1:] + [a[1].value[-1]], '-', label='a_y', drawstyle='steps-post')
plt.plot(m.time, a[2].value[1:] + [a[2].value[-1]], '-', label='a_z', drawstyle='steps-post')
plt.grid()
plt.legend(loc='best')
plt.ylabel('Input')
plt.subplot(2,1,2)
plt.plot(m.time,p[0].value,'-',label='p_x')
plt.plot(m.time,p[1].value,'-',label='p_y')
plt.plot(m.time,p[2].value,'-',label='p_z')
plt.grid()
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(p[0].value, p[1].value, p[2].value)
for s in spheres:
    plot_sphere(ax, s[0], s[1], s[2], s[3])

plt.show()