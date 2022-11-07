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

m = GEKKO()
nt = 51
m.time = np.linspace(0, 10, nt)

a = [] # acceleration (control input)
v = [] # velocity
p = [] # position
p_f = [] # final position
v_f = [] # final velocity

# 3D
for _ in range(3):
    # initialize varibales
    a.append(m.Var(value=0, lb=-1, ub=1))
    v.append(m.Var(value=0))
    p.append(m.Var(value=0))

    # dynamics
    m.Equation(p[-1].dt() == v[-1])
    m.Equation(v[-1].dt() == a[-1])

    # set up end constraints
    p_f.append(m.Var()) # should be FV instead of Var but doesn't work for some reason
    m.Connection(p[-1], p_f[-1], 'end', 'end')
    v_f.append(m.Var())
    m.Connection(v[-1], v_f[-1], 'end', 'end')

# end constraints
m.Equations((p_f[0] == 10, p_f[1] == 10, p_f[2] == 10))
m.Equations(vf == 0 for vf in v_f)

# spherical obstacles
spheres = [
    (1, 1, 1, 1), # x, y, z, r
    (8, 8, 8, 2)
]
for s in spheres:
    m.Equation((p[0]-s[0])**2 + (p[1]-s[1])**2 + (p[2]-s[2])**2 >= s[3]**2)

# minimize control input
m.Minimize(a[0]**2 + a[1]**2 + a[2]**2)

m.options.IMODE = 6 # control
m.solve()

# get additional solution information
# import json
# with open(m.path+'//results.json') as f:
#     results = json.load(f)

plt.figure()
plt.subplot(2,1,1)
plt.plot(m.time,a[0].value,'-',label='a_x')
plt.plot(m.time,a[1].value,'-',label='a_y')
plt.plot(m.time,a[2].value,'-',label='a_z')
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
