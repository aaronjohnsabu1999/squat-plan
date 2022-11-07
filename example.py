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
m.time = np.linspace(0, 5, nt)

# useful vector for gekko equations
final_m = np.zeros(nt)
final_m[-1] = 1.0
final = m.Param(value=final_m)

a = [] # acceleration (control input)
v = [] # velocity
p = [] # position

# 3D
for _ in range(3):
    # initialize varibales
    a.append(m.Var(value=0, lb=-1, ub=1))
    v.append(m.Var(value=0))
    p.append(m.Var(value=0))

    # dynamics
    m.Equation(p[-1].dt() == v[-1])
    m.Equation(v[-1].dt() == a[-1])

    # final condition (reach position (10, 10, 10))
    m.Equation((p[-1] - 10)*final == 0)

# spherical obstacle
m.Equation((p[0]-5)**2 + (p[1]-5)**2 + (p[2]-5)**2 >= 1)

# minimize control input
m.Minimize((a[0]*a[0] + a[1]*a[1] + a[2]*a[2]))

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
plt.legend(loc='best')
plt.ylabel('Input')
plt.subplot(2,1,2)
plt.plot(m.time,p[0].value,'-',label='p_x')
plt.plot(m.time,p[1].value,'-',label='p_y')
plt.plot(m.time,p[2].value,'-',label='p_z')
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(p[0].value, p[1].value, p[2].value)
plot_sphere(ax, 5, 5, 5, 1)

plt.show()
