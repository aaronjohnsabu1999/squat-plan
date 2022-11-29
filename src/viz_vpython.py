from vpython import *
import numpy as np

import config

objects = {'static': []}

def init():
    pass # nothing to initialize

def show_once():
    pass # VPython updates display automatically

def show():
    while True:
        sleep(1)

def update_obj(obj, name, index=0):
    if name in objects and index < len(objects[name]):
        objects[name][index].visible = False
        objects[name][index] = obj
    elif name in objects and index == len(objects[name]):
        objects[name].append(obj)
    elif name not in objects and index == 0:
        objects[name] = [obj]

def make_ellipsoid(x, y, z, rx, ry, rz):
    return ellipsoid(pos=vector(x,y,z), length=2*rx, height=2*ry, width=2*rz)

def make_cylinder(axis, x_or_y, y_or_z, r):
    axes = ['x', 'y', 'z']
    axis_str = axes[axis]
    axes.remove(axis_str)

    map_width = (config.MAP_WX, config.MAP_WY, config.MAP_WZ)[axis]

    pos = vector(0,0,0)
    setattr(pos, axes[0], x_or_y)
    setattr(pos, axes[1], y_or_z)
    setattr(pos, axis_str, 0)

    axis = vector(0,0,0)
    setattr(axis, axis_str, map_width)

    return cylinder(pos=pos, axis=axis, radius=r)

def make_box(x, y, z, rx, ry, rz):
    return box(pos=(x,y,z), length=2*rx, height=2*ry, width=2*rz) 

def make_path(x, y, z, color):
    points = np.array([x, y, z]).T
    return curve(*list(vector(*p) for p in points), color=color, radius=0.05)

def make_axes(p, q):
    R = q.rot_mat()
    b_x = arrow(pos=vector(*p), axis=vector(*R[:,0]), up=vector(*R[:,1]), length=1.5, color=color.red)
    b_y = arrow(pos=vector(*p), axis=vector(*R[:,1]), up=vector(*R[:,2]), length=1.5, color=color.green)
    b_z = arrow(pos=vector(*p), axis=vector(*R[:,2]), up=vector(*R[:,0]), length=1.5, color=color.blue)
    return b_x, b_y, b_z

def add_ellipsoid_obstacle(x, y, z, rx, ry, rz):
    obj = make_ellipsoid(x, y, z, rx, ry, rz)
    obj.color = vector(1,0,0)
    objects['static'].append(obj)

def add_box_obstacle(x, y, z, rx, ry, rz):
    obj = make_box(x, y, z, rx, ry, rz)
    obj.color = vector(1,0,0)
    objects['static'].append(obj)

def add_cylinder_obstacle(axis, x_or_y, y_or_z, r):
    obj = make_cylinder(axis, x_or_y, y_or_z, r)
    obj.color = vector(1,0,0)
    objects['static'].append(obj)

def add_obstacles(obstacles):
    for obs in obstacles:
        if obs.type == 'ellipsoid':
            add_ellipsoid_obstacle(obs.x, obs.y, obs.z, obs.rx, obs.ry, obs.rz)
        elif obs.type == 'box':
            add_box_obstacle(obs.x, obs.y, obs.z, obs.rx, obs.ry, obs.rz)
        elif obs.type == 'cylinder':
            add_cylinder_obstacle(obs.axis, obs.x_or_y, obs.y_or_z, obs.r)

def update_gekko_path(x, y, z):
    obj = make_path(x, y, z, color=vector(0,1,0))
    update_obj(obj, 'gekko_path')

def update_smooth_path(x, y, z):
    obj = make_path(x, y, z, color=vector(0,0,1))
    update_obj(obj, 'smooth_path')

def update_vehicle(p, q, collision_r, sense_r, plan_r):
    x, y, z = p

    obj = make_ellipsoid(x, y, z, collision_r, collision_r, collision_r)
    obj.color = vector(1,0,1)
    update_obj(obj, 'vehicle')

    b_x, b_y, b_z = make_axes(p, q)
    update_obj(b_x, 'vehicle_pose', 0)
    update_obj(b_y, 'vehicle_pose', 1)
    update_obj(b_z, 'vehicle_pose', 2)

    obj = make_ellipsoid(x, y, z, sense_r, sense_r, sense_r)
    obj.color = vector(1,0,1)
    obj.opacity = 0.25
    update_obj(obj, 'sensing_horizon')

    obj = make_ellipsoid(x, y, z, plan_r, plan_r, plan_r)
    obj.color = vector(1,0,1)
    obj.opacity = 0.125
    update_obj(obj, 'planning_radius')

def update_goal(x, y, z):
    obj = make_ellipsoid(x, y, z, 0.2, 0.2, 0.2)
    obj.color = vector(0,0,1)
    update_obj(obj, 'goal')