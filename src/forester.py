import random
import numpy as np
from dataclasses import dataclass
from trajopt import Obstacle

# Obstacle types
@dataclass
class ObsEllipsoid:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    type: str = 'ellipsoid'

@dataclass
class ObsBox(ObsEllipsoid):
    type: str = 'box'

@dataclass
class ObsCylinder:
    axis: int
    x_or_y: float
    y_or_z: float
    r: float
    type: str = 'cylinder'


def obstacle_intersects_sphere(obs, p, r):
    obs_point = np.zeros(3)
    obs_radius = 0

    if obs.type == 'ellipsoid':
        obs_point = np.array([obs.x, obs.y, obs.z])
        obs_radius = max(obs.rx, obs.ry, obs.rz)

    elif obs.type == 'box':
        obs_point = np.array([obs.x, obs.y, obs.z])
        obs_radius = np.sqrt(obs.rx**2 + obs.ry**2 + obs.rz**2)

    elif obs.type == 'cylinder':
        # uses https://www.wikiwand.com/en/Line%E2%80%93plane_intersection
        n = np.zeros(3)
        n[obs.axis] = 1

        axes = [0, 1, 2]
        axes.remove(obs.axis)
        l0 = np.zeros(3)
        l0[axes[0]] = obs.x_or_y
        l0[axes[1]] = obs.y_or_z

        obs_point = l0 + ((p - l0) @ n) * n # closest point on cylinder axis to p
        obs_radius = obs.r

    return np.linalg.norm(obs_point - p) < obs_radius + r

def gen_random_forest(num_obs, map_wx, map_wy, map_wz, clear_pts, clear_rad, min_rad=0.5, max_rad=3.0, seed=None):
    random.seed(seed)
    
    obstacles = []
    while len(obstacles) < num_obs:
        t = random.random()

        if t < 0.5: # 50% ellipsoids, 50% cylinders (no boxes for now)
            # ellipsoid
            x = random.uniform(0, map_wx)
            y = random.uniform(0, map_wy)
            z = random.uniform(0, map_wz)
            rx = random.uniform(min_rad, max_rad)
            ry = random.uniform(min_rad, max_rad)
            rz = random.uniform(min_rad, max_rad)
            obs = ObsEllipsoid(x, y, z, rx, ry, rz)
        else:
            # cylinder
            axis = random.randint(0, 1) * 2 # 0 or 2 - no cylinders along y axis
            map_widths = [map_wx, map_wy, map_wz]
            del map_widths[axis]
            x_or_y = random.uniform(0, map_widths[0])
            y_or_z = random.uniform(0, map_widths[1])
            r = random.uniform(min_rad, max_rad)
            obs = ObsCylinder(axis, x_or_y, y_or_z, r)

        valid = True
        for clear_pt in clear_pts:
            if obstacle_intersects_sphere(obs, clear_pt, clear_rad):
                valid = False
                break
        
        if valid:
            obstacles.append(obs)

    return obstacles
