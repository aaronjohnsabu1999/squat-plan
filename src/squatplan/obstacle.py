# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * Department of Mechanical and Aerospace Engineering      *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Ryan Nemiroff, Brett T. Lopez *
# * Contact: {aaronjs, ryguyn, btlopez}@ucla.edu            *
# *                                                         *
# ***********************************************************/

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Union

ObstacleType = Literal["sphere", "cylinder"]


@dataclass
class Obstacle:
    type: ObstacleType
    params: dict[str, float]

    def function(self, x: float, y: float, z: float) -> float:
        if self.type == "sphere":
            dx = x - self.params["x"]
            dy = y - self.params["y"]
            dz = z - self.params["z"]
            r2 = self.params["r"] ** 2
            return dx**2 + dy**2 + dz**2 - r2
        elif self.type == "cylinder":
            dx = x - self.params["x"]
            dy = y - self.params["y"]
            r2 = self.params["r"] ** 2
            return dx**2 + dy**2 - r2
        else:
            raise ValueError(f"Unsupported obstacle type: {self.type}")

    def __str__(self) -> str:
        return f"{self.type.capitalize()} @ ({self.params['x']:.2f}, {self.params['y']:.2f}, {self.params['z']:.2f}), r={self.params['r']:.2f}"


def gen_random_forest(
    num_obstacles: int,
    boundary_points: list[list[float]],
    max_radius: float = 4.0,
    min_radius: float = 0.5,
    seed: int = 0,
) -> list[Obstacle]:
    np.random.seed(seed)
    P_I, P_F = boundary_points
    obstacles: list[Obstacle] = []

    while len(obstacles) < num_obstacles:
        remaining = num_obstacles - len(obstacles)
        new_obstacles = []
        for _ in range(remaining):
            x = np.random.uniform(P_I[0], P_F[0])
            y = np.random.uniform(P_I[1], P_F[1])
            z = np.random.uniform(P_I[2], P_F[2])
            r = np.random.uniform(min_radius, max_radius)
            new_obstacles.append(Obstacle("sphere", {"x": x, "y": y, "z": z, "r": r}))

        for obstacle in new_obstacles:
            if all(obstacle.function(*pt) >= 0 for pt in boundary_points):
                obstacles.append(obstacle)

    return obstacles
