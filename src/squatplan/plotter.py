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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Literal, Sequence


class Plotter:
    def __init__(self):
        self.obstacles = []
        self.axes = ["x", "y", "z"]
        self.labels = [
            "Position",
            "Velocity",
            "Quaternion",
            "Angular Velocity",
            "Thrust",
            "Moment",
        ]
        self.title = "SQuAT Trajectory Optimization Results"
        self.legend = []
        self.time: list[float] = []
        self.data = {}

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_axes(self, axes: list[str]):
        self.axes = axes

    def set_labels(self, labels: list[str]):
        self.labels = labels

    def set_title(self, title: str):
        self.title = title

    def set_legend(self, legend: list[str]):
        self.legend = legend

    def set_time(self, time: list[float]):
        self.time = time

    def set_data(
        self,
        pos: list[list[float]],
        vel: list[list[float]],
        quat: list[list[float]],
        ang_vel: list[list[float]],
        force: list[list[float]],
        ang_mom: list[list[float]],
    ):
        self.data = {
            "pos": pos,
            "vel": vel,
            "quat": quat,
            "ang_vel": ang_vel,
            "force": force,
            "ang_mom": ang_mom,
        }

    def _plot_surface(self, ax, x, y, z, r, type: Literal["sphere", "cylinder"]):
        u, v = (
            np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
            if type == "sphere"
            else np.mgrid[0 : 2 * np.pi : 30j, 0:10:30j]
        )
        if type == "sphere":
            X = x + r * np.cos(u) * np.sin(v)
            Y = y + r * np.sin(u) * np.sin(v)
            Z = z + r * np.cos(v)
        else:
            X = x + r * np.cos(u)
            Y = y + r * np.sin(u)
            Z = z + v
        ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

    def _plot_3d_path(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot3D(*self.data["pos"], "red")
        for obs in self.obstacles:
            self._plot_surface(
                ax,
                obs.params["x"],
                obs.params["y"],
                obs.params["z"],
                obs.params["r"],
                obs.type,
            )

    def _plot_time_series(self):
        fig, axs = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        time = self.time
        labels = self.labels
        ax_labels = self.axes

        series_to_plot = [
          ("pos", 3, lambda i: f"p_{ax_labels[i]}"),
          ("vel", 3, lambda i: f"v_{ax_labels[i]}"),
          ("quat", 4, lambda i: f"q_{i}", True),
          ("ang_vel", 3, lambda i: f"w_{ax_labels[i]}"),
          ("force", 1, lambda _: "F_B", False, {"drawstyle": "steps-post"}),
          ("ang_mom", 3, lambda i: f"M_B{ax_labels[i]}"),
        ]
        
        for idx, (key, dim, label_fn, *extras) in enumerate(series_to_plot):
            ax = axs[idx]
            plot_norm = extras[0] if extras else False
            plot_kwargs = extras[1] if len(extras) > 1 else {}

            for i in range(dim):
                ax.plot(time, self.data[key][i], label=label_fn(i), **plot_kwargs)

            if plot_norm:
                norm = [np.linalg.norm([self.data[key][j][t] for j in range(dim)]) for t in range(len(time))]
                ax.plot(time, norm, "-.", label="||q||")

            ax.set_ylabel(labels[idx])
            ax.legend()
            ax.grid()

        axs[-1].set_xlabel("Time")
        fig.suptitle(self.title)
        plt.tight_layout()
        plt.show()


    def plot(self):
        self._plot_3d_path()
        self._plot_time_series()
