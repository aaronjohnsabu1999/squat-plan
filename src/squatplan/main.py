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

from squatplan.trajopt import TrajOpt
from squatplan.plotter import Plotter
from squatplan.obstacle import gen_random_forest


def main():
    # Solver configuration
    solver_params = {
        "num_steps": 200,
        "total_time": 10.0,
        "max_iter": 1000,
        "set_QF": False,
        "set_WF": False,
    }

    # System model
    system_params = {
        "M": 10,
        "J": np.diag([0.1, 0.1, 0.1]),
    }

    # Boundary conditions
    boundary_conditions = {
        "P_I": [0.0, 0.0, 0.0],
        "P_F": [10.0, 10.0, 10.0],
        "V_I": [0, 0, 0],
        "V_F": [0, 0, 0],
        "Q_I": [1, 0, 0, 0],
        "Q_F": [1, 0, 0, 0],
        "W_I": [0, 0, 0],
        "W_F": [0, 0, 0],
    }

    # Control weights
    control_params = {
        "K_f": 0.02,
        "K_m": 10.0,
        "K_p": 0.0,
        "K_v": 0.0,
    }

    # Generate obstacles
    obstacles = gen_random_forest(
        num_obstacles=16,
        boundary_points=[boundary_conditions["P_I"], boundary_conditions["P_F"]],
        min_radius=0.5,
        max_radius=2.0,
        seed=0,
    )

    # Run trajectory optimization
    trajopt = TrajOpt(
        solver_params, system_params, boundary_conditions, control_params, obstacles
    )
    time, force, moments, pos, vel, ang_vel, quat = trajopt.run()

    # Print diagnostics
    print("=== SQuAT Plan Config ===")
    print(f"Solver Params: {solver_params}")
    print(f"System Params: {system_params}")
    print(f"Boundary Conditions: {boundary_conditions}")
    print(f"Control Params: {control_params}")
    print(f"Obstacles ({len(obstacles)}): {[str(ob) for ob in obstacles]}")

    # Plot results
    plotter = Plotter()
    plotter.set_obstacles(obstacles)
    plotter.set_axes(["x", "y", "z"])
    plotter.set_labels(
        ["Position", "Velocity", "Quaternion", "Angular Velocity", "Thrust", "Moment"]
    )
    plotter.set_title("SQuAT Trajectory Optimization Results")
    plotter.set_legend(
        [
            "p_x",
            "p_y",
            "p_z",
            "v_x",
            "v_y",
            "v_z",
            "q_0",
            "q_1",
            "q_2",
            "q_3",
            "w_x",
            "w_y",
            "w_z",
            "F_B",
            "M_Bx",
            "M_By",
            "M_Bz",
        ]
    )
    plotter.set_time(time)
    plotter.set_data(pos, vel, quat, ang_vel, force, moments)
    plotter.plot()
