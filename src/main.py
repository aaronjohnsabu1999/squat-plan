#!/usr/bin/env python3

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

# PACKAGE IMPORTS
from dataclasses import dataclass
import numpy as np
import time
from threading import Thread
import signal
import os
from trajopt_linear import Problem
from curve_fitter import Trajectory, Trajectory3D
import viz_vpython as viz # can import viz_rviz or viz_vpython

import config

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

# MAIN PROGRAM
if __name__ == '__main__':
    # IMPORTANT OBSTACLE BUG: A sphere or cylinder directly in front of the vehicle will cause it to get stuck. It needs to be slightly offset so that the optimization is pushed in some direction.
    obstacles = [
        ObsEllipsoid(2.51, 2, 2.5, 0.5, 1, 0.5),
        # ObsBox(2.51, 4.5, 2.5, 0.75, 0.75, 0.75), # Box obstacles not working yet :(
        ObsEllipsoid(2.49, 10, 2.5, 1, 1, 1),
        ObsCylinder(0, 7, 2.51, 1),
    ]

    viz.init()
    viz.add_obstacles(obstacles)
    viz.update_goal(config.GOAL_POS[0], config.GOAL_POS[1], config.GOAL_POS[2])

    state = np.array([config.INIT_POS, np.zeros(3), np.zeros(3), np.zeros(3)])

    dt = config.MPC_TIME_HORIZON / config.MPC_NUM_TIME_STEPS
    traj = Trajectory3D(*list(Trajectory(*state[:,d], np.zeros(config.MPC_NUM_TIME_STEPS), dt) for d in range(3))) # zero-input initial trajectory
    new_traj = traj
    reached_traj_end = False

    t_start = time.time()
    t_offset = 0
    new_t_offset = t_offset
    t = 0

    def run_trajopt():
        global new_traj
        global new_t_offset

        t_start = time.time()

        t_init = t + config.MPC_EXPECTED_SOLVE_TIME
        state_init = traj.state(min(t_init - t_offset, config.MPC_TIME_HORIZON))

        prob = Problem(*state_init)
        prob.add_obstacles(obstacles) # TODO: only add obstacles in sensing horizon
        prob.add_sensing_horizon_contraint(state[0])
        prob.add_final_state_objective(config.GOAL_POS)

        try:
            new_traj = prob.solve(disp=False)
            new_t_offset = t_init

            path_x = list(new_traj.traj_x.state(t)[0] for t in prob.m.time)
            path_y = list(new_traj.traj_y.state(t)[0] for t in prob.m.time)
            path_z = list(new_traj.traj_z.state(t)[0] for t in prob.m.time)

            viz.update_gekko_path(prob.p[0].value, prob.p[1].value, prob.p[2].value)
            viz.update_smooth_path(path_x, path_y, path_z)
        except:
            print("Optimization failed!")

        print("Total solve time: ", time.time() - t_start)

    trajopt_thread = Thread(target=run_trajopt)
    trajopt_thread.start()

    run = True
    def signal_handler(signal, frame):
        global run
        run = False

    signal.signal(signal.SIGINT, signal_handler)

    while run:
        t = time.time() - t_start

        find_new_traj = False
        if (not trajopt_thread.is_alive()) and t >= new_t_offset:
            traj = new_traj
            t_offset = new_t_offset
            reached_traj_end = False
            find_new_traj = True

        if t - t_offset > config.MPC_TIME_HORIZON and not reached_traj_end:
            print("WARNING: MPC solve took too long, and there is no more trajectory to follow!")
            reached_traj_end = True

        state = traj.state(min(t - t_offset, config.MPC_TIME_HORIZON))

        viz.update_vehicle(*state[0], config.COLLISION_RADIUS, config.SENSING_HORIZON)

        if find_new_traj:
            trajopt_thread = Thread(target=run_trajopt)
            trajopt_thread.start()

        viz.show_once()
        time.sleep(0.01)

    os._exit(0)
