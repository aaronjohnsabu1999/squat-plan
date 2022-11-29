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
import numpy as np
import time
from threading import Thread
import signal
import os
from trajopt_linear import Problem
from curve_fitter import Trajectory1D, Trajectory3D
from forester import *
from dynamics import Controller
import viz_vpython as viz # can import viz_rviz or viz_vpython

import config

# MAIN PROGRAM
if __name__ == '__main__':
    # exit reliably on Ctrl+C
    signal.signal(signal.SIGINT, lambda signum, frame : os._exit(0))

    # IMPORTANT OBSTACLE BUG: A sphere or cylinder directly in front of the vehicle will cause it to get stuck. It needs to be slightly offset so that the optimization is pushed in some direction.
    # obstacles = [
    #     ObsEllipsoid(2.51, 2, 2.5, 0.5, 1, 0.5),
    #     # ObsBox(2.51, 4.5, 2.5, 0.75, 0.75, 0.75), # Box obstacles not working yet :(
    #     ObsEllipsoid(2.49, 10, 2.5, 1, 1, 1),
    #     ObsCylinder(0, 7, 2.51, 1),
    # ]

    # random obstacles
    obstacles = gen_random_forest(30, config.MAP_WX, config.MAP_WY, config.MAP_WZ, [config.INIT_POS, config.GOAL_POS], config.COLLISION_RADIUS, seed=3)

    viz.init()
    viz.add_obstacles(obstacles)
    viz.update_goal(config.GOAL_POS[0], config.GOAL_POS[1], config.GOAL_POS[2])

    ref_state = np.array([config.INIT_POS, np.zeros(3), np.zeros(3), np.zeros(3)])

    ctrl = Controller(config.INIT_POS)

    dt = config.MPC_TIME_HORIZON / config.MPC_NUM_TIME_STEPS
    traj = Trajectory3D(*list(Trajectory1D(*ref_state[:,d], np.zeros(config.MPC_NUM_TIME_STEPS), dt) for d in range(3))) # zero-input initial trajectory
    new_traj = traj
    reached_traj_end = False
    reached_new_traj_start = False

    # for visualization
    gekko_path = [[], [], []]
    smooth_path = [[], [], []]

    t = 0
    t_offset = 0
    new_t_offset = t_offset

    t_start = time.time()
    target_time = t_start

    def run_trajopt():
        global new_traj
        global new_t_offset

        global gekko_path
        global smooth_path

        t_opt_start = time.time()

        t_init = t + config.MPC_EXPECTED_SOLVE_TIME
        state_init = traj.state(min(t_init - t_offset, config.MPC_TIME_HORIZON))

        prob = Problem(*state_init)
        visible_obstacles = [obs for obs in obstacles if obstacle_intersects_sphere(obs, ref_state[0], config.SENSING_HORIZON)]
        prob.add_obstacles(visible_obstacles)
        prob.add_sensing_horizon_contraint(ref_state[0])
        prob.add_final_state_objective(config.GOAL_POS)

        try:
            new_traj = prob.solve(disp=False)
            new_t_offset = t_init

            gekko_path = [prob.p[0].value, prob.p[1].value, prob.p[2].value]
            smooth_path = np.array(list(new_traj.state(t)[0] for t in prob.m.time)).T
        except:
            print("Optimization failed!")

        print("Total solve time: ", time.time() - t_opt_start)

    trajopt_thread = Thread(target=run_trajopt)
    trajopt_thread.start()

    while True:
        find_new_traj = False
        if new_t_offset > t_offset and t >= new_t_offset:
            if (not trajopt_thread.is_alive()):
                traj = new_traj
                t_offset = new_t_offset
                reached_traj_end = False
                reached_new_traj_start = False
                find_new_traj = True

                viz.update_gekko_path(*gekko_path)
                viz.update_smooth_path(*smooth_path)
            elif not reached_new_traj_start:
                reached_new_traj_start = True
                print("WARNING: MPC solve took longer than expected. Trajectory will not be continuous!")

        if t - t_offset > config.MPC_TIME_HORIZON and not reached_traj_end:
            print("WARNING: MPC solve took too long, and there is no more trajectory to follow!")
            reached_traj_end = True

        traj_t = min(t - t_offset, config.MPC_TIME_HORIZON)
        ref_state = traj.state(traj_t)

        if find_new_traj:
            trajopt_thread = Thread(target=run_trajopt)
            trajopt_thread.start()

        ctrl.step(*ref_state, traj.input(traj_t))

        viz.update_vehicle(ctrl.plant.p, ctrl.plant.q, config.COLLISION_RADIUS, config.SENSING_HORIZON, config.SENSING_HORIZON_CONSERVATIVE)
        viz.show_once()

        t += config.DYNAMICS_DT

        # Best effort to keep up with desired simulation rate. If simulation is too slow, it will just run in slow motion.
        target_time += config.DYNAMICS_DT / config.SIM_SPEED_FACTOR
        delay = target_time - time.time()
        if (delay < 0):
            print("WARNING: Can't simulate dynamics fast enough! (behind by {} seconds)".format(-delay))
            target_time = time.time()
        else:
            time.sleep(delay)
