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
import matplotlib.pyplot as plt
import time
from threading import Thread
import signal
import os
from trajopt import Problem
from traj_eval import TrajectoryPolynomial, TrajectoryQuadrotor
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

    ctrl = Controller(config.INIT_POS)

    if config.MPC_USE_LINEAR_MODEL:
        ref_state = np.array([config.INIT_POS, np.zeros(3), np.zeros(3), np.zeros(3)])
        traj = TrajectoryPolynomial(*list([rs, rs] for rs in ref_state), [np.zeros(3)], config.MPC_TIME_HORIZON) # zero-input initial trajectory
    else:
        ref_state = [config.INIT_POS, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)]
        traj = TrajectoryQuadrotor(*list([rs, rs] for rs in ref_state), [np.zeros(3)], [config.DRONE_MASS*config.g], config.MPC_TIME_HORIZON)

    new_traj = traj
    reached_traj_end = False
    reached_new_traj_start = False
    optimization_failed = False

    # for visualization
    gekko_path = [[], [], []]
    smooth_path = [[], [], []]

    # for plotting
    p_ref_data = []
    p_data = []
    v_data = []
    q_data = []
    omega_data = []
    M_B_data = []
    T_data = []

    t = 0
    t_offset = 0
    new_t_offset = t_offset

    t_start = time.time()
    target_time = t_start

    def run_trajopt():
        global new_traj
        global new_t_offset
        global optimization_failed

        global gekko_path
        global smooth_path

        t_opt_start = time.time()

        t_init = t + config.MPC_MAX_SOLVE_TIME
        traj_t_init = t_init - t_offset
        init_from_stop = False
        if traj_t_init >= config.MPC_TIME_HORIZON:
            traj_t_init = config.MPC_TIME_HORIZON
            init_from_stop = True
        state_init = traj.state(traj_t_init)

        prob = Problem(*state_init, time_limit=(not init_from_stop))
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
            optimization_failed = True
            print("Optimization failed or timed out!")

        print("Real solve time: {}, Sim solve time: {}".format(time.time() - t_opt_start, t - (t_init - config.MPC_MAX_SOLVE_TIME)))

    trajopt_thread = Thread(target=run_trajopt)
    trajopt_thread.start()

    while True:
        find_new_traj = False

        if not trajopt_thread.is_alive() and optimization_failed:
            optimization_failed = False
            find_new_traj = True
        elif new_t_offset > t_offset and t >= new_t_offset: # if a new trajectory is available and we've reached the start of it
            if (not trajopt_thread.is_alive()):
                traj = new_traj
                t_offset = new_t_offset
                reached_traj_end = False
                reached_new_traj_start = False
                optimization_failed = False
                find_new_traj = True

                viz.update_gekko_path(*gekko_path)
                viz.update_smooth_path(*smooth_path)
            elif not reached_new_traj_start:
                reached_new_traj_start = True
                print("WARNING: MPC solve took too long. Trajectory will not be continuous! (This should almost never happen)")

        if t - t_offset > config.MPC_TIME_HORIZON and not reached_traj_end:
            print("INFO: MPC solve took too long or failed, and the end of the trajectory has been reached.")
            reached_traj_end = True

        traj_t = min(t - t_offset, config.MPC_TIME_HORIZON)
        ref_state = traj.state(traj_t)

        if find_new_traj:
            trajopt_thread = Thread(target=run_trajopt)
            trajopt_thread.start()

        M_B, T = ctrl.step(*ref_state, traj.input(traj_t))

        p_ref_data.append(ref_state[0])
        p_data.append(ctrl.plant.p.copy())
        v_data.append(ctrl.plant.v.copy())
        q_data.append(ctrl.plant.q.copy())
        omega_data.append(ctrl.plant.omega.copy())
        M_B_data.append(M_B)
        T_data.append(T)

        viz.update_vehicle(ctrl.plant.p, ctrl.plant.q, config.COLLISION_RADIUS, config.SENSING_HORIZON, config.SENSING_HORIZON_CONSERVATIVE)
        viz.show_once()

        # if np.linalg.norm(ctrl.plant.p - config.GOAL_POS) < 0.1:
        #     print("Goal reached!")
        #     break

        if t >= 20 - config.DYNAMICS_DT/2:
            break

        t += config.DYNAMICS_DT

        # Best effort to keep up with desired simulation rate. If simulation is too slow, it will just run in slow motion.
        target_time += config.DYNAMICS_DT / config.SIM_SPEED_FACTOR
        delay = target_time - time.time()
        if (delay < 0):
            print("WARNING: Can't simulate dynamics fast enough! (behind by {} seconds)".format(-delay))
            target_time = time.time()
        else:
            time.sleep(delay)

    # Plot
    t_data = np.arange(0, len(p_data)) * config.DYNAMICS_DT

    p_data = np.array(p_data).T
    v_data = np.array(v_data).T
    q_data = list(np.linalg.norm(q.vec()) for q in q_data)
    omega_data = np.array(omega_data).T
    M_B_data = np.array(M_B_data).T

    all_xlabels = False
    fig, axs = plt.subplots(6, sharex=(not all_xlabels))
    fig.tight_layout()
    legend_loc = 'upper right'

    axs[0].plot(t_data, p_data[0], label="$p_x$")
    axs[0].plot(t_data, p_data[1], label="$p_y$")
    axs[0].plot(t_data, p_data[2], label="$p_z$")
    axs[0].set(xlabel=("Time [s]" if all_xlabels else None), ylabel="Position [m]")
    axs[0].legend(loc=legend_loc)

    axs[1].plot(t_data, v_data[0], label="$v_x$")
    axs[1].plot(t_data, v_data[1], label="$v_y$")
    axs[1].plot(t_data, v_data[2], label="$v_z$")
    axs[1].set(xlabel=("Time [s]" if all_xlabels else None), ylabel="Velocity [m/s]")
    axs[1].legend(loc=legend_loc)

    axs[2].plot(t_data, q_data, label="$||\\vec{{q}}||$")
    axs[2].set(xlabel=("Time [s]" if all_xlabels else None), ylabel="$||\\vec{{q}}||$")
    axs[2].legend(loc=legend_loc)

    axs[3].plot(t_data, omega_data[0], label="$\\omega_x$")
    axs[3].plot(t_data, omega_data[1], label="$\\omega_y$")
    axs[3].plot(t_data, omega_data[2], label="$\\omega_z$")
    axs[3].set(xlabel=("Time [s]" if all_xlabels else None), ylabel="Angular vel [rad/s]")
    axs[3].legend(loc=legend_loc)

    axs[4].plot(t_data, M_B_data[0], label="$M_{B,x}$")
    axs[4].plot(t_data, M_B_data[1], label="$M_{B,y}$")
    axs[4].plot(t_data, M_B_data[2], label="$M_{B,z}$")
    axs[4].set(xlabel=("Time [s]" if all_xlabels else None), ylabel="Torque inp [N m]")
    axs[4].axhline(y=-config.MAX_TORQUE, color='black', linestyle='--')
    axs[4].axhline(y=config.MAX_TORQUE, color='black', linestyle='--')
    axs[4].legend(loc=legend_loc)

    axs[5].plot(t_data, T_data, label="$T$")
    axs[5].set(xlabel="Time [s]", ylabel="Thrust inp [N]")
    axs[5].axhline(y=config.MIN_THRUST, color='black', linestyle='--')
    axs[5].axhline(y=config.MAX_THRUST, color='black', linestyle='--')
    axs[5].legend(loc=legend_loc)

    plt.figure()
    r_e = list(np.linalg.norm(p_data[:,i] - p_ref_data[i]) for i in range(len(p_ref_data)))
    plt.plot(t_data, r_e, label="$||\\vec{{r}}_e||$")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    axs[4].legend(loc=legend_loc)

    plt.show()
