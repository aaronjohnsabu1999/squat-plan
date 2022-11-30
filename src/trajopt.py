from gekko import GEKKO
import numpy as np
from traj_eval import TrajectoryPolynomial, TrajectoryQuadrotor
from quaternion import quat_mult, quat_conj
import os

import config

M   = config.DRONE_MASS
J   = config.DRONE_INERTIA
F_I = [0, 0, -config.DRONE_MASS * config.g]

class Problem:
    def __init__(self, p_init, v_init, a_or_q_init, j_or_w_init, time_limit=True):
        self.p_init = p_init

        solver = 1 # 1 is APOPT, 3 is IPOPT
        self.m = GEKKO(remote=(True if solver == 3 and os.name != 'nt' else False)) # run locally when possible (need remote for IPOPT on non-Windows platforms)
        self.m.options.IMODE = 6 # dynamic control, "simultaneous" approach (see https://gekko.readthedocs.io/en/latest/imode.html and https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/DiehlFerreauHaverbeke_mpc-overview.pdf)
        self.m.options.SOLVER = solver
        self.m.options.MAX_ITER = 1000
        if time_limit:
            self.m.options.MAX_TIME = config.MPC_MAX_SOLVE_TIME / config.SIM_SPEED_FACTOR

        # Choose max and abs functions (can be max2/abs2 or max3/abs3)
        # Only APOPT can use max3/abs3 because they require mixed integer programming, but max2/abs2 seem to yield more optimal solutions anyway.
        self.m_max = self.m.max2
        self.m_abs = self.m.abs2

        self.m.time = np.linspace(0, config.MPC_TIME_HORIZON, config.MPC_NUM_TIME_STEPS + 1)

        # vector to weight final state but not other states
        final_np = np.zeros(config.MPC_NUM_TIME_STEPS + 1)
        final_np[-1] = 1
        self.final = self.m.Param(final_np)

        if config.MPC_USE_LINEAR_MODEL:
            a_init = a_or_q_init
            j_init = j_or_w_init

            # variables
            self.s = [] # snap (control input)
            self.j = [] # jerk
            self.a = [] # acceleration
            self.v = [] # velocity
            self.p = [] # position

            # Repeat 3 times for X, Y, Z
            for d in range(3):
                # initialize variables
                self.s.append(self.m.Var(value=0, lb=-config.MAX_SNAP, ub=config.MAX_SNAP))
                self.j.append(self.m.Var(value=j_init[d]))
                self.a.append(self.m.Var(value=a_init[d]))
                self.v.append(self.m.Var(value=v_init[d]))
                self.p.append(self.m.Var(value=p_init[d]))

                # fix final velocity and derivatives to 0 (so that trajectory ends in a complete stop)
                self.m.fix_final(self.v[d], 0)

                if solver != 1: # fix_final doesn't work here for APOPT for some reason
                    self.m.fix_final(self.a[d], 0)

                self.m.fix_final(self.j[d], 0)

                # dynamics
                self.m.Equations((
                    self.p[d].dt() == self.v[d],
                    self.v[d].dt() == self.a[d],
                    self.a[d].dt() == self.j[d],
                    self.j[d].dt() == self.s[d]
                ))

            if solver == 1: # alternative to fix_final for APOPT
                self.m.Equation((self.a[0]**2 + self.a[1]**2 + self.a[2]**2)*self.final == 0)

            # Add control cost to objective (note: the cost is scaled by dt so it matches an integral over time)
            dt = config.MPC_TIME_HORIZON / config.MPC_NUM_TIME_STEPS
            self.m.Minimize((self.s[0]**2 + self.s[1]**2 + self.s[2]**2) * dt)

        else: # nonlinear model
            q_init = a_or_q_init
            w_init = j_or_w_init

            p = [] # linear position
            v = [] # linear velocity
            q = [] # angular position
            w = [] # angular velocity

            m_B = [] # body moments
            f_B = self.m.Var(value=0, lb=config.MIN_THRUST, ub=config.MAX_THRUST, fixed_initial=False) # body forces

            q.append(self.m.Var(value=q_init[0]))

            for d in range(3):
                m_B.append(self.m.Var(value=0, lb=-config.MAX_TORQUE, ub=config.MAX_TORQUE))
                w.append(self.m.Var(value=w_init[d]))
                q.append(self.m.Var(value=q_init[d+1]))
                v.append(self.m.Var(value=v_init[d]))
                p.append(self.m.Var(value=p_init[d]))

                self.m.fix_final(v[d], 0)
                self.m.fix_final(q[d+1], 0)
                self.m.fix_final(w[d], 0)

            # dynamics
            eqs = []
            for i in range(3):
                # dynamics
                eqs.append(p[i].dt()   == v[i])
                eqs.append(M*v[i].dt() == F_I[i] + quat_mult(quat_mult(q, [0, 0, 0, f_B]), quat_conj(q))[i+1])
                eqs.append(q[i+1].dt() == 0.5 * quat_mult(q, [0, w[0], w[1], w[2]])[i+1])
                eqs.append(np.sum([J[i][j] * w[j].dt() for j in range(len(w))]) == m_B[i]
                                                                                 - np.cross(w,
                                                                                            np.matmul(J,
                                                                                                      w))[i])
            eqs.append(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 == 1)
            self.m.Equations(eqs)

            # minimize control input
            K_f = 0.01
            K_m = 20
            dt = config.MPC_TIME_HORIZON / config.MPC_NUM_TIME_STEPS
            self.m.Minimize((K_f * f_B**2 + K_m * (m_B[0]**2 + m_B[1]**2 + m_B[2]**2)) * dt)

            self.p = p
            self.v = v
            self.q = q
            self.w = w
            self.m_B = m_B
            self.f_B = f_B


    def add_ellipsoid_obstacle(self, x, y, z, rx, ry, rz):
        rx += config.COLLISION_RADIUS
        ry += config.COLLISION_RADIUS
        rz += config.COLLISION_RADIUS
        self.m.Equation(((self.p[0]-x)/rx)**2 + ((self.p[1]-y)/ry)**2 + ((self.p[2]-z)/rz)**2 >= 1)

    def add_box_obstacle(self, x, y, z, rx, ry, rz):
        # TODO Box obstacles aren't working properly (constraints are violated). No idea why.
        rx += config.COLLISION_RADIUS
        ry += config.COLLISION_RADIUS
        rz += config.COLLISION_RADIUS
        self.m.Equation(self.m_max(self.m_max(self.m_abs(self.p[0]-x)/rx, self.m_abs(self.p[1]-y)/ry), self.m_abs(self.p[2]-z)/rz) >= 1)

    def add_cylinder_obstacle(self, axis, x_or_y, y_or_z, r):
        r += config.COLLISION_RADIUS
        axes = [0, 1, 2]
        axes.remove(axis)
        self.m.Equation((self.p[axes[0]]-x_or_y)**2 + (self.p[axes[1]]-y_or_z)**2 >= r**2)

    def add_obstacles(self, obstacles):
        for obs in obstacles:
            if obs.type == 'ellipsoid':
                self.add_ellipsoid_obstacle(obs.x, obs.y, obs.z, obs.rx, obs.ry, obs.rz)
            elif obs.type == 'box':
                self.add_box_obstacle(obs.x, obs.y, obs.z, obs.rx, obs.ry, obs.rz)
            elif obs.type == 'cylinder':
                self.add_cylinder_obstacle(obs.axis, obs.x_or_y, obs.y_or_z, obs.r)

    def add_sensing_horizon_contraint(self, p_cur):
        self.m.Equation((self.p[0]-p_cur[0])**2 + (self.p[1]-p_cur[1])**2 + (self.p[2]-p_cur[2])**2 <= config.SENSING_HORIZON_CONSERVATIVE**2)
        # self.p_cur = p_cur

    def add_final_state_objective(self, p_d):
        # Experimental: Force exploration by constraining the ending position to be on the sensing horizon.
        # if np.linalg.norm(p_d - self.p_cur) > config.SENSING_HORIZON_CONSERVATIVE:
        #     self.m.Equation(((self.p[0]-self.p_cur[0])**2 + (self.p[1]-self.p_cur[1])**2 + (self.p[2]-self.p_cur[2])**2) * self.final == (config.SENSING_HORIZON_CONSERVATIVE**2) * self.final)

        # minimize squared distance to goal
        normalization = 2 * np.linalg.norm(self.p_init - p_d) # keep the slope consistent at the initial position
        normalization = max(normalization, 1e-2)
        final_state_cost = ((self.p[0]-p_d[0])**2 + (self.p[1]-p_d[1])**2 + (self.p[2]-p_d[2])**2) / normalization

        if config.MPC_USE_LINEAR_MODEL:
            K = 100000 # weight of final state cost relative to control cost
        else:
            K = 100000
        self.m.Minimize(K*final_state_cost*self.final)


    def solve(self, disp=False):
        self.m.solve(disp=disp)

        p = np.array([self.p[0].value, self.p[1].value, self.p[2].value]).T
        v = np.array([self.v[0].value, self.v[1].value, self.v[2].value]).T
        dt = config.MPC_TIME_HORIZON / config.MPC_NUM_TIME_STEPS

        if config.MPC_USE_LINEAR_MODEL:
            a = np.array([self.a[0].value, self.a[1].value, self.a[2].value]).T
            j = np.array([self.j[0].value, self.j[1].value, self.j[2].value]).T
            s = (np.array([self.s[0].value, self.s[1].value, self.s[2].value]).T)[1:]

            return TrajectoryPolynomial(p, v, a, j, s, dt)

        else:
            q = np.array([self.q[0].value, self.q[1].value, self.q[2].value, self.q[3].value]).T
            w = np.array([self.w[0].value, self.w[1].value, self.w[2].value]).T
            M_B = (np.array([self.m_B[0].value, self.m_B[1].value, self.m_B[2].value]).T)[1:]
            T = self.f_B.value[1:]

            return TrajectoryQuadrotor(p, v, q, w, M_B, T, dt)


    def __del__(self):
        self.m.cleanup()
