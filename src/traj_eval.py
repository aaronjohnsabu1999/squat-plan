import numpy as np
import cvxpy as cp

import config

class TrajectoryPolynomial:
    def __init__(self, p, v, a, j, s, dts):
        self.p = p
        self.v = v
        self.a = a
        self.j = j
        self.s = s
        self.dts = dts
        self.N = len(s)

    def input(self, t):
        i = int(np.floor(t / self.dts))

        s = np.zeros(3)
        if i < self.N:
            s = self.s[i]

        return s

    def state(self, t):
        i = int(np.floor(t / self.dts))
        start_t = i * self.dts
        dt = t - start_t

        p = self.p[self.N]
        v = self.v[self.N]
        a = self.a[self.N]
        j = self.j[self.N]
        if i < self.N:
            # linear interpolation for each variable
            p = self.p[i] + (self.p[i+1] - self.p[i]) * (dt / self.dts)
            v = self.v[i] + (self.v[i+1] - self.v[i]) * (dt / self.dts)
            a = self.a[i] + (self.a[i+1] - self.a[i]) * (dt / self.dts)
            j = self.j[i] + (self.j[i+1] - self.j[i]) * (dt / self.dts)

        return p, v, a, j

class TrajectoryQuadrotor:
    def __init__(self, p, v, q, w, M_B, T, dts):
        self.p = p
        self.v = v
        self.q = q
        self.w = w
        self.M_B = M_B
        self.T = T
        self.dts = dts
        self.N = len(T)

    def input(self, t):
        i = int(np.floor(t / self.dts))

        M_B = np.zeros(3)
        T = config.DRONE_MASS * config.g
        if i < self.N:
            M_B = self.M_B[i]
            T = self.T[i]

        return M_B, T

    def state(self, t):
        i = int(np.floor(t / self.dts))
        start_t = i * self.dts
        dt = t - start_t

        p = self.p[self.N]
        v = self.v[self.N]
        q = self.q[self.N]
        w = self.w[self.N]
        if i < self.N:
            # linear interpolation for each variable
            p = self.p[i] + (self.p[i+1] - self.p[i]) * (dt / self.dts)
            v = self.v[i] + (self.v[i+1] - self.v[i]) * (dt / self.dts)
            q = self.q[i] + (self.q[i+1] - self.q[i]) * (dt / self.dts)
            q = np.array(q) / np.linalg.norm(q)
            w = self.w[i] + (self.w[i+1] - self.w[i]) * (dt / self.dts)

        return p, v, q, w
