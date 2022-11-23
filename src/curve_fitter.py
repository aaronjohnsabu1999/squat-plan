import numpy as np
import cvxpy as cp

class Trajectory3D:
    def __init__(self, traj_x, traj_y, traj_z):
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.traj_z = traj_z

    def input(self, t):
        return np.array([self.traj_x.input(t), self.traj_y.input(t), self.traj_z.input(t)])

    def state(self, t):
        return np.array([self.traj_x.state(t), self.traj_y.state(t), self.traj_z.state(t)]).T


class Trajectory:
    def __init__(self, p0, v0, a0, j0, s, dts):
        self.p0 = p0
        self.v0 = v0
        self.a0 = a0
        self.j0 = j0
        self.s = s
        self.dts = dts
        self.N = len(s)

        self.states = np.zeros((self.N + 1, 4))
        self.states[0] = np.array([p0, v0, a0, j0])

        for i in range(1, self.N + 1):
            p_prev, v_prev, a_prev, j_prev = self.states[i - 1]
            s = self.s[i - 1]

            j = j_prev + s*dts
            a = a_prev + j_prev*dts + (1/2)*s*dts**2
            v = v_prev + a_prev*dts + (1/2)*j_prev*dts**2 + (1/6)*s*dts**3
            p = p_prev + v_prev*dts + (1/2)*a_prev*dts**2 + (1/6)*j_prev*dts**3 + (1/24)*s*dts**4

            self.states[i] = np.array([p, v, a, j])

    def input(self, t):
        i = int(np.floor(t / self.dts))

        s = 0
        if i < self.N:
            s = self.s[i]

        return s

    def state(self, t):
        start_i = int(np.floor(t / self.dts))
        start_t = start_i * self.dts
        dt = t - start_t

        p_init, v_init, a_init, j_init = self.states[start_i]

        s = 0
        if start_i < self.N:
            s = self.s[start_i]

        j = j_init + s*dt
        a = a_init + j_init*dt + (1/2)*s*dt**2
        v = v_init + a_init*dt + (1/2)*j_init*dt**2 + (1/6)*s*dt**3
        p = p_init + v_init*dt + (1/2)*a_init*dt**2 + (1/6)*j_init*dt**3 + (1/24)*s*dt**4

        return p, v, a, j


def fit_snap_input(p, v0, a0, j0, vN, aN, jN, dt, max_snap):
    N = p.shape[0] - 1

    def s_coeffs(i):
        c = np.zeros(N + 1)
        c[i] = 1
        return c

    def j_coeffs(i):
        c = np.zeros(N+1)
        c += np.sum(list( (s_coeffs(j)*dt) for j in range(i)), axis=0)
        c[-1] += j0
        return c

    def a_coeffs(i):
        c = np.zeros(N+1)
        c += np.sum(list( (j_coeffs(j)*dt + (1/2)*s_coeffs(j)*dt**2) for j in range(i)), axis=0)
        c[-1] += a0
        return c

    def v_coeffs(i):
        c = np.zeros(N+1)
        c += np.sum(list( (a_coeffs(j)*dt + (1/2)*j_coeffs(j)*dt**2 + (1/6)*s_coeffs(j)*dt**3) for j in range(i)), axis=0)
        c[-1] += v0
        return c

    def p_coeffs(i):
        c = np.zeros(N+1)
        c += np.sum(list( (v_coeffs(j)*dt + (1/2)*a_coeffs(j)*dt**2 + (1/6)*j_coeffs(j)*dt**3 + (1/24)*s_coeffs(j)*dt**4) for j in range(i)), axis=0)
        c[-1] += p[0]
        return c

    # minimize ||As - b||^2 subject to Cs = d, with variable s
    s = cp.Variable(N)

    A = np.row_stack(list(np.array(list(p_coeffs(i) for i in range(N+1)))))
    b = (p - A[:,-1])
    A = A[:, :-1]

    # add snap control cost
    weight = 0.00001
    A = np.row_stack((A, weight * np.identity(N)))
    b = np.concatenate((b, np.zeros(N)))

    objective = cp.Minimize(cp.sum_squares(A/dt**4 @ s - b/dt**4)) # CVXPY doesn't like really small numbers. Scaling by 1/dt^4 seems to help.

    C = np.row_stack([p_coeffs(N), v_coeffs(N), a_coeffs(N), j_coeffs(N)])
    d = (np.array((p[N], vN, aN, jN)) - C[:,-1])
    C = C[:, :-1]
    constraints = [C/dt**4 @ s == d/dt**4, cp.abs(s) <= max_snap] # also constrain snap

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    return Trajectory(p[0], v0, a0, j0, s.value, dt)
