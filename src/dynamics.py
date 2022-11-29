import numpy as np

import config

# cross-product matrix from vector
def cross_mat(v):
    return np.cross(v, np.identity(v.shape[0]) * -1)

# quaternion class
class Quat(np.ndarray):
    @classmethod
    def new(cls, arr=[1., 0., 0., 0.]):
        return cls(4, buffer=np.array(arr))

    @classmethod
    def pair(cls, q0, qv):
        return cls.new(np.concatenate(([q0], qv)))

    @classmethod
    def pure(cls, vec):
        return cls.pair(0.0, vec)

    def __mul__(self, q):
        p = self
        r0 = p[0]*q[0] - p[1:]@q[1:]
        rv = p[0]*q[1:] + q[0]*p[1:] + np.cross(p[1:], q[1:])
        return Quat.pair(r0, rv)

    def normalize(self):
        self /= np.linalg.norm(self)

    def conj(self):
        return Quat.pair(self[0], -self[1:])

    def rotate_vec(self, vec):
        return np.array((self * Quat.pure(vec) * self.conj())[1:])

    def rot_mat(self):
        return np.array([
            self.rotate_vec([1.0, 0.0, 0.0]),
            self.rotate_vec([0.0, 1.0, 0.0]),
            self.rotate_vec([0.0, 0.0, 1.0])
        ]).T


# constants
i_z = np.array([0.0, 0.0, 1.0])

g = config.g
m = config.DRONE_MASS
J = config.DRONE_INERTIA
J_inv = np.linalg.inv(J)

dt = config.DYNAMICS_DT

# dynamics
def f(p, v, q, omega, M_B, F_z):
    p_dot = v
    v_dot = (F_z * q.rotate_vec(i_z) + np.array([0, 0, -m*g])) / m
    q_dot = 1/2 * q * Quat.pure(omega)
    omega_dot = J_inv @ (np.cross(-omega, J@omega) + M_B)
    return p_dot, v_dot, q_dot, omega_dot

# plant
class Plant:
    def __init__(self, p0, v0=np.zeros(3), q0=Quat.new(), omega0=np.zeros(3)):
        self.p = p0
        self.v = v0
        self.q = q0
        self.omega = omega0

    def step(self, M_B, F_z):
        p_dot, v_dot, q_dot, omega_dot = f(self.p, self.v, self.q, self.omega, M_B, F_z)
        self.p += p_dot * dt + (1/2) * v_dot * dt**2 # add v_dot term for slightly better integration
        self.v += v_dot * dt
        self.q += dt * q_dot
        self.q.normalize()
        self.omega += omega_dot * dt

# controller
class Controller:
    def __init__(self, p0, v0=np.zeros(3), q0=Quat.new(), omega0=np.zeros(3)):
        self.plant = Plant(p0, v0, q0, omega0)

        self.tau = g # specific thrust (F_z / m)
        self.tau_dot = 0

        self.S_prev = None # used to approximate S_dot (see https://arxiv.org/pdf/1809.04048.pdf)

    def step(self, p_ref, v_ref, a_ref, j_ref, s_ref): # TODO turn into actual closed-loop controller
        # Get omega_dot and tau_ddot from snap (see https://arxiv.org/pdf/1809.04048.pdf)
        R = self.plant.q.rot_mat()
        b_x = R[:,0]
        b_y = R[:,1]
        b_z = R[:,2]
        yaw_vec = np.array([b_x[0], b_x[1], 0])
        S = 1 / (yaw_vec.T @ yaw_vec) * np.array([[-b_x[1], b_x[0]]]) @ np.array([[0, -b_z[0], b_y[0]], [0, -b_z[1], b_y[1]]])
        S_dot = np.zeros((1, 3))
        if self.S_prev is not None:
            S_dot = (S - self.S_prev) / dt
        self.S_prev = S
        yaw_ref_ddot = 0 # don't care about yaw

        mat = np.linalg.inv(np.row_stack((np.column_stack((self.tau * R @ cross_mat(i_z).T, b_z)),
                                          np.column_stack((S, 0)))))
        vec = np.concatenate((s_ref - R @ (2*self.tau_dot*np.identity(3) + self.tau*cross_mat(self.plant.omega)) @ cross_mat(i_z).T @ self.plant.omega, yaw_ref_ddot - S_dot @ self.plant.omega))
        omega_dot_tau_ddot = mat @ vec
        omega_dot = omega_dot_tau_ddot[:3]
        tau_ddot = omega_dot_tau_ddot[3]

        # TODO add torque and force limits, or to make more realistic, compute motor speeds and set motor speed limits 
        M_B = J @ omega_dot - np.cross(-self.plant.omega, J@self.plant.omega)
        F_z = m * self.tau

        self.plant.step(M_B, F_z)

        self.tau += self.tau_dot * dt + (1/2) * tau_ddot * dt**2 # add tau_ddot term for slightly better integration
        self.tau_dot += tau_ddot * dt
