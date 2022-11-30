import numpy as np

import config

# sign function that only outputs -1 or 1
def sgn(x):
    if x > 0:
        return 1
    else:
        return -1

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

    def vec(self):
        return np.array(self[1:])

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
g_vec = np.array([0, 0, -g])
m = config.DRONE_MASS
J = config.DRONE_INERTIA
J_inv = np.linalg.inv(J)

Kp_att = config.KP_ATT
Kd_att = config.KD_ATT

Kp_pos = config.KP_POS
Kd_pos = config.KD_POS

dt = config.DYNAMICS_DT

# dynamics
def f(p, v, q, omega, M_B, T):
    p_dot = v
    v_dot = q.rotate_vec(T * i_z) / m + g_vec
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

    def step(self, M_B, T):
        p_dot, v_dot, q_dot, omega_dot = f(self.p, self.v, self.q, self.omega, M_B, T)
        self.p += p_dot * dt + (1/2) * v_dot * dt**2 # add v_dot term for slightly better integration
        self.v += v_dot * dt
        self.q += dt * q_dot
        self.q.normalize()
        self.omega += omega_dot * dt

# controller
class Controller:
    def __init__(self, p0, v0=np.zeros(3), q0=Quat.new(), omega0=np.zeros(3)):
        self.plant = Plant(p0, v0, q0, omega0)
        self.S_prev = None # used to approximate S_dot (see https://arxiv.org/pdf/1809.04048.pdf)        
        self.tau_prev = None # used to approximate tau_dot

    # Get omega_ref and omega_dot_ref from jerk and snap (see https://arxiv.org/pdf/1809.04048.pdf)
    def get_omega_ref(self, j_ref, s_ref, tau):
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

        tau_dot = 0
        if self.tau_prev is not None:
            tau_dot = (tau - self.tau_prev) / dt
        self.tau_prev = tau

        mat = np.linalg.inv(np.row_stack((np.column_stack((tau * R @ cross_mat(i_z).T, b_z)),
                                          np.column_stack((S, 0)))))

        yaw_ref_dot = 0 # don't care about yaw
        jerk_vec = np.concatenate((j_ref, [yaw_ref_dot]))

        yaw_ref_ddot = 0 # don't care about yaw
        snap_vec = np.concatenate((s_ref - R @ (2*tau_dot*np.identity(3) + tau*cross_mat(self.plant.omega)) @ cross_mat(i_z).T @ self.plant.omega, yaw_ref_ddot - S_dot @ self.plant.omega))

        omega_ref = (mat @ jerk_vec)[:3]
        omega_dot_ref = (mat @ snap_vec)[:3]

        return omega_ref, omega_dot_ref

    def step(self, p_ref, v_ref, a_or_q_ref, j_or_w_ref, input_ref):
        if config.MPC_USE_LINEAR_MODEL:
            a_ref = a_or_q_ref
            j_ref = j_or_w_ref
            s_ref = input_ref
        else:
            q_ref = Quat.new(a_or_q_ref)
            omega_ref = j_or_w_ref
            M_B_ref, T_ref = input_ref
            omega_dot_ref = J_inv @ M_B_ref

        # step 1: compute specific thrust using feedforward (a_ref) and PD control (Lecture 16)
        if not config.MPC_USE_LINEAR_MODEL:
            a_ref = q_ref.rotate_vec(T_ref * i_z) / m + g_vec
        r_e = self.plant.p - p_ref
        r_e_dot = self.plant.v - v_ref
        tau_vec = a_ref - g_vec - Kp_pos * r_e - Kd_pos * r_e_dot
        tau = np.linalg.norm(tau_vec)
        tau_vec_hat = tau_vec / tau

        # limit thrust
        tau = np.clip(tau, config.MIN_THRUST / m, config.MAX_THRUST / m)

        if config.MPC_USE_LINEAR_MODEL:
            # step 2: compute reference angular velocity and angular acceleration, utilizing differential flatness
            omega_ref, omega_dot_ref = self.get_omega_ref(j_ref, s_ref, tau)

        # step 3: compute desired attitude (from Lecture 16)
        T_hat = i_z
        v_hat = tau_vec_hat
        q_d = 1 / np.sqrt(2*(1 + T_hat@v_hat)) * Quat.pair(1 + T_hat@v_hat, np.cross(T_hat, v_hat))

        # step 4: compute attitude and angular velocity error (from Lecture 15)
        q_e = q_d.conj() * self.plant.q
        omega_e = self.plant.omega - q_e.conj().rotate_vec(omega_ref)

        # step 5: compute omega_dot using feedforward (omega_dot_ref) and PD control (Lecture 15)
        omega_dot = omega_dot_ref - Kp_att * sgn(q_e[0]) * q_e.vec() - Kd_att * omega_e

        # step 6: compute M_B and T from omega_dot and tau
        M_B = J @ omega_dot # ignore the np.cross(-omega, J@omega) term for simplicity
        T = m * tau

        # limit torque
        M_B = M_B.clip(-config.MAX_TORQUE, config.MAX_TORQUE)

        self.plant.step(M_B, T)

        return M_B, T
