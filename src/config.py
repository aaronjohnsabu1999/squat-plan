import numpy as np

# Map dimensions, using ENU (East North Up) coordinate frame
MAP_WX = 10
MAP_WY = 40
MAP_WZ = 10

# Vehicle properties
DRONE_INERTIA = np.array([[0.0125, 0.0   , 0.0  ],
                          [0.0   , 0.0125, 0.0  ],
                          [0.0   , 0.0   , 0.025]]) # [kg m^2] Inertia matrix of drone, in body frame
DRONE_MASS = 2.0 # [kg]

# Controller parameters
KP_ATT = np.array([250.0, 250.0, 250.0]) # x, y, z
KD_ATT = np.array([50.0, 50.0, 50.0])

KP_POS = np.array([5.0, 5.0, 5.0])
KD_POS = np.array([10.0, 10.0, 10.0])

g = 9.80665 # m/s^2

DYNAMICS_DT = 0.01 # [s] Time step for dynamics simulation
SIM_SPEED_FACTOR = 0.5 # Simulation speed. For example, 0.5 means run at 2x slow motion.

COLLISION_RADIUS = 0.5 # [m] Artificially increase all obtacles by this size
SENSING_HORIZON = 5.0 # [m]
SENSING_HORIZON_CONSERVATIVE = SENSING_HORIZON - COLLISION_RADIUS

MAX_SNAP = 200.0 # [m/s^4] Max snap, component-wise for X, Y, Z. TODO choose
MAX_THRUST = 100.0 # [N] Max thrust TODO choose
MAX_TORQUE = 200.0 # [N m] Max torque, component-wise for X, Y, Z

INIT_POS = np.array([MAP_WX/2, 0     , MAP_WZ/2])
GOAL_POS = np.array([MAP_WX/2, MAP_WY, MAP_WZ/2])

MPC_USE_LINEAR_MODEL = False

MPC_TIME_HORIZON = 3.0 # [s]
MPC_NUM_TIME_STEPS = 15
MPC_MAX_SOLVE_TIME = 0.5 if MPC_USE_LINEAR_MODEL else 0.5 # [s] Max time allowed for MPC solve, which determines the initial state to begin from.
