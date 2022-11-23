import numpy as np

# Using ENU (East North Up) coordinate frame
MAP_WX = 10
MAP_WY = 40
MAP_WZ = 10

COLLISION_RADIUS = 0.5 # [m] Artificially increase all obtacles by this size
SENSING_HORIZON = 5.0 # [m]
SENSING_HORIZON_CONSERVATIVE = SENSING_HORIZON - COLLISION_RADIUS

MAX_SNAP = 200.0 # [m/s^4] Max snap, component-wise for X, Y, Z. TODO: Can estimate from max angular acceleration of quadcopter.

INIT_POS = np.array([MAP_WX/2, 0     , MAP_WZ/2])
GOAL_POS = np.array([MAP_WX/2, MAP_WY, MAP_WX/2])

MPC_TIME_HORIZON = 3.0 # [s]
MPC_NUM_TIME_STEPS = 15
MPC_EXPECTED_SOLVE_TIME = 0.5 # [s] Expected computation time for MPC solve, which determines the initial state to begin from.

# g = 9.80665 # m/s^2
