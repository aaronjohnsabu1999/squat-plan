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

#!/usr/bin/env python3

# FUNCTION DEFINITIONS
def quat_mult(q_1, q_2):
    w_1, x_1, y_1, z_1 = q_1
    w_2, x_2, y_2, z_2 = q_2
    w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
    x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
    y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
    z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2
    return [w, x, y, z]

def quat_conj(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

def quat_rot(q, v):
    q_conj = quat_conj(q)
    q_v = [0, v[0], v[1], v[2]]
    q_v_rot = quat_mult(q, quat_mult(q_v, q_conj))
    return [q_v_rot[1], q_v_rot[2], q_v_rot[3]]