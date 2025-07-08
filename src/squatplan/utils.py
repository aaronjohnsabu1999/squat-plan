# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * Department of Mechanical and Aerospace Engineering      *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Ryan Nemiroff, Brett T. Lopez *
# * Contact: {aaronjs, ryguyn, btlopez}@ucla.edu            *
# *                                                         *
# ***********************************************************/

from typing import Sequence


def quat_mult(q1: Sequence[float], q2: Sequence[float]) -> list[float]:
    """Multiplies two quaternions q1 and q2 (Hamilton product).

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Resulting quaternion [w, x, y, z]
    """
    if len(q1) != 4 or len(q2) != 4:
        raise ValueError("Both quaternions must be 4-dimensional [w, x, y, z]")

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
    ]


def quat_conj(q: Sequence[float]) -> list[float]:
    """Returns the conjugate of a quaternion.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Conjugate [w, -x, -y, -z]
    """
    if len(q) != 4:
        raise ValueError("Quaternion must be 4-dimensional [w, x, y, z]")
    return [q[0], -q[1], -q[2], -q[3]]


def quat_rot(q: Sequence[float], v: Sequence[float]) -> list[float]:
    """Rotates a 3D vector v by quaternion q.

    Args:
        q: Unit quaternion [w, x, y, z]
        v: 3D vector [x, y, z]

    Returns:
        Rotated vector [x', y', z']
    """
    if len(v) != 3:
        raise ValueError("Vector must be 3-dimensional [x, y, z]")

    q_v = [0.0] + list(v)
    q_rot = quat_mult(quat_mult(q, q_v), quat_conj(q))
    return q_rot[1:]  # Extract vector part
