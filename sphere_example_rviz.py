#!/usr/bin/env python3

from gekko import GEKKO
import numpy as np

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def make_marker_sphere(id, x, y, z, r):
    marker = Marker()

    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()

    # set shape
    marker.type = Marker.SPHERE
    marker.id = id

    # Set the scale of the marker
    marker.scale.x = r*2
    marker.scale.y = r*2
    marker.scale.z = r*2

    # Set the color
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Set the pose of the marker
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    return marker

def make_marker_path(id, x, y, z):
    marker = Marker()

    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()

    # set shape
    marker.type = Marker.LINE_STRIP
    marker.id = id

    # Set the scale of the marker
    marker.scale.x = 0.1

    # Set the color
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Set the pose of the marker
    for i in range(len(x)):
        p = Point()
        p.x = x[i]
        p.y = y[i]
        p.z = z[i]
        marker.points.append(p)
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    return marker

rospy.init_node('squat', anonymous=True)
marker_pub = rospy.Publisher("/squat/marker", Marker, queue_size=100)

m = GEKKO()
nt = 51
m.time = np.linspace(0, 10, nt)

a = [] # acceleration (control input)
v = [] # velocity
p = [] # position
p_f = [] # final position
v_f = [] # final velocity

# 3D
for _ in range(3):
    # initialize varibales
    a.append(m.Var(value=None, lb=-1, ub=1, fixed_initial=False))
    v.append(m.Var(value=0))
    p.append(m.Var(value=0))

    # dynamics
    m.Equation(p[-1].dt() == v[-1])
    m.Equation(v[-1].dt() == a[-1])

    # set up end constraints
    p_f.append(m.Var()) # should be FV instead of Var but doesn't work for some reason
    m.Connection(p[-1], p_f[-1], 'end', 'end')
    v_f.append(m.Var())
    m.Connection(v[-1], v_f[-1], 'end', 'end')

# end constraints
m.Equations((p_f[0] == 10, p_f[1] == 10, p_f[2] == 10))
m.Equations(vf == 0 for vf in v_f)

# spherical obstacles
spheres = [
    (1, 1, 1, 1), # x, y, z, r
    (8, 8, 8, 2)
]
for s in spheres:
    m.Equation((p[0]-s[0])**2 + (p[1]-s[1])**2 + (p[2]-s[2])**2 >= s[3]**2)

# minimize control input
m.Minimize(a[0]**2 + a[1]**2 + a[2]**2)

m.options.IMODE = 6 # control
m.solve()

# get additional solution information
# import json
# with open(m.path+'//results.json') as f:
#     results = json.load(f)

markers = []

for s in spheres:
    markers.append(make_marker_sphere(len(markers), s[0], s[1], s[2], s[3]))

markers.append(make_marker_path(len(markers), p[0].value, p[1].value, p[2].value))

while not rospy.is_shutdown():
    for marker in markers:
        marker_pub.publish(marker)
    rospy.rostime.wallsleep(1.0)
