import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
import numpy as np

import config

markers = {'static': []}
poses = {}
pose_pubs = {}

def init():
    global marker_pub
    rospy.init_node('squat', anonymous=True)
    marker_pub = rospy.Publisher("/squat/marker", Marker, queue_size=10)

def show_once():
    for ns in markers:
        for marker in markers[ns]:
            marker_pub.publish(marker)
    for name in poses:
        if not name in pose_pubs:
            pose_pubs[name] = rospy.Publisher("/squat/pose/" + name, PoseStamped, queue_size=10)
        pose_pubs[name].publish(poses[name])

def show():
    while not rospy.is_shutdown():
        show_once()
        rospy.rostime.wallsleep(1.0)

def new_marker(namespace, id=0):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "map"
    marker.ns = namespace
    marker.id = id
    marker.type = Marker.CUBE

    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    return marker

def make_ellipsoid(marker, x, y, z, rx, ry, rz):
    marker.type = Marker.SPHERE
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.scale.x = rx*2
    marker.scale.y = ry*2
    marker.scale.z = rz*2

def make_cylinder(marker, axis, x_or_y, y_or_z, r):
    marker.type = Marker.CYLINDER

    axes = ['x', 'y', 'z']
    axis_str = axes[axis]
    axes.remove(axis_str)

    map_width = (config.MAP_WX, config.MAP_WY, config.MAP_WZ)[axis]

    setattr(marker.pose.position, axes[0], x_or_y)
    setattr(marker.pose.position, axes[1], y_or_z)
    setattr(marker.pose.position, axis_str, map_width / 2)
    if axis != 2:
        rot_axis = np.zeros(3)
        rot_axis[1 - axis] = 1
        marker.pose.orientation.x = np.sqrt(2)/2 * rot_axis[0]
        marker.pose.orientation.y = np.sqrt(2)/2 * rot_axis[1]
        marker.pose.orientation.z = np.sqrt(2)/2 * rot_axis[2]
        marker.pose.orientation.w = np.sqrt(2)/2

    marker.scale.x = r*2
    marker.scale.y = r*2
    marker.scale.z = map_width

def make_box(marker, x, y, z, rx, ry, rz):
    marker.type = Marker.CUBE
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.scale.x = rx*2
    marker.scale.y = ry*2
    marker.scale.z = rz*2

def make_path(marker, x, y, z):
    marker.type = Marker.LINE_STRIP
    marker.scale.x = 0.1
    for i in range(len(x)):
        p = Point()
        p.x = x[i]
        p.y = y[i]
        p.z = z[i]
        marker.points.append(p)

def make_pose(p, q):
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "map"

    pose.pose.position.x = p[0]
    pose.pose.position.y = p[1]
    pose.pose.position.z = p[2]
    pose.pose.orientation.w = q[0]
    pose.pose.orientation.x = q[1]
    pose.pose.orientation.y = q[2]
    pose.pose.orientation.z = q[3]
    return pose

def add_ellipsoid_obstacle(x, y, z, rx, ry, rz):
    marker = new_marker('static', len(markers['static']))
    make_ellipsoid(marker, x, y, z, rx, ry, rz)
    marker.color.r = 1.0
    markers['static'].append(marker)

def add_box_obstacle(x, y, z, rx, ry, rz):
    marker = new_marker('static', len(markers['static']))
    make_box(marker, x, y, z, rx, ry, rz)
    marker.color.r = 1.0
    markers['static'].append(marker)

def add_cylinder_obstacle(axis, x_or_y, y_or_z, r):
    marker = new_marker('static', len(markers['static']))
    make_cylinder(marker, axis, x_or_y, y_or_z, r)
    marker.color.r = 1.0
    markers['static'].append(marker)

def add_obstacles(obstacles):
    for obs in obstacles:
        if obs.type == 'ellipsoid':
            add_ellipsoid_obstacle(obs.x, obs.y, obs.z, obs.rx, obs.ry, obs.rz)
        elif obs.type == 'box':
            add_box_obstacle(obs.x, obs.y, obs.z, obs.rx, obs.ry, obs.rz)
        elif obs.type == 'cylinder':
            add_cylinder_obstacle(obs.axis, obs.x_or_y, obs.y_or_z, obs.r)

def update_gekko_path(x, y, z):
    marker = new_marker('gekko_path')
    make_path(marker, x, y, z)
    marker.color.g = 1.0
    markers['gekko_path'] = [marker]

def update_smooth_path(x, y, z):
    marker = new_marker('smooth_path')
    make_path(marker, x, y, z)
    marker.color.b = 1.0
    markers['smooth_path'] = [marker]

def update_vehicle(p, q, collision_r, sense_r, plan_r):
    x, y, z = p

    marker = new_marker('vehicle')
    make_ellipsoid(marker, x, y, z, collision_r, collision_r, collision_r)
    marker.color.r = 1.0
    marker.color.b = 1.0
    markers['vehicle'] = [marker]

    poses['vehicle_pose'] = make_pose(p, q)

    marker = new_marker('sensing_horizon')
    make_ellipsoid(marker, x, y, z, sense_r, sense_r, sense_r)
    marker.color.r = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.25
    markers['sensing_horizon'] = [marker]

    marker = new_marker('planning_radius')
    make_ellipsoid(marker, x, y, z, plan_r, plan_r, plan_r)
    marker.color.r = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.125
    markers['planning_radius'] = [marker]

def update_goal(x, y, z):
    marker = new_marker('goal')
    make_ellipsoid(marker, x, y, z, 0.2, 0.2, 0.2)
    marker.color.b = 1.0
    markers['goal'] = [marker]
