import numpy as np
import rospy
import threading
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator,RegularGridInterpolator
from spatialmath.base import rotz

from geometry_msgs.msg import Twist, PoseArray
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray

from mobile_manipulation_central import ros_utils


# TODO add protections if time since last message is too large

class MapInterface:
    """
        ROS interface for receiving maps
    """

    def __init__(self, topic_name: str):
        self.map_sub = rospy.Subscriber(topic_name, MarkerArray, self._map_cb)
        self.mutex = threading.Lock()
        self.map_points = None
        self.map_vals = None
        self.msg_received = False
        self.valid = False
        self.map_updated = False
    
    def ready(self):
        return self.msg_received and self.valid
    
    def get_map(self):
        if self.map_updated:
            self.mutex.acquire(blocking=True)
            map_points_copy = self.map_points.copy()
            map_vals_copy = self.map_vals.copy()
            self.map_updated = False
            self.mutex.release()

            return True, (map_points_copy, map_vals_copy)
        else:
            return False, None
    
    def _map_cb(self, msg):
    
        if len(msg.markers)>0:
            self.mutex.acquire(blocking=True)
            self.map_points = msg.markers[0].points
            self.map_vals = msg.markers[0].colors
            self.map_updated = True
            self.mutex.release()

            self.valid = True

        self.msg_received = True

class MapInterfaceNew:
    """
        ROS interface for receiving and post-processing maps
    """

    def __init__(self, config, topic_name: str="/pocd_slam_node/occupied_ef_dist_nodes"):
        self.map_sub = rospy.Subscriber(topic_name, MarkerArray, self._map_cb)
        self.mutex = threading.Lock()
        self.robot_pose_mutex = threading.Lock()

        self.map_received = False
        self.joint_states_received = False
        self.valid = False

        self.map = None
        self.tsdf = None
        self.tsdf_val = None

        self.mul=10
        self.map_coverage = np.array(config["map"]["map_coverage"])
        self.default_val = config["map"]["default_val"]
        self.voxel_size = config["map"]["voxel_size"]
        self.map_size = np.ceil(self.map_coverage / self.voxel_size).astype(int)
        
        self.joint_state_sub = rospy.Subscriber(
            "/ridgeback/joint_states", JointState, self._joint_state_cb
        )

    def ready(self):
        return self.map_received and self.valid and self.joint_states_received
    
    def get_map(self):
        if self.ready():
            self.mutex.acquire(blocking=True)
            map = self.map
            self.mutex.release()

            return True, map
        else:
            return False, None
    
    def _map_cb(self, msg):
    
        if len(msg.markers)>0 and self.joint_states_received:

            tsdf = msg.markers[0].points
            tsdf_vals = msg.markers[0].colors
            map = self._create_map(tsdf, tsdf_vals)
            self.mutex.acquire(blocking=True)
            self.map = map
            self.tsdf = tsdf
            self.tsdf_vals = tsdf_vals
            self.mutex.release()

            self.valid = True

        self.map_received = True

    def _joint_state_cb(self, msg):
        """Callback for Ridgeback joint feedback."""
        q = np.array(msg.position)
        Tbw = np.eye(4)
        Tbw[:2, 3] = q[:2]
        Tbw[:3, :3] = rotz(q[2])
        self.robot_pose_mutex.acquire(blocking=True)
        self.curr_robot_pose = Tbw
        self.robot_pose_mutex.release()

        self.joint_states_received = True

    
    def _create_map(self, tsdf, tsdf_vals):
        pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
        vs = [c.r * self.mul for c in tsdf_vals]

        xg, yg = self._get_grid()

        X, Y = np.meshgrid(xg, yg, indexing='ij')
        map_ir = LinearNDInterpolator(pts, vs)
        map_ir((0,0))

        v = np.nan_to_num(map_ir(X, Y), True, self.default_val)
        v = v.ravel(order='F')

        return xg, yg, v
    
    def _get_grid(self):
        # Limit the map to a certain size around the robot
        self.robot_pose_mutex.acquire(blocking=True)
        curr_robot_pose = self.curr_robot_pose.copy()
        self.robot_pose_mutex.release()
        max_x = np.around(curr_robot_pose[0,3]+self.map_coverage[0]/2, 2)
        min_x = np.around(curr_robot_pose[0,3]-self.map_coverage[0]/2, 2)
        max_y = np.around(curr_robot_pose[1,3]+self.map_coverage[1]/2, 2)
        min_y = np.around(curr_robot_pose[1,3]-self.map_coverage[1]/2, 2)

        xg = np.linspace(min_x, max_x, self.map_size[0])
        yg = np.linspace(min_y, max_y, self.map_size[1])

        return xg, yg

class JoystickButtonInterface:
    """
        Monitor on joy stick button. Flag event when the button is pressed. Event flag can only be cleared externally.
    """

    def __init__(self, button_index):

        self.button_index = button_index
        self.button = 0             # 1 pressed, 0 available
        self.busy = False
        self.button_lock = threading.Lock()
        self.block_out_time = 0.5 # 0.5 second
        self.last_reset_time = rospy.Time.now().to_sec()

        self.msg_received = False
        self.joy_sub = rospy.Subscriber("/bluetooth_teleop/joy", Joy, self._joy_cb, queue_size=10)


    def _joy_cb(self, msg):
        if msg.buttons[self.button_index] == 1:
            self._update_button(1)
            print("set button {}".format(self.button))


        self.msg_received = True

    def reset_button(self):
        self._update_button(0, True)
        print("reset button {}".format(self.button))
        self.last_reset_time = rospy.Time.now().to_sec()


    def _update_button(self, value, force=False):
        t_now = rospy.Time.now().to_sec()
        if t_now - self.last_reset_time > self.block_out_time or force:
            if value != self.button:
                self.button_lock.acquire()
                self.button = value
                print("update button {}".format(self.button))

                self.button_lock.release()

    def ready(self):
        """True if a Vicon message has been received."""
        return self.msg_received


class ViconObjectInterface:
    """ROS interface for receiving Vicon measurements for an object's pose."""

    def __init__(self, name):
        topic = "/vicon/" + name + "/" + name
        self.msg_received = False
        self.sub = rospy.Subscriber(topic, TransformStamped, self._transform_cb)

    def ready(self):
        """True if a Vicon message has been received."""
        return self.msg_received

    def _transform_cb(self, msg):
        L = msg.transform.translation
        Q = msg.transform.rotation

        self.position = np.array([L.x, L.y, L.z])
        self.orientation = np.array([Q.x, Q.y, Q.z, Q.w])

        self.msg_received = True

class ViconMarkerSwarmInterface:
    """ROS interface for receiving state estiation for a swarm of vicon markers."""

    def __init__(self, topic):
        self.msg_received = False
        self.sub = rospy.Subscriber(topic, PoseArray, self._pose_array_cb)

    def ready(self):
        """True if a swarm state message has been received."""
        return self.msg_received

    def _pose_array_cb(self, msg):

        self.position = []
        self.orientation = []
        for pose in msg.poses:
            L = pose.position
            Q = pose.orientation
            self.position.append([L.x, L.y, L.z])
            self.orientation.append([Q.x, Q.y, Q.z, Q.w])

        self.position = np.array(self.position)
        self.orientation = np.array(self.orientation)

        self.msg_received = True

class RobotROSInterface:
    """Base class for defining ROS interfaces for robots."""

    def __init__(self, nq, nv):
        self.nq = nq
        self.nv = nv

        self.q = np.zeros(self.nq)
        self.v = np.zeros(self.nv)

        self.joint_states_received = False

    def brake(self):
        """Brake (stop) the robot."""
        self.publish_cmd_vel(np.zeros(self.nv))

    def ready(self):
        """True if joint state messages have been received."""
        return self.joint_states_received


class RidgebackROSInterface(RobotROSInterface):
    """ROS interface for the Ridgeback mobile base."""

    def __init__(self):
        super().__init__(nq=3, nv=3)

        self.cmd_pub = rospy.Publisher("/ridgeback/cmd_vel", Twist, queue_size=1)
        self.joint_state_sub = rospy.Subscriber(
            "/ridgeback/joint_states", JointState, self._joint_state_cb
        )

    def _joint_state_cb(self, msg):
        """Callback for Ridgeback joint feedback."""
        self.q = np.array(msg.position)
        self.v = np.array(msg.velocity)

        self.joint_states_received = True

    def publish_cmd_vel(self, cmd_vel, bodyframe=False):
        """Command the velocity of the robot's joints.

        Setting bodyframe to True indicated that the command is in the base's
        body frame; False indicates it is in the world frame.
        """
        assert cmd_vel.shape == (self.nv,)

        # rotate into body frame from world frame if needed
        if not bodyframe:
            C_bw = rotz(-self.q[2])
            cmd_vel = C_bw @ cmd_vel

        msg = Twist()
        msg.linear.x = cmd_vel[0]
        msg.linear.y = cmd_vel[1]
        msg.angular.z = cmd_vel[2]
        self.cmd_pub.publish(msg)


class UR10ROSInterface(RobotROSInterface):
    """ROS interface for the UR10 arm."""

    def __init__(self):
        super().__init__(nq=6, nv=6)

        self.cmd_pub = rospy.Publisher("/ur10/cmd_vel", Float64MultiArray, queue_size=1)
        self.joint_state_sub = rospy.Subscriber(
            "/ur10/joint_states", JointState, self._joint_state_cb
        )

    def _joint_state_cb(self, msg):
        """Callback for arm joint feedback."""
        _, self.q, self.v = ros_utils.parse_ur10_joint_state_msg(msg)
        self.joint_states_received = True

    def publish_cmd_vel(self, cmd_vel, bodyframe=None):
        """Command the velocity of the robot's joints.

        The bodyframe option changes nothing, but is provided for compatibility
        with the Ridgeback interface.
        """
        assert cmd_vel.shape == (self.nv,)

        msg = Float64MultiArray()
        msg.data = list(cmd_vel)
        self.cmd_pub.publish(msg)


class MobileManipulatorROSInterface:
    """ROS interface to the real mobile manipulator."""

    def __init__(self):
        self.arm = UR10ROSInterface()
        self.base = RidgebackROSInterface()

        self.nq = self.arm.nq + self.base.nq
        self.nv = self.arm.nv + self.base.nv

    def brake(self):
        """Brake (stop) the robot."""
        self.base.brake()
        self.arm.brake()

    def ready(self):
        """True if joint state messages have been received for both arm and base."""
        return self.base.ready() and self.arm.ready()

    def publish_cmd_vel(self, cmd_vel, bodyframe=False):
        """Command the velocity of the robot's joints.

        Setting bodyframe to True indicated that the command is in the base's
        body frame; False indicates it is in the world frame.
        """
        assert cmd_vel.shape == (self.nv,)

        self.base.publish_cmd_vel(cmd_vel[: self.base.nv], bodyframe=bodyframe)
        self.arm.publish_cmd_vel(cmd_vel[self.base.nv :])

    @property
    def q(self):
        """Latest joint configuration measurement."""
        return np.concatenate((self.base.q, self.arm.q))

    @property
    def v(self):
        """Latest joint velocity measurement.

        Note that the base velocity is in the world frame.
        """
        return np.concatenate((self.base.v, self.arm.v))
