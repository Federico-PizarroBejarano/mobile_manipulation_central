import rospy
import threading
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Joy, JointState

from mm_utils.gripper_cable_guard import (
    CABLE_GUARD_ENABLED_PARAM,
    CABLE_GUARD_RELEASE_PARAM,
    DEFAULT_RELEASE_MARGIN_RAD,
    GripperCableGuard,
)
from mm_utils.teleop_mapping import hardware_deadman_held
from mobile_manipulation_central.ros_utils import parse_ur10_joint_state_msg

class ControlCommandRelayInterface:
    """Base class for control command topic relay interface"""

    def __init__(self):
        self.switch = 0     # 0 - send empty message, 1 - send controller messages
        self.switch_lock = threading.Lock()

    def update_switch(self, val):
        if self.switch != val:
            self.switch_lock.acquire()
            self.switch = val
            self.switch_lock.release()

class RidgebackControlCommandRelayInterface(ControlCommandRelayInterface):

    def __init__(self):
        super().__init__()

        self.cmd_sub = rospy.Subscriber("/ridgeback/cmd_vel", Twist, self._cmd_vel_cb)
        self.relay_pub = rospy.Publisher("/ridgeback_velocity_controller/cmd_vel", Twist, queue_size=1)
        self.empty_msg = Twist()

    def _cmd_vel_cb(self, msg):
        self.switch_lock.acquire()
        switch = self.switch
        self.switch_lock.release()

        if switch == 1:
            self.relay_pub.publish(msg)
        else:
            self.relay_pub.publish(self.empty_msg)

class UR10ControlCommandRelayInterface(ControlCommandRelayInterface):

    def __init__(self):
        super().__init__()

        self.cmd_sub = rospy.Subscriber("/ur10/cmd_vel", Float64MultiArray, self._cmd_vel_cb)
        self.relay_pub = rospy.Publisher("/ur10/ur10_velocity_controller/cmd_vel", Float64MultiArray, queue_size=1)
        self.joint_state_sub = rospy.Subscriber(
            "/ur10/joint_states", JointState, self._joint_state_cb
        )
        self.q = None
        self._cable_guard = GripperCableGuard(enabled=False)
        self.empty_msg = Float64MultiArray()
        self.empty_msg.data = [0]*6

    def _joint_state_cb(self, msg):
        _, q, _, _ = parse_ur10_joint_state_msg(msg)
        self.q = q

    def _cmd_vel_cb(self, msg):
        self.switch_lock.acquire()
        switch = self.switch
        self.switch_lock.release()

        if switch == 1:
            data = np.asarray(msg.data, dtype=float).reshape(-1)
            if self.q is not None and data.size == 6:
                self._cable_guard.enabled = bool(
                    rospy.get_param(CABLE_GUARD_ENABLED_PARAM, False)
                )
                self._cable_guard.release_margin_rad = float(
                    rospy.get_param(CABLE_GUARD_RELEASE_PARAM, DEFAULT_RELEASE_MARGIN_RAD)
                )
                data, clipped = self._cable_guard.apply(self.q, data)
                if clipped:
                    rospy.logwarn_throttle(
                        2.0,
                        "Gripper cable guard: arm held at the cable limit. "
                        "Yaw the gripper back to move again.",
                    )
                msg = Float64MultiArray()
                msg.data = data.tolist()
            self.relay_pub.publish(msg)
        else:
            self.relay_pub.publish(self.empty_msg)

class MobileManipulatorControlCommandRelayInterface:

    def __init__(self):
        self.arm = UR10ControlCommandRelayInterface()
        self.base = RidgebackControlCommandRelayInterface()

        self.joy_sub = rospy.Subscriber("/bluetooth_teleop/joy", Joy, self._joy_cb)

    def _joy_cb(self, msg):
        pressed = 1 if hardware_deadman_held(msg.axes) else 0
        self.arm.update_switch(pressed)
        self.base.update_switch(pressed)