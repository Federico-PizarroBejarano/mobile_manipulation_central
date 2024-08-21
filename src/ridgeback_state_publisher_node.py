#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import argparse
import sys
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler, rotation_matrix,inverse_matrix, quaternion_from_matrix
from sensor_msgs.msg import JointState

class RidgebackStatePublisherNode:

    def __init__(self):
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument("--tf_prefix", type=str, default="", help="TF publish frequency (Hz).")
        parser.add_argument("--joint_states_topic", type=str, default="ridgeback/joint_states", help="TF publish frequency (Hz).")
        args = parser.parse_args(argv[1:])

        self.tf_prefix = args.tf_prefix

        self.broadcaster_static = tf2_ros.StaticTransformBroadcaster()
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.odom_map_tf = TransformStamped()
        self.odom_map_tf.header.stamp = rospy.Time.now()
        self.odom_map_tf.header.frame_id = "world"
        self.odom_map_tf.child_frame_id = "odom"
        self.odom_init = False

        self.baselink_map_tf = TransformStamped()
        self.baselink_map_tf.header.stamp = rospy.Time.now()
        self.baselink_map_tf.header.frame_id = "base_link"
        self.baselink_map_tf.child_frame_id = "world"

        self.sub = rospy.Subscriber(args.joint_states_topic, JointState, self._joint_states_cb)

    def _joint_states_cb(self, msg):
        '''
        if not self.odom_init:
            self.odom_map_tf.transform.translation.x = msg.position[0]
            self.odom_map_tf.transform.translation.y = msg.position[1]
            self.odom_map_tf.transform.translation.z = 0.

            quat = quaternion_from_euler(0., 0., msg.position[2])
            self.odom_map_tf.transform.rotation.x = quat[0]
            self.odom_map_tf.transform.rotation.y = quat[1]
            self.odom_map_tf.transform.rotation.z = quat[2]
            self.odom_map_tf.transform.rotation.w = quat[3]

            self.odom_init = True

        # self.odom_map_tf.header.stamp = rospy.Time.now()
        self.odom_map_tf.header.stamp = msg.header.stamp
        self.broadcaster_static.sendTransform(self.odom_map_tf)
        '''
        # self.baselink_map_tf.header.stamp = rospy.Time.now()

        Twb = rotation_matrix(msg.position[2], (0,0,1))
        Twb[0, 3] = msg.position[0]
        Twb[1, 3] = msg.position[1]
        Tbw = inverse_matrix(Twb)
        quat = quaternion_from_matrix(Tbw)
        self.baselink_map_tf.header.stamp = msg.header.stamp
        self.baselink_map_tf.transform.translation.x = Tbw[0, 3]
        self.baselink_map_tf.transform.translation.y = Tbw[1, 3]
        self.baselink_map_tf.transform.translation.z = 0.
        self.baselink_map_tf.transform.rotation.x = quat[0]
        self.baselink_map_tf.transform.rotation.y = quat[1]
        self.baselink_map_tf.transform.rotation.z = quat[2]
        self.baselink_map_tf.transform.rotation.w = quat[3]
        self.broadcaster.sendTransform(self.baselink_map_tf)

if __name__ == "__main__":
    rospy.init_node("ridgeback_state_publisher")

    node = RidgebackStatePublisherNode()

    rospy.spin()
