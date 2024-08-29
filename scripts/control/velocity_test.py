#!/usr/bin/env python3
"""Send a constant velocity to a single joint for a given duration to test the system response."""
import numpy as np
import rospy
import argparse

from mobile_manipulation_central import MobileManipulatorROSInterface

VELOCITY = 0.5
DURATION = 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "joint_index",
        help="Index of the robot joint to send velocity step to.",
        type=int,
    )
    parser.add_argument(
        "--dry-run",
        help="Don't send any commands, just print out what would be sent.",
        action="store_true",
    )
    args = parser.parse_args()

    rospy.init_node("velocity_test")

    robot = MobileManipulatorROSInterface()

    # wait until robot feedback has been received
    rate = rospy.Rate(125)
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    cmd_vel = np.zeros(robot.nv)
    cmd_vel[args.joint_index] = VELOCITY

    # send command, wait, then brake
    if args.dry_run:
        print(cmd_vel)
    else:
        t_init = rospy.get_time()
        while not rospy.is_shutdown() and rospy.get_time() - t_init < DURATION:
            robot.publish_cmd_vel(cmd_vel)
            rate.sleep()

    robot.brake()


if __name__ == "__main__":
    main()
