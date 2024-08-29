#!/usr/bin/env python3
"""Send a constant velocity to a single joint for a given duration to test the system response."""
import numpy as np
import rospy
import argparse
from scipy.interpolate import interp1d

from mobile_manipulation_central import MobileManipulatorROSInterface

VELOCITY = 1
ACCELERATION = 1.0
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
    hz = 100
    rate = rospy.Rate(50)
    while not args.dry_run and not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    
    dt = 1.0/125
    
    acc_time = VELOCITY/ACCELERATION
    acc_time_steps = int(acc_time/dt)+1
    cmd_vel_acc = np.zeros((acc_time_steps, robot.nv))
    cmd_vel_acc[:, args.joint_index] = ACCELERATION * np.arange(acc_time_steps)* dt

    cruise_time_steps = int(DURATION/dt) + 1
    cmd_vel_cruise = np.zeros((cruise_time_steps, robot.nv))
    cmd_vel_cruise[:, args.joint_index] = VELOCITY

    cmd_vel = np.vstack((cmd_vel_acc, cmd_vel_cruise))
    t = np.arange(acc_time_steps+cruise_time_steps) * dt
    cmd_vel_fcn = interp1d(t, cmd_vel, axis=0, bounds_error=False, fill_value="extrapolate")
    

    # send command, wait, then brake
    if args.dry_run:
        print(cmd_vel[:, args.joint_index])
    else:
        # to register cmd with ros bag
        robot.brake()
        rospy.sleep(2.0)
        t_init = rospy.get_time()
        while not rospy.is_shutdown():
            t_current = rospy.get_time()
            if t_current - t_init > DURATION+acc_time:
                break

            robot.publish_cmd_vel(cmd_vel_fcn(t_current - t_init))
            print(cmd_vel_fcn(t_current - t_init))
            rate.sleep()
    robot.brake()


if __name__ == "__main__":
    main()
