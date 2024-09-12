"""Plot UR10 and Ridgeback joint position and velocity from a ROS bag."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    est_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]
    vicon_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/ThingBase_3/ThingBase_3")]

    tbs, qbs = ros_utils.parse_ridgeback_vicon_msgs(vicon_msgs)
    tb_ests, qb_ests, vb_ests = ros_utils.parse_ridgeback_joint_state_msgs(est_msgs)
    vb_nums = np.diff(qbs, axis=0).T/np.diff(tbs)
    vb_nums = vb_nums.T
    if (len(tbs) > 0):
        tbs = tbs - tbs[0]
    tb_ests = tb_ests - tb_ests[0]

    # TODO trim messages to only start once we get a command
    plt.figure()
    if (len(tbs) > 0):
        plt.plot(tbs, qbs[:, 0], label="x_vicon")
        plt.plot(tbs, qbs[:, 1], label="y_vicon")
        plt.plot(tbs, qbs[:, 2], label="θ_vicon")
    plt.plot(tb_ests, qb_ests[:, 0], label="x_est")
    plt.plot(tb_ests, qb_ests[:, 1], label="y_est")
    plt.plot(tb_ests, qb_ests[:, 2], label="θ_est")
    plt.title("Ridgeback Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    if (len(tbs) > 0):
        plt.plot(tbs[:-1], vb_nums[:, 0], label="vx_vicon")
        plt.plot(tbs[:-1], vb_nums[:, 1], label="vy_vicon")
        plt.plot(tbs[:-1], vb_nums[:, 2], label="vθ_vicon")
    plt.plot(tb_ests, vb_ests[:, 0], label="vx_est")
    plt.plot(tb_ests, vb_ests[:, 1], label="vy_est")
    plt.plot(tb_ests, vb_ests[:, 2], label="ω_est")
    plt.title("Ridgeback Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
