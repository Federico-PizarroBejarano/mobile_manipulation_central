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

    tracking_pt_msgs = [msg for _, msg, _ in bag.read_messages("/mpc_tracking_pt")]
    tool_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/ThingWoodTray/ThingWoodTray")]
    ridgeback_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/ThingBase/ThingBase")]

    t_ees, pose_ees = ros_utils.parse_transform_stamped_msgs(tool_msgs, False)
    t_rs, pose_rs = ros_utils.parse_multidofjointtrajectory_msg(tracking_pt_msgs, False)
    t_bs, q_bs = ros_utils.parse_ridgeback_vicon_msgs(ridgeback_msgs)

    pose_rbs = pose_rs.get("base", [])
    pose_rees = pose_rs.get("EE", [])

    t_ees -= t_rs[0]
    t_bs -= t_rs[0]
    t_rs -= t_rs[0]

    if len(pose_rbs) > 0:
        plt.figure()
        plt.plot(t_bs, q_bs[:, 0], 'r', label="x")
        plt.plot(t_bs, q_bs[:, 1], 'b', label="y")
        plt.plot(t_rs, pose_rbs[:, 0], 'r--', label="x")
        plt.plot(t_rs, pose_rbs[:, 1], 'g--', label="y")
        plt.title("Ridgeback Joint Positions")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint position")
        plt.legend()
        plt.grid()

    if len(pose_rees) > 0:
        plt.figure()
        plt.plot(t_rs, pose_rees[:, 0], label="xd", color='r')
        plt.plot(t_rs, pose_rees[:, 1], label="yd", color="g")
        plt.plot(t_rs, pose_rees[:, 2], label="zd", color="b")
        plt.plot(t_ees, pose_ees[:, 0],'r--', label="x")
        plt.plot(t_ees, pose_ees[:, 1],'g--', label="y")
        plt.plot(t_ees, pose_ees[:, 2],'b--', label="z")
        plt.title("EE tracking")
        plt.xlabel("Time (s)")
        plt.ylabel("EE position (m)")
        plt.legend()
        plt.grid()

    # plt.figure()
    # for i in range(6):
    #     plt.plot(tas, qas[:, i], label=f"θ_{i+1}")
    # plt.title("UR10 Joint Positions")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Joint position (rad)")
    # plt.legend()
    # plt.grid()

    # plt.figure()
    # for i in range(6):
    #     plt.plot(tas, vas[:, i], label=f"v_{i+1}")
    # plt.title("UR10 Joint Velocities")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Joint velocity (rad/s)")
    # plt.legend()
    # plt.grid()

    # plt.figure()
    # for i in range(6):
    #     plt.plot(ur10_cmd_ts, ur10_cmd_vels[:, i], label=f"vc_{i+1}")
    # plt.title("UR10 Commanded Joint Velocities")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Joint velocity (rad/s)")
    # plt.legend()
    # plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
