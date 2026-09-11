"""Validate /ridgeback/joint_states velocity against odom in a ROS bag."""

from __future__ import annotations

import argparse

import numpy as np
import rosbag

from mm_utils.base_velocity_guard import body_twist_to_world


def _nearest(samples, t, tol=0.05):
    best = None
    best_dt = tol
    for item in samples:
        dt = abs(item[0] - t)
        if dt < best_dt:
            best_dt = dt
            best = item
    return best


def analyze_bag(bagfile: str, t_start: float | None, t_end: float | None):
    bag = rosbag.Bag(bagfile)
    t0 = None
    est = []
    odom = []

    for topic, msg, t in bag.read_messages(
        topics=["/ridgeback/joint_states", "/odometry/filtered"]
    ):
        ts = t.to_sec()
        if t0 is None:
            t0 = ts
        rel = ts - t0
        if t_start is not None and rel < t_start:
            continue
        if t_end is not None and rel > t_end:
            continue

        if topic == "/ridgeback/joint_states":
            if len(msg.velocity) < 3 or len(msg.position) < 3:
                continue
            est.append(
                (
                    rel,
                    np.array(msg.velocity[:3], dtype=float),
                    float(msg.position[2]),
                )
            )
        else:
            tw = msg.twist.twist
            odom.append(
                (
                    rel,
                    np.array([tw.linear.x, tw.linear.y, tw.angular.z], dtype=float),
                )
            )

    bag.close()

    if not est or not odom:
        raise RuntimeError("bag missing /ridgeback/joint_states or /odometry/filtered")

    diffs = []
    spikes = 0
    for t, v_est, yaw in est:
        o = _nearest(odom, t)
        if o is None:
            continue
        v_odom_world = body_twist_to_world(o[1], yaw)
        diffs.append(float(np.linalg.norm(v_est - v_odom_world)))
        if np.max(np.abs(v_est[:2])) > 1.0:
            spikes += 1

    diffs_arr = np.array(diffs, dtype=float)
    print("Bag:", bagfile)
    print("Samples:", len(diffs_arr))
    print("max |v_est - v_odom_world|:", float(np.max(diffs_arr)))
    print("median disagreement:", float(np.median(diffs_arr)))
    print("p99 disagreement:", float(np.percentile(diffs_arr, 99)))
    print("count |v_est_linear| > 1.0 m/s:", spikes)


def main():
    parser = argparse.ArgumentParser(
        description="Check ridgeback joint_states velocity vs odom."
    )
    parser.add_argument("bagfile")
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    args = parser.parse_args()
    analyze_bag(args.bagfile, args.t_start, args.t_end)


if __name__ == "__main__":
    main()
