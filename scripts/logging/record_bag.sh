#!/bin/sh
BAG_DIR=$MOBILE_MANIPULATION_CENTRAL_BAG_DIR
mkdir -p "$BAG_DIR/$1"

 rosbag record -o "$BAG_DIR/$1" \
   /clock \
   --regex "/ridgeback/(.*)" \
   --regex "/ridgeback_velocity_controller/(.*)" \
   --regex "/ur10/(.*)" \
   --regex "/vicon/(.*)" \
   --regex "/mpc_tracking_pt" \
   --regex "/semantic_pointcloud" \
   --regex "/tf" \
   --regex "/tf_static" \
   --regex "/controller_visualization" \
   --regex "/controller_visualization_array" \
   --regex "/plan_visualization" \
   --regex "/pose_plan_visualization" \
   --regex "/pose_path_visualization" \
   --regex "/pose_waypoint_visualization" \
   --regex "/current_plan_visualization" \
   --regex "/controller_reference" \
   --regex "/controller_tracking_pt" \
   --regex "/revis_node/tracked_objects_cloud" \
   --regex "/planned_global_path" \
   --regex "/nbv"\
   --regex "/camera_base/color/image_raw/compressed"\
   --regex "/camera_hand/color/image_raw/compressed"\
   --regex "/system_delay_diagnostics/control" \
   --regex "/system_delay_diagnostics/slam"\
   --regex "/system_delay_diagnostics/slam"\
   --regex "/ground_truth_map"\
   --regex "/revis_node/tracked_objects_label"\
  #  --regex "/pocd_slam_node/occupied_ef_nodes" \
#rosbag record -a -o "$BAG_DIR/$1"