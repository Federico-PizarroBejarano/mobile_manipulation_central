#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <ros/console.h>
#include <Eigen/Eigen>
#include <vector>
#include <mobile_manipulation_central/exponential_smoothing.h>
#include <mobile_manipulation_central/fixed_size_vector.h>

#include <mobile_manipulation_central/wrap.h>
#include <iostream>
// The node listens to tf transforms for the Ridgeback base and
// converts it into a JointState message which includes a numerically
// differentiated and filtered velocity estimate on (x, y, yaw).
class RidgebackTfOdomEstimatorNode {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RidgebackTfOdomEstimatorNode(): t_odom_hist(40), vb_odom_hist(40),q_hist(40) {}

    // Start the node.
    bool start(ros::NodeHandle& nh) {
        // Retrieve parameters, including the frame id with a default value
        nh.param<std::string>("target_frame_id", target_frame_id_, "base_link");
        nh.param<std::string>("source_frame_id", source_frame_id_, "world");
        nh.param<double>("tau_linear", tau_linear, 0.045);
        nh.param<double>("tau_angular", tau_angular, 0.025);
        nh.param<std::string>("odom_topic", odom_topic_, "/odometry/filtered");
        
        std::string base_vicon_topic;
        nh.param<std::string>("base_vicon_topic", base_vicon_topic,
                              "/vicon/ThingBase_3/ThingBase_3");

        ridgeback_vicon_sub = nh.subscribe(
            base_vicon_topic, 1,
            &RidgebackTfOdomEstimatorNode::vicon_cb, this);

        ridgeback_joint_states_pub =
            nh.advertise<sensor_msgs::JointState>("/ridgeback/joint_states", 1);
        ridgeback_odom_sub = nh.subscribe(
            odom_topic_, 1,
            &RidgebackTfOdomEstimatorNode::ridgeback_odom_cb, this);

        // Velocity is assumed to be 0 initially. Values for tau taken from
        // dsl__estimation__vicon package.
        linear_velocity_filter.init(tau_linear, Eigen::Vector2d::Zero());
        angular_velocity_filter.init(tau_angular, 0);

        initial_state_ready = false;
        calibration_ready = false;
        new_tf_available = false;
        return true;
    }

    bool calculate_odom_bias(){
        if (odom_msg_count > 50){
            v_o = vb_odom_hist.avg();
            ROS_INFO_STREAM("TF_ODOM_ESTIMATOR: Calibrated odom velocity bias: "<< v_o);

            return true;
        }
        else{
            return false;
        }
    }

    void vicon_cb(const geometry_msgs::TransformStamped& msg) {
        tf::StampedTransform transform;
        tf::transformStampedMsgToTF(msg, transform);
        transform.frame_id_ = "my_world";
        transform.child_frame_id_ = "vicon_base_link";
        transform.getOrigin().setZ(0.0);
        tf_br.sendTransform(transform);
    }

    void ridgeback_odom_cb(const nav_msgs::Odometry &msg) {
        check_tf();

        if (!initial_state_ready && new_tf_available){
            t_prev = t_tf;
            q_prev = q_tf;
            q_curr = q_tf;
            initial_state_ready = true; 
            ROS_INFO_STREAM("TF_ODOM_ESTIMATOR: Initialized with first TF");
        }

        if (!initial_state_ready) {
            return;
        }

        ros::Time msg_stamp = msg.header.stamp;
        double t = msg_stamp.toSec();
        Eigen::Vector3d vb;
        vb << msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z;
        t_odom_hist.add(t);
        vb_odom_hist.add(vb);
        q_hist.add(q_curr);
        ++odom_msg_count;

        // calibration
        if (!calibration_ready){
            calibration_ready = calculate_odom_bias();
            if (calibration_ready){
                ROS_INFO_STREAM("TF_ODOM_ESTIMATOR: Calibration ready");
            }
        }

        if (calibration_ready){
            // prediction
            v_curr = to_world_frame(vb - v_o, q_prev(2));
            q_curr = propergate_motion_model(q_prev, v_curr, t-t_prev);


            // correction
            if(new_tf_available){
                // sanity check
                if (t - t_tf > 0.3){
                    ROS_WARN_STREAM("TF delayed by " << t - t_tf << "secs. \n");
                }

                // forward integration from the lastest TF pose using past odom velocities
                auto result = t_odom_hist.getSubVectorLargerThan(t_tf);
                std::vector<double> t_odom_sub = result.first;
                int start_index = result.second;
                if (start_index == -1){
                    ROS_WARN_STREAM("TF significantly delayed by " << t_odom_hist.getElementByIndex(0) - t_tf << "secs.");
                }
                else{
                    // sanity check
                    Eigen::Vector3d q_odom = q_hist.getElementByIndex(start_index);
                    auto q_diff_norm = (q_odom - q_tf).norm();

                    if (q_diff_norm > 0.5){
                        ROS_WARN_STREAM("TF diviates from odom by " << q_diff_norm << " m. ");
                    }


                    std::vector<Eigen::Vector3d> vb_odom_sub = vb_odom_hist.getSubVectorFromIndex(start_index);
                    double t_fw = t_tf;
                    double t_fw_next;
                    double dt;

                    Eigen::Vector3d q_fw = q_tf;
                    Eigen::Vector3d vb_fw;
                    Eigen::Vector3d v_fw;


                    for (int i=0; i<t_odom_sub.size(); ++i){
                        t_fw_next = t_odom_sub[i];
                        dt = t_fw_next - t_fw;
                        vb_fw = vb_odom_sub[i];
                        v_fw = to_world_frame(vb_fw, q_fw(2));
                        q_fw = propergate_motion_model(q_fw, v_fw, dt);

                        t_fw = t_fw_next;
                    }

                    q_curr = q_fw;
                    v_curr = v_fw;
                    
                }

                new_tf_available = false;
            }
            
            publish_ridgeback_joint_states(msg_stamp, q_curr, v_curr);
            q_prev = q_curr;
            t_prev = t;
        }

    }

    void check_tf() {
        // Listen for the tf transform between the source and target frames
        tf::StampedTransform transform;
        try {
            tf_listener_.lookupTransform(source_frame_id_, target_frame_id_, ros::Time(0), transform);
        } catch (tf::TransformException& ex) {
            ROS_WARN("%s", ex.what());
            return;
        }

        // Get the current time and joint configuration
        double t = transform.stamp_.toSec();
        double t_now = ros::Time::now().toSec();
        // std::cout << "Time delay: " << t_now - t;
        Eigen::Vector3d q;
        q << transform.getOrigin().x(), transform.getOrigin().y(), tf::getYaw(transform.getRotation());

        if (tf_msg_count == 0 || t - t_tf > 0.0) {
            t_tf = t;
            q_tf = q;

            new_tf_available = true;
            ++tf_msg_count;
        }
        else{
            new_tf_available = false;
        }
    }

   private:
    /* FUNCTIONS */

    void publish_ridgeback_joint_states(const ros::Time& t,
                                        const Eigen::Vector3d& q,
                                        const Eigen::Vector3d& v) {
        sensor_msgs::JointState msg;
        // msg.header.stamp = ros::Time::now();
        msg.header.stamp = t;

        msg.name = {"x", "y", "yaw"};

        for (int i = 0; i < 3; ++i) {
            msg.position.push_back(q(i));
            msg.velocity.push_back(v(i));
        }

        ridgeback_joint_states_pub.publish(msg);
    }

    Eigen::Matrix3d getRotationMatrixFromYaw(double yaw)
    {
        // Create a 3x3 identity matrix
        Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();

        // Set the rotation matrix values based on the yaw (rotation around Z-axis)
        rotation_matrix(0, 0) = cos(yaw);
        rotation_matrix(0, 1) = -sin(yaw);
        rotation_matrix(1, 0) = sin(yaw);
        rotation_matrix(1, 1) = cos(yaw);

        // Z-axis remains unchanged
        rotation_matrix(2, 2) = 1.0;

        return rotation_matrix;
    }

    Eigen::Vector3d to_world_frame(const Eigen::Vector3d v,
                                   const float yaw){
        Eigen::Matrix3d rot = getRotationMatrixFromYaw(yaw);
        Eigen::Vector3d vw = rot * v;

        return vw;
    }

    Eigen::Vector3d propergate_motion_model(const Eigen::Vector3d&q,
                                            const Eigen::Vector3d&v,
                                            const float& dt){
        Eigen::Vector3d q_next = q + v * dt;

        return q_next;
    }

    /* VARIABLES */

    // Publisher for position and velocity of the base (x, y, theta).
    ros::Publisher ridgeback_joint_states_pub;
    ros::Subscriber ridgeback_odom_sub;

    // Store last received time and configuration
    bool new_tf_available;
    double t_tf;
    Eigen::Vector3d q_tf;

    // Store odom velocity bias (robot should be stationary) and latest estimated state info
    Eigen::Vector3d v_o;
    double t_prev;
    Eigen::Vector3d q_prev;
    Eigen::Vector3d q_curr;
    Eigen::Vector3d v_curr;
    FixedSizeVector<double> t_odom_hist;
    FixedSizeVector<Eigen::Vector3d> vb_odom_hist;
    FixedSizeVector<Eigen::Vector3d> q_hist;


    bool calibration_ready;
    bool initial_state_ready;


    // Exponential smoothing filters to remove noise from numerically
    // differentiated velocity.
    mm::ExponentialSmoother<double> angular_velocity_filter;
    mm::ExponentialSmoother<Eigen::Vector2d> linear_velocity_filter;

    // Number of messages received.
    uint32_t tf_msg_count = 0;
    uint32_t odom_msg_count = 0;


    // TF Listener to query transforms
    tf::TransformListener tf_listener_;

    // Frame IDs for the transform query
    std::string target_frame_id_;
    std::string source_frame_id_;
    std::string odom_topic_;

    // exponetial filter param
    double tau_linear;
    double tau_angular;

    ros::Subscriber ridgeback_vicon_sub;
    // tf broadcast
    tf::TransformBroadcaster tf_br;

};  // class RidgebackTfOdomEstimatorNode

int main(int argc, char** argv) {
    ros::init(argc, argv, "ridgeback_tf_estimator_node");
    ros::NodeHandle nh("~");

    RidgebackTfOdomEstimatorNode node;
    if (!node.start(nh)) {
        ROS_ERROR("Failed to start RidgebackTfOdomEstimatorNode.");
        return -1;
    }

    ros::spin();

    return 0;
}
