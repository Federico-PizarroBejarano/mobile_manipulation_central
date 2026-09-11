#include <ros/console.h>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>

#include <mobile_manipulation_central/exponential_smoothing.h>
#include <mobile_manipulation_central/fixed_size_vector.h>
#include <mobile_manipulation_central/ridgeback_base_fusion.h>

class RidgebackTfOdomEstimatorNode {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RidgebackTfOdomEstimatorNode()
        : vb_odom_hist(40), tau_linear(0.08), tau_angular(0.08) {}

    bool start(ros::NodeHandle& nh) {
        nh.param<std::string>("world_frame_id", world_frame_id_, "my_world");
        nh.param<std::string>("vicon_base_frame_id", vicon_base_frame_id_,
                              "vicon_base_link");
        nh.param<std::string>("odom_topic", odom_topic_, "/odometry/filtered");
        nh.param<double>("tau_linear", tau_linear, 0.08);
        nh.param<double>("tau_angular", tau_angular, 0.08);

        nh.param("v_max_linear", fusion_params_.v_max_linear, 0.8);
        nh.param("v_max_angular", fusion_params_.v_max_angular, 0.5);
        nh.param("max_vicon_innovation_m", fusion_params_.max_vicon_innovation_m,
                 0.15);
        nh.param("max_vicon_yaw_innovation_rad",
                 fusion_params_.max_vicon_yaw_innovation_rad, 0.5);

        std::string base_vicon_topic;
        nh.param<std::string>("base_vicon_topic", base_vicon_topic,
                              "/vicon/ThingBase_Fed/ThingBase_Fed");

        ridgeback_vicon_sub = nh.subscribe(
            base_vicon_topic, 1,
            &RidgebackTfOdomEstimatorNode::vicon_cb, this);

        ridgeback_joint_states_pub =
            nh.advertise<sensor_msgs::JointState>("/ridgeback/joint_states", 1);
        ridgeback_odom_sub = nh.subscribe(
            odom_topic_, 1,
            &RidgebackTfOdomEstimatorNode::ridgeback_odom_cb, this);

        linear_velocity_filter.init(tau_linear, Eigen::Vector2d::Zero());
        angular_velocity_filter.init(tau_angular, 0);

        initial_state_ready = false;
        calibration_ready = false;
        new_tf_available = false;
        return true;
    }

    bool calculate_odom_bias() {
        if (odom_msg_count > 50) {
            v_o = vb_odom_hist.avg();
            ROS_INFO_STREAM(
                "TF_ODOM_ESTIMATOR: Calibrated odom velocity bias: " << v_o);
            return true;
        }
        return false;
    }

    void vicon_cb(const geometry_msgs::TransformStamped& msg) {
        const double t = msg.header.stamp.toSec();
        q_tf << msg.transform.translation.x, msg.transform.translation.y,
            tf::getYaw(msg.transform.rotation);

        if (tf_msg_count == 0 || t - t_tf > 0.0) {
            t_tf = t;
            new_tf_available = true;
            ++tf_msg_count;
        } else {
            new_tf_available = false;
        }

        tf::StampedTransform transform;
        tf::transformStampedMsgToTF(msg, transform);
        transform.frame_id_ = world_frame_id_;
        transform.child_frame_id_ = vicon_base_frame_id_;
        transform.getOrigin().setZ(0.0);
        tf_br.sendTransform(transform);
    }

    void ridgeback_odom_cb(const nav_msgs::Odometry& msg) {
        if (!initial_state_ready && new_tf_available) {
            fusion_state_.q = q_tf;
            fusion_state_.v.setZero();
            fusion_state_.q_vicon_target = q_tf;
            fusion_state_.has_vicon_target = true;
            fusion_state_.t_last_vicon = t_tf;
            fusion_state_.has_last_vicon = true;
            t_prev = t_tf;
            initial_state_ready = true;
            ROS_INFO_STREAM("TF_ODOM_ESTIMATOR: Initialized with first TF");
        }

        if (!initial_state_ready) {
            return;
        }

        const ros::Time msg_stamp = msg.header.stamp;
        const double t = msg_stamp.toSec();
        Eigen::Vector3d vb(msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                           msg.twist.twist.angular.z);

        double dt = 0.0;
        if (odom_msg_count >= 1) {
            dt = t - t_prev;
        }

        if (odom_msg_count >= 2 && dt > 0.0) {
            Eigen::Vector3d vb_filtered;
            vb_filtered << linear_velocity_filter.next(vb.head(2), dt),
                angular_velocity_filter.next(vb(2), dt);
            vb = vb_filtered;
        }

        vb_odom_hist.add(vb);
        ++odom_msg_count;

        if (!calibration_ready) {
            calibration_ready = calculate_odom_bias();
            if (calibration_ready) {
                ROS_INFO_STREAM("TF_ODOM_ESTIMATOR: Calibration ready");
            }
            t_prev = t;
            return;
        }

        if (new_tf_available) {
            if (t - t_tf > 0.3) {
                ROS_WARN_STREAM("TF delayed by " << t - t_tf << " s");
            }

            const mm::ViconTargetResult target_result =
                mm::setViconTarget(fusion_state_, q_tf, t_tf, fusion_params_);

            if (target_result == mm::ViconTargetResult::RejectedInnovation) {
                ROS_WARN_STREAM_THROTTLE(
                    1.0,
                    "TF_ODOM_ESTIMATOR: rejected Vicon target (innovation "
                    "too large)");
            }

            new_tf_available = false;
        }

        if (dt > 0.0) {
            const Eigen::Vector3d vb_body = vb - v_o;
            mm::predictFromOdom(fusion_state_, vb_body, dt);
            mm::applyViconCorrection(fusion_state_, dt, fusion_params_);
            mm::clampVelocity(fusion_state_, fusion_params_);
        }

        publish_ridgeback_joint_states(msg_stamp, fusion_state_.q,
                                       fusion_state_.v);
        t_prev = t;
    }

   private:
    void publish_ridgeback_joint_states(const ros::Time& t,
                                        const Eigen::Vector3d& q,
                                        const Eigen::Vector3d& v) {
        sensor_msgs::JointState msg;
        msg.header.stamp = t;
        msg.name = {"x", "y", "yaw"};

        for (int i = 0; i < 3; ++i) {
            msg.position.push_back(q(i));
            msg.velocity.push_back(v(i));
        }

        ridgeback_joint_states_pub.publish(msg);
    }

    ros::Publisher ridgeback_joint_states_pub;
    ros::Subscriber ridgeback_odom_sub;
    ros::Subscriber ridgeback_vicon_sub;
    tf::TransformBroadcaster tf_br;

    bool new_tf_available = false;
    double t_tf = 0.0;
    Eigen::Vector3d q_tf = Eigen::Vector3d::Zero();

    Eigen::Vector3d v_o = Eigen::Vector3d::Zero();
    double t_prev = 0.0;
    mm::RidgebackFusionState fusion_state_;
    mm::RidgebackFusionParams fusion_params_;

    FixedSizeVector<Eigen::Vector3d> vb_odom_hist;

    bool calibration_ready = false;
    bool initial_state_ready = false;

    uint32_t tf_msg_count = 0;
    uint32_t odom_msg_count = 0;

    std::string world_frame_id_;
    std::string vicon_base_frame_id_;
    std::string odom_topic_;

    mm::ExponentialSmoother<double> angular_velocity_filter;
    mm::ExponentialSmoother<Eigen::Vector2d> linear_velocity_filter;

    double tau_linear;
    double tau_angular;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ridgeback_tf_odom_estimator_node");
    ros::NodeHandle nh("~");

    RidgebackTfOdomEstimatorNode node;
    if (!node.start(nh)) {
        ROS_ERROR("Failed to start RidgebackTfOdomEstimatorNode.");
        return -1;
    }

    ros::spin();
    return 0;
}
