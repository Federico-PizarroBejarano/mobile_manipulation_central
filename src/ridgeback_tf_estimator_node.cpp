#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <tf/transform_listener.h>
#include <Eigen/Eigen>
#include <mobile_manipulation_central/exponential_smoothing.h>
#include <mobile_manipulation_central/wrap.h>

// The node listens to tf transforms for the Ridgeback base and
// converts it into a JointState message which includes a numerically
// differentiated and filtered velocity estimate on (x, y, yaw).
class RidgebackTfEstimatorNode {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RidgebackTfEstimatorNode() {}

    // Start the node.
    bool start(ros::NodeHandle& nh) {
        // Retrieve parameters, including the frame id with a default value
        nh.param<std::string>("target_frame_id", target_frame_id_, "base_link");
        nh.param<std::string>("source_frame_id", source_frame_id_, "world");
        nh.param<double>("tau_linear", tau_linear, 0.045);
        nh.param<double>("tau_angular", tau_angular, 0.025);
        ridgeback_joint_states_pub =
            nh.advertise<sensor_msgs::JointState>("/ridgeback/joint_states", 1);

        // Velocity is assumed to be 0 initially. Values for tau taken from
        // dsl__estimation__vicon package.
        linear_velocity_filter.init(tau_linear, Eigen::Vector2d::Zero());
        angular_velocity_filter.init(tau_angular, 0);

        return true;
    }

    void update() {
        // Listen for the tf transform between the source and target frames
        tf::StampedTransform transform;
        try {
            tf_listener_.lookupTransform(source_frame_id_, target_frame_id_, ros::Time(0), transform);
        } catch (tf::TransformException& ex) {
            ROS_WARN("%s", ex.what());
            return;
        }

        // Get the current time and joint configuration
        double t = ros::Time::now().toSec();
        Eigen::Vector3d q;
        q << transform.getOrigin().x(), transform.getOrigin().y(), tf::getYaw(transform.getRotation());

        // Wait until we have at least two messages so we can numerically differentiate.
        if (msg_count >= 2) {
            double dt = t - t_prev;

            // Compute measured velocity via numerical differentiation
            Eigen::Vector3d delta = q - q_prev;
            delta(2) = mm::wrap_to_pi(delta(2));
            Eigen::Vector3d v_measured = delta / dt;

            // Filter velocity
            Eigen::Vector3d v_filtered;
            v_filtered << linear_velocity_filter.next(v_measured.head(2), dt),
                angular_velocity_filter.next(v_measured(2), dt);

            // Publish the joint states
            publish_ridgeback_joint_states(q, v_filtered);
        }

        // Store the current values for the next iteration
        t_prev = t;
        q_prev = q;
        ++msg_count;
    }

   private:
    /* FUNCTIONS */

    void publish_ridgeback_joint_states(const Eigen::Vector3d& q,
                                        const Eigen::Vector3d& v) {
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time::now();
        msg.name = {"x", "y", "yaw"};

        for (int i = 0; i < 3; ++i) {
            msg.position.push_back(q(i));
            msg.velocity.push_back(v(i));
        }

        ridgeback_joint_states_pub.publish(msg);
    }

    /* VARIABLES */

    // Publisher for position and velocity of the base (x, y, theta).
    ros::Publisher ridgeback_joint_states_pub;

    // Store last received time and configuration for numerical differentiation
    double t_prev;
    Eigen::Vector3d q_prev;

    // Exponential smoothing filters to remove noise from numerically
    // differentiated velocity.
    mm::ExponentialSmoother<double> angular_velocity_filter;
    mm::ExponentialSmoother<Eigen::Vector2d> linear_velocity_filter;

    // Number of messages received.
    uint32_t msg_count = 0;

    // TF Listener to query transforms
    tf::TransformListener tf_listener_;

    // Frame IDs for the transform query
    std::string target_frame_id_;
    std::string source_frame_id_;

    // exponetial filter param
    double tau_linear;
    double tau_angular;

};  // class RidgebackTfEstimatorNode

int main(int argc, char** argv) {
    ros::init(argc, argv, "ridgeback_tf_estimator_node");
    ros::NodeHandle nh("~");

    RidgebackTfEstimatorNode node;
    if (!node.start(nh)) {
        ROS_ERROR("Failed to start RidgebackTfEstimatorNode.");
        return -1;
    }

    int update_rate;
    nh.param<int>("update_rate", update_rate, 10);

    ros::Rate rate(update_rate);  // Adjust the rate as per your requirements
    while (ros::ok()) {
        node.update();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
