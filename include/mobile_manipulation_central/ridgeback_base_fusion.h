#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Eigen>

#include <mobile_manipulation_central/wrap.h>

namespace mm {

struct RidgebackFusionParams {
    double v_max_linear = 0.8;
    double v_max_angular = 0.5;
    double max_vicon_innovation_m = 0.15;
    double max_vicon_yaw_innovation_rad = 0.5;
    // Frame-to-frame Vicon motion allowed while counting a lag→jump cluster.
    double max_vicon_jump_m = 0.05;
    double max_vicon_yaw_jump_rad = 0.1;
    // Consecutive large-but-stable Vicon samples before accepting a new target.
    int vicon_relock_consistent_count = 5;
    double min_vicon_dt = 1e-4;
};

struct RidgebackFusionState {
    Eigen::Vector3d q = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d q_vicon_target = Eigen::Vector3d::Zero();
    bool has_vicon_target = false;
    double t_last_vicon = 0.0;
    bool has_last_vicon = false;
    // Coasting / lag→jump confirmation state.
    Eigen::Vector3d q_vicon_candidate = Eigen::Vector3d::Zero();
    bool has_vicon_candidate = false;
    int vicon_outlier_streak = 0;
};

inline Eigen::Matrix2d rotation2dFromYaw(double yaw) {
    Eigen::Matrix2d rot;
    rot << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
    return rot;
}

inline Eigen::Vector3d bodyTwistToWorld(const Eigen::Vector3d& vb_body,
                                        double yaw) {
    Eigen::Vector2d vw_xy = rotation2dFromYaw(yaw) * vb_body.head<2>();
    return Eigen::Vector3d(vw_xy(0), vw_xy(1), vb_body(2));
}

inline Eigen::Vector3d positionError(const Eigen::Vector3d& q_target,
                                    const Eigen::Vector3d& q) {
    Eigen::Vector3d err = q_target - q;
    err(2) = wrap_to_pi(err(2));
    return err;
}

inline void clampVectorNorm(Eigen::Vector2d& vec, double max_norm) {
    const double norm = vec.norm();
    if (norm > max_norm && norm > 0.0) {
        vec *= max_norm / norm;
    }
}

inline bool withinGate(const Eigen::Vector3d& err,
                       double max_xy,
                       double max_yaw) {
    return err.head<2>().norm() <= max_xy && std::abs(err(2)) <= max_yaw;
}

enum class ViconTargetResult {
    Accepted,
    RejectedInnovation,
    RelockedAfterCoast,
};

inline void clearViconCandidate(RidgebackFusionState& state) {
    state.has_vicon_candidate = false;
    state.vicon_outlier_streak = 0;
}

inline void acceptViconTarget(RidgebackFusionState& state,
                              const Eigen::Vector3d& q_vicon,
                              double t_vicon) {
    state.q_vicon_target = q_vicon;
    state.has_vicon_target = true;
    state.t_last_vicon = t_vicon;
    state.has_last_vicon = true;
    clearViconCandidate(state);
}

// Accept Vicon as the position target when the innovation is within gate.
// Large jumps clear the target so the filter coasts on odom. If Vicon then
// stays at a new pose for vicon_relock_consistent_count samples, accept that
// target (without snapping q); catch-up is rate-limited in applyViconCorrection.
inline ViconTargetResult setViconTarget(RidgebackFusionState& state,
                                        const Eigen::Vector3d& q_vicon,
                                        double t_vicon,
                                        const RidgebackFusionParams& params) {
    if (state.has_last_vicon) {
        const double dt_vicon = t_vicon - state.t_last_vicon;
        if (dt_vicon >= params.min_vicon_dt) {
            const Eigen::Vector3d innov = positionError(q_vicon, state.q);
            const bool innovation_ok = withinGate(
                innov, params.max_vicon_innovation_m,
                params.max_vicon_yaw_innovation_rad);

            if (!innovation_ok) {
                // Coast: do not pull toward a stale target.
                state.has_vicon_target = false;
                state.t_last_vicon = t_vicon;
                state.has_last_vicon = true;

                const bool same_cluster =
                    state.has_vicon_candidate &&
                    withinGate(positionError(q_vicon, state.q_vicon_candidate),
                               params.max_vicon_jump_m,
                               params.max_vicon_yaw_jump_rad);

                if (same_cluster) {
                    state.q_vicon_candidate = q_vicon;
                    ++state.vicon_outlier_streak;
                } else {
                    state.q_vicon_candidate = q_vicon;
                    state.has_vicon_candidate = true;
                    state.vicon_outlier_streak = 1;
                }

                if (state.vicon_outlier_streak >=
                    params.vicon_relock_consistent_count) {
                    acceptViconTarget(state, q_vicon, t_vicon);
                    return ViconTargetResult::RelockedAfterCoast;
                }
                return ViconTargetResult::RejectedInnovation;
            }
        }
    }

    acceptViconTarget(state, q_vicon, t_vicon);
    return ViconTargetResult::Accepted;
}

inline void predictFromOdom(RidgebackFusionState& state,
                            const Eigen::Vector3d& vb_body,
                            double dt) {
    if (dt <= 0.0) {
        return;
    }

    const double yaw = state.q(2);
    state.v = bodyTwistToWorld(vb_body, yaw);
    state.q += state.v * dt;
    state.q(2) = wrap_to_pi(state.q(2));
}

inline void applyViconCorrection(RidgebackFusionState& state,
                                 double dt_odom,
                                 const RidgebackFusionParams& params) {
    if (!state.has_vicon_target || dt_odom <= 0.0) {
        return;
    }

    Eigen::Vector3d err =
        positionError(state.q_vicon_target, state.q);

    Eigen::Vector2d step_xy = err.head<2>();
    clampVectorNorm(step_xy, params.v_max_linear * dt_odom);

    const double max_yaw_step = params.v_max_angular * dt_odom;
    double step_yaw =
        std::max(-max_yaw_step, std::min(max_yaw_step, err(2)));

    state.q(0) += step_xy(0);
    state.q(1) += step_xy(1);
    state.q(2) = wrap_to_pi(state.q(2) + step_yaw);
}

inline void clampVelocity(RidgebackFusionState& state,
                          const RidgebackFusionParams& params) {
    Eigen::Vector2d v_xy = state.v.head<2>();
    clampVectorNorm(v_xy, params.v_max_linear);
    state.v(0) = v_xy(0);
    state.v(1) = v_xy(1);
    state.v(2) = std::max(-params.v_max_angular,
                          std::min(params.v_max_angular, state.v(2)));
}

}  // namespace mm
