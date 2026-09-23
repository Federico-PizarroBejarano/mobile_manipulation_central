#include <gtest/gtest.h>

#include <mobile_manipulation_central/ridgeback_base_fusion.h>

using mm::RidgebackFusionParams;
using mm::RidgebackFusionState;
using mm::ViconTargetResult;
using mm::applyViconCorrection;
using mm::predictFromOdom;
using mm::setViconTarget;

namespace {

RidgebackFusionState seededState(const Eigen::Vector3d& q) {
    RidgebackFusionState state;
    state.q = q;
    state.q_vicon_target = q;
    state.has_vicon_target = true;
    state.t_last_vicon = 1.0;
    state.has_last_vicon = true;
    return state;
}

}  // namespace

TEST(RidgebackBaseFusion, RejectClearsTargetSoCorrectionCoasts) {
    RidgebackFusionParams params;
    params.max_vicon_innovation_m = 0.15;
    params.vicon_relock_consistent_count = 5;

    RidgebackFusionState state = seededState(Eigen::Vector3d(0.0, 0.0, 0.0));
    const Eigen::Vector3d q_outlier(0.5, 0.0, 0.0);

    const ViconTargetResult result =
        setViconTarget(state, q_outlier, 1.1, params);

    EXPECT_EQ(result, ViconTargetResult::RejectedInnovation);
    EXPECT_FALSE(state.has_vicon_target);

    const Eigen::Vector3d q_before = state.q;
    applyViconCorrection(state, 0.02, params);
    EXPECT_TRUE(state.q.isApprox(q_before));
}

TEST(RidgebackBaseFusion, MomentaryOutlierThenReturnAcceptsWithoutRelock) {
    RidgebackFusionParams params;
    params.max_vicon_innovation_m = 0.15;
    params.vicon_relock_consistent_count = 5;

    RidgebackFusionState state = seededState(Eigen::Vector3d(0.0, 0.0, 0.0));

    EXPECT_EQ(setViconTarget(state, Eigen::Vector3d(0.5, 0.0, 0.0), 1.1, params),
              ViconTargetResult::RejectedInnovation);
    EXPECT_FALSE(state.has_vicon_target);

    EXPECT_EQ(setViconTarget(state, Eigen::Vector3d(0.05, 0.0, 0.0), 1.2, params),
              ViconTargetResult::Accepted);
    EXPECT_TRUE(state.has_vicon_target);
    EXPECT_TRUE(state.q_vicon_target.isApprox(Eigen::Vector3d(0.05, 0.0, 0.0)));
}

TEST(RidgebackBaseFusion, ConsistentLagJumpRelocksTargetWithoutSnappingQ) {
    RidgebackFusionParams params;
    params.max_vicon_innovation_m = 0.15;
    params.max_vicon_jump_m = 0.05;
    params.vicon_relock_consistent_count = 3;

    RidgebackFusionState state = seededState(Eigen::Vector3d(0.0, 0.0, 0.0));
    const Eigen::Vector3d q_new(0.4, 0.0, 0.0);

    EXPECT_EQ(setViconTarget(state, q_new, 1.1, params),
              ViconTargetResult::RejectedInnovation);
    EXPECT_EQ(setViconTarget(state, q_new, 1.2, params),
              ViconTargetResult::RejectedInnovation);

    const Eigen::Vector3d q_before = state.q;
    EXPECT_EQ(setViconTarget(state, q_new, 1.3, params),
              ViconTargetResult::RelockedAfterCoast);
    EXPECT_TRUE(state.has_vicon_target);
    EXPECT_TRUE(state.q_vicon_target.isApprox(q_new));
    // State pose must not snap; catch-up is rate-limited separately.
    EXPECT_TRUE(state.q.isApprox(q_before));
}

TEST(RidgebackBaseFusion, UnstableOutliersResetRelockStreak) {
    RidgebackFusionParams params;
    params.max_vicon_innovation_m = 0.15;
    params.max_vicon_jump_m = 0.05;
    params.vicon_relock_consistent_count = 3;

    RidgebackFusionState state = seededState(Eigen::Vector3d(0.0, 0.0, 0.0));

    EXPECT_EQ(setViconTarget(state, Eigen::Vector3d(0.4, 0.0, 0.0), 1.1, params),
              ViconTargetResult::RejectedInnovation);
    EXPECT_EQ(setViconTarget(state, Eigen::Vector3d(0.4, 0.0, 0.0), 1.2, params),
              ViconTargetResult::RejectedInnovation);
    // Jump to a different outlier cluster — streak resets.
    EXPECT_EQ(setViconTarget(state, Eigen::Vector3d(0.0, 0.4, 0.0), 1.3, params),
              ViconTargetResult::RejectedInnovation);
    EXPECT_EQ(setViconTarget(state, Eigen::Vector3d(0.0, 0.4, 0.0), 1.4, params),
              ViconTargetResult::RejectedInnovation);
    EXPECT_FALSE(state.has_vicon_target);

    EXPECT_EQ(setViconTarget(state, Eigen::Vector3d(0.0, 0.4, 0.0), 1.5, params),
              ViconTargetResult::RelockedAfterCoast);
    EXPECT_TRUE(state.q_vicon_target.isApprox(Eigen::Vector3d(0.0, 0.4, 0.0)));
}

TEST(RidgebackBaseFusion, RelockThenCorrectionPullsTowardNewTarget) {
    RidgebackFusionParams params;
    params.max_vicon_innovation_m = 0.15;
    params.vicon_relock_consistent_count = 2;
    params.v_max_linear = 0.8;

    RidgebackFusionState state = seededState(Eigen::Vector3d(0.0, 0.0, 0.0));
    const Eigen::Vector3d q_new(0.4, 0.0, 0.0);

    setViconTarget(state, q_new, 1.1, params);
    setViconTarget(state, q_new, 1.2, params);
    ASSERT_TRUE(state.has_vicon_target);

    applyViconCorrection(state, 0.1, params);
    EXPECT_GT(state.q(0), 0.0);
    EXPECT_LT(state.q(0), q_new(0));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
