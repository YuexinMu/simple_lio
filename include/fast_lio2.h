//
// Created by myx on 2025/3/14.
//

#pragma once

#pragma once
#include "lio_base.h"

#include "ivox3d/ivox3d.h"
#include "common/common_lib.h"

namespace simple_lio{

//struct FastLio2Config{
//  // ivox params
//  float resolution;
//  int nearby_type;
//
//  // filter params
//  float map_filter_size;
//  bool extrinsic_est_en;
//  float plane_threshold;
//  int max_iteration;
//};
//
//class fast_lio2 : public LioBase{
//  //class FasterLio3{
//public:
//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//  using IVoxType = ivox3d::IVox<3, ivox3d::IVoxNodeType::DEFAULT, PointType>;
//
//  fast_lio2() = default;
//  ~fast_lio2();
//
//  bool Init(ros::NodeHandle &nh);
//  void Run();
//  // interface of mtk, customized obseravtion model
//  void ObsModel(state_ikfom &state_ikf,
//                esekfom::dyn_share_datastruct<double> &ekfom_data);
//
//private:
//  FasterLioConfig faster_lio_cfg_;
//
//
//  SopSE3 State2SE3(state_ikfom state);
//
//
//  // sync lidar with imu
//  void MapIncremental();
//
//  void SubAndPubToROS(ros::NodeHandle &nh);
//
//
//  // debug save / show
//  void PointBodyToWorld(PointType const *pi, PointType *const po);
//  void PointBodyToWorld(const Vec3f &pi, PointType *const po);
//  CloudPtr PointCloudLidarToIMU(CloudPtr &pi);
//
//private:
//  esekfom::esekf<state_ikfom, 12, input_ikfom> kf_; // esekf
//  state_ikfom state_point_;
//
//  // nearest neighbor type select
//  IVoxType::Options ivox_options_;
//  std::shared_ptr<IVoxType> ivox_ = nullptr;
//
//  /// point clouds data
//  pcl::VoxelGrid<PointType> voxel_scan_;    // voxel filter for current scan
//  CloudPtr scan_undistort_{new PointCloudType()};  // scan after undistortion
//  CloudPtr scan_down_body_{new PointCloudType()};  // downsampled scan in body
//  CloudPtr scan_down_world_{new PointCloudType()}; // downsampled scan in world
//  std::vector<PointVector> nearest_points_; // nearest points of current scan
//  VVec4F corr_pts_;                         // inlier pts
//  VVec4F corr_norm_;                        // inlier plane norms
//  std::vector<float> residuals_;            // point-to-plane residuals
//  std::vector<bool> point_selected_surf_;   // selected points
//  VVec4F plane_coef_;                       // plane coeffs
//
//  double first_lidar_time_ = 0.0;
//
//  /// statistics and flags ///
//  bool flg_first_scan_ = true;
//  bool flg_EKF_inited_ = false;
//  int effect_feat_num_ = 0;
//
//  size_t frame_id_ = 0;
//};

} // namespace simple_lio