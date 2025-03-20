//
// Created by myx on 2024/8/17.
//

#pragma once

#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <condition_variable>
#include <thread>

#include "common/imu_processing.hpp"
#include "ivox3d/ivox3d.h"
#include "common/pointcloud_preprocess.h"
#include "common/use-ikfom.hpp"
#include <ikd-Tree/ikd_Tree.hpp>

namespace simple_lio {
enum nearest_neighbor_type{
  IKD_TREE,
  IVOX
};

struct SimpleLioConfig {
  // common params
  std::string lid_topic;
  std::string imu_topic;
  bool time_sync_en = false;

  // lidar params
  int lidar_type;
  int scan_line;
  double blind;
  float time_scale;

  // imu params
  double acc_cov;
  double gyr_cov;
  double b_acc_cov;
  double b_gyr_cov;

  // lidar-imu params
  std::vector<double> extrinsic_T{3, 0.0};
  std::vector<double> extrinsic_R{9, 0.0};

  // preprocess params
  int point_filter_num;
  float filter_size_surf;
  bool feature_extract_enable;

  // filter params
  float filter_size_map;
  bool extrinsic_est_en;
  float plane_threshold;
  int max_iteration;

  // frame info params
  std::string body_frame;
  std::string init_frame;
  std::string odom_topic;
  std::string path_topic;
  std::string cloud_world_topic;
  std::string cloud_imu_topic;

  int nn_type;

  float resolution;
  int nearby_type;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class lio{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  using IVoxType = ivox3d::IVox<3, ivox3d::IVoxNodeType::DEFAULT, PointType>;

  lio();
  ~lio();

  // init with ros
  bool Init(ros::NodeHandle &nh);

  void Run();

  // callbacks of lidar and imu
  void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);

  // sync lidar with imu
  bool SyncPackages();

  void ObsModel(state_ikfom &state_ikf, esekfom::dyn_share_datastruct<double> &ekfom_data);
  void Finish();

  // debug save / show
  void PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world);

private:
  template <typename T>
  void SetPoseStamp(SopSE3 pose, T &out);
  void PublishPath(const SopSE3& pose);
  void PublishOdom(const SopSE3& pose);
  void PublishPointCloud(const CloudPtr& cloud, std::string frame_id,
                         const ros::Publisher& pub);

  SopSE3 State2SE3(state_ikfom state);

  void PointBodyToWorld(PointType const *pi, PointType *const po);
  void PointBodyToWorld(const Vec3f &pi, PointType *const po);
//  void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);
  CloudPtr PointCloudLidarToIMU(CloudPtr &pi);

  void MapIncremental();

  void SubAndPubToROS(ros::NodeHandle &nh);

  bool LoadParams(ros::NodeHandle &nh);
  bool LoadParamsFromYAML(const std::string &yaml);

  void PrintState(const state_ikfom &s);

private:
  SimpleLioConfig config_;

  // nearest neighbor type select
  nearest_neighbor_type nn_type_ = nearest_neighbor_type::IVOX;

  IVoxType::Options ivox_options_;
  std::shared_ptr<IVoxType> ivox_ = nullptr;

  KD_TREE<PointType> ikd_tree_;

  std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
  std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

  /// point clouds data
  CloudPtr scan_undistort_{new PointCloudType()};   // scan after undistortion
  CloudPtr scan_down_body_{new PointCloudType()};   // downsampled scan in body
  CloudPtr scan_down_world_{new PointCloudType()};  // downsampled scan in world
  std::vector<PointVector> nearest_points_;         // nearest points of current scan
  VVec4F corr_pts_;                           // inlier pts
  VVec4F corr_norm_;                          // inlier plane norms
  pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan
  std::vector<float> residuals_;                    // point-to-plane residuals
  std::vector<bool> point_selected_surf_;           // selected points
  VVec4F plane_coef_;                         // plane coeffs

  ros::Publisher pub_odom_;
  ros::Publisher pub_path_;
  ros::Publisher pub_laser_cloud_world_;
  ros::Publisher pub_laser_cloud_imu_;

  nav_msgs::Odometry odometry_;
  nav_msgs::Path path_;

  std::mutex mtx_buffer_;
  std::deque<double> time_buffer_;
  std::deque<PointCloudType::Ptr> lidar_buffer_;
  std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;

  /// options
  double timediff_lidar_wrt_imu_ = 0.0;
  double last_timestamp_lidar_ = 0;
  double lidar_end_time_ = 0;
  double last_timestamp_imu_ = -1.0;
  double first_lidar_time_ = 0.0;
  bool lidar_pushed_ = false;

  /// statistics and flags ///
  int scan_count_ = 0;
  int publish_count_ = 0;
  bool flg_first_scan_ = true;
  bool flg_EKF_inited_ = false;
  double lidar_mean_scantime_ = 0.0;
  int scan_num_ = 0;
  bool timediff_set_flg_ = false;
  int effect_feat_num_ = 0, frame_num_ = 0;

  ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
  MeasureGroup measures_;                    // sync IMU and lidar scan
  esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
  state_ikfom state_point_;                          // ekf current state
  vect3 pos_lidar_;                                  // lidar position after eskf update
  bool extrinsic_est_en_ = true;
};

}  // namespace simple_lio
