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
#include "ikd-Tree/ikd_Tree.hpp"

#include "lio_base.h"

namespace simple_lio {

class fast_lio2 : public LioBase{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  fast_lio2();
  ~fast_lio2();

  // init with ros
  bool InitROS(ros::NodeHandle &nh);

  void Run();

  // interface of mtk, customized obseravtion model
  void ObsModel(state_ikfom &state_ikf, esekfom::dyn_share_datastruct<double> &ekfom_data);

  // debug save / show
  void PublishPath(const ros::Publisher& pub_path);
  void PublishOdometry(const ros::Publisher &pub_odom_aft_mapped);
  void PublishFrameWorld();
  void PublishFrameBody();
  void PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world);
  void SaveTrajectory(const std::string &traj_file);

  void Finish();

private:
  template <typename T>
  void SetPoseStamp(T &out);

  void PointBodyToWorld(PointType const *pi, PointType *const po);
  void PointBodyToWorld(const Vec3f &pi, PointType *const po);
  void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);

  void MapIncremental();

  void SubAndPubToROS(ros::NodeHandle &nh);

  bool InitParamsFromROS(ros::NodeHandle &nh);

private:
  float ESTI_PLANE_THRESHOLD_ = 0.1;
  int NUM_MAX_ITERATIONS_ = 3;

  KD_TREE<PointType> ikd_tree_;

  std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
  std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

  /// local map related
  float det_range_ = 300.0f;
  double cube_len_ = 0;

  float filter_size_surf_min_ = 0.0;
  float filter_size_map_min_ = 0.0;

  /// params
  std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation
  std::vector<double> extrinR_{9, 0.0};  // lidar-imu rotation
  std::string map_file_path_;

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

  /// ros pub and sub stuffs
  ros::Subscriber sub_pcl_;
  ros::Subscriber sub_imu_;
  ros::Publisher pub_laser_cloud_world_;
  ros::Publisher pub_laser_cloud_body_;
  ros::Publisher pub_laser_cloud_effect_world_;
  ros::Publisher pub_match_cloud_world_;
  ros::Publisher pub_odom_aft_mapped_;
  ros::Publisher pub_path_;
  std::string tf_imu_frame_;
  std::string tf_world_frame_;

  std::mutex mtx_buffer_;
  std::deque<double> time_buffer_;
  std::deque<PointCloudType::Ptr> lidar_buffer_;
  std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
  nav_msgs::Odometry odom_aft_mapped_;

  /// options
  bool time_sync_en_ = false;
  double timediff_lidar_wrt_imu_ = 0.0;
  double last_timestamp_lidar_ = 0;
  double lidar_end_time_ = 0;
  double last_timestamp_imu_ = -1.0;
  double first_lidar_time_ = 0.0;
  bool lidar_pushed_ = false;

  /// statistics and flags ///
  int scan_count_ = 0;
  int publish_count_ = 0;
  bool flg_EKF_inited_ = false;
  double lidar_mean_scantime_ = 0.0;
  int scan_num_ = 0;
  bool timediff_set_flg_ = false;
  int effect_feat_num_ = 0, frame_num_ = 0;

  ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
  MeasureGroup measures_;                    // sync IMU and lidar scan
  esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
  state_ikfom state_point_;                          // ekf current state

  bool extrinsic_est_en_ = true;

  /////////////////////////  debug show / save /////////////////////////////////////////////////////////
  bool run_in_offline_ = false;
  bool path_pub_en_ = true;
  bool scan_pub_en_ = false;
  bool dense_pub_en_ = false;
  bool scan_body_pub_en_ = false;
  bool scan_effect_pub_en_ = false;
  bool pcd_save_en_ = false;
  bool runtime_pos_log_ = true;
  int pcd_save_interval_ = -1;
  bool traj_save_en_ = false;

  PointCloudType::Ptr pcl_wait_save_{new PointCloudType()};  // debug save
  nav_msgs::Path path_;
  geometry_msgs::PoseStamped msg_body_pose_;

};

}  // namespace simple_lio
