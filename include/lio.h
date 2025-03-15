//
// Created by myx on 2024/8/17.
//

#pragma once

#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <condition_variable>
#include <thread>

#include "common/imu_processing.hpp"
#include "ivox3d/ivox3d.h"
#include "common/pointcloud_preprocess.h"
#include "common/common_lib.h"
#include "ikd-Tree/ikd_Tree.hpp"
#include "common/use-ikfom.hpp"

namespace simple_lio {
enum nearest_neighbor_type{
  IKD_TREE,
  IVOX
};


class lio_test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  using IVoxType = ivox3d::IVox<3, ivox3d::IVoxNodeType::DEFAULT, PointType>;

  lio_test();
  ~lio_test();

  // init with ros
  bool InitROS(ros::NodeHandle &nh);
  void Run();
  void Finish();

  // interface of mtk, customized obseravtion model
  void ObsModel(state_ikfom &state_ikf, esekfom::dyn_share_datastruct<double> &ekfom_data);

private:
  template <typename T>
  void SetPoseStamp(T &out);
  SopSE3 State2SE3(state_ikfom state);

  // callbacks of lidar and imu
  void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);


  // sync lidar with imu
  bool SyncPackages();
  void MapIncremental();

  void SubAndPubToROS(ros::NodeHandle &nh);
  bool InitParamsFromROS(ros::NodeHandle &nh);

  // debug save / show
  void PointBodyToWorld(PointType const *pi, PointType *const po);
  void PointBodyToWorld(const Vec3f &pi, PointType *const po);
  void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);
  void SopSE3ToTransOrientation(SopSE3 &se_3, Vec3d &trans, Vec4d &ori);
  void SopSE3ToPose(SopSE3 se_3, geometry_msgs::Pose &pose);

  // publish data
  void PublishPath();
  void PublishOptPath();
  void PublishOdom();
  void PublishPointCloudWorld();
  void PublishPointCloudBody();
  void PublishRosInfo();
  void SavePathTum(const std::string &traj_file, nav_msgs::Path path);


private:
  float ESTI_PLANE_THRESHOLD_ = 0.1;
  int NUM_MAX_ITERATIONS_ = 3;

  /// nearest neighbor type select
  nearest_neighbor_type nn_type_ = nearest_neighbor_type::IVOX;
  IVoxType::Options ivox_options_;
  std::shared_ptr<IVoxType> ivox_ = nullptr;
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
  ros::Subscriber sub_rtk_;
  ros::WallTimer loop_wall_timer_;
  ros::WallTimer graph_wall_timer_;
  ros::Publisher pub_laser_cloud_world_;
  ros::Publisher pub_laser_cloud_body_;
  ros::Publisher cur_loop_map_;
  ros::Publisher pre_loop_map_;
  ros::Publisher pub_odom_;
  ros::Publisher pub_path_;
  ros::Publisher pub_opt_path_;
  ros::Publisher pub_rtk_path_;
  ros::Publisher pub_loop_;
  nav_msgs::Odometry odometry_;
  nav_msgs::Path path_;
  nav_msgs::Path loop_path_;
  nav_msgs::Path opt_path_;

  std::string imu_frame_id_;
  std::string world_frame_id_;

  std::mutex mtx_buffer_;
  std::deque<double> time_buffer_;
  std::deque<PointCloudType::Ptr> lidar_buffer_;
  std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;

  /// options
  bool time_sync_en_ = false;
  double timediff_lidar_wrt_imu_ = 0.0;
  double last_timestamp_lidar_ = 0;
  double lidar_end_time_ = 0;
  double last_timestamp_imu_ = -1.0;
  double first_lidar_time_ = 0.0;
  bool lidar_pushed_ = false;

  /// statistics and flags ///
  int publish_count_ = 0;
  bool flg_first_scan_ = true;
  bool flg_EKF_inited_ = false;
  int pcd_index_ = 0;
  double lidar_mean_scantime_ = 0.0;
  int scan_num_ = 0;
  bool timediff_set_flg_ = false;
  int effect_feat_num_ = 0;

  ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
  MeasureGroup measures_;                    // sync IMU and lidar scan
  esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
  state_ikfom state_point_;                          // ekf current state
  bool extrinsic_est_en_ = true;

  /////////////////////////  debug show / save /////////////////////////////////////////////////////////
  bool output_debug_en_ = false;
  bool scan_pub_en_ = false;
  bool dense_pub_en_ = false;
  bool scan_body_pub_en_ = false;
  bool pcd_save_en_ = false;
  int pcd_save_interval_ = -1;
  bool traj_save_en_ = false;
  std::string dataset_;

  PointCloudType::Ptr pcl_wait_save_{new PointCloudType()};  // debug save
  geometry_msgs::PoseStamped msg_body_pose_;



  ///////////////////////// loop closure ///////////////////////////
  SopSE3 last_pose_;

  std::vector<PoseTime> opt_pose_time_;
  IdType frame_id_ = 0;
  IdType opt_pose_size_ = 0;
  bool opt_pose_update_ = false;
  std::mutex temp_kf_queue_mtx_;
  std::mutex kf_buf_mtx_;
  std::mutex opt_pose_mtx_;


  bool lp_closure_enable_ = true;
  bool lp_sparse_enable_ = true;
  double lp_closure_frequency_ = 50.0;
  // graph optimization iter num
  int graph_iter_num_ = 50;
  // seconds, history_time_threshold
  double history_time_th_ = 60.0;
  // meters, the distance between too keyframe
  double lp_dis_th_ = 20.0;
  double lp_delta_trans_ = 5.0;
  double lp_delta_degree_ = 45.0;
  // the score of between two scan-map by ndt align
  double lp_ndt_trans_score_th_ = 5.0;
  // the fitness score of between two scan-map by ndt align
  double lp_ndt_fitness_score_th = 1.0;
  // the keyframe number that add to sub_map
  int kf_num_sub_map_ = 10;
  // 被选为候选的两个关键帧之间的ID差值
  int min_id_th_ = 20;
  // 如果选择了一个候选帧，那么隔开多少个ID之后再选一个
  int skip_id_ = 5;
  SopSE3 last_loop_pose_;

  // loop observe nose, meter deg
  double lp_pos_noise_ = 0.01, lp_ang_noise_ = 0.05;
  double lidar_pos_noise_ = 0.01, lidar_ang_noise_ = 0.1;
  // set the window size of the error.
  // A squared error above delta^2 is considered as outlier in the data.
  double lp_robust_kernel_th_ = 4.4;

  SopSE3 pcd_cur_pose_;
};

}  // namespace faster_lio
