//
// Created by myx on 2025/3/6.
//

#pragma once
#include <string>

#include <livox_ros_driver/CustomMsg.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <condition_variable>
#include <thread>
#include <utility>
#include <nav_msgs/Path.h>


#include "common/imu_processing.hpp"
#include "common/pointcloud_preprocess.h"
#include "common/common_lib.h"

#include "ivox3d/ivox3d.h"

namespace simple_lio{

struct LIOOutput {
  LIOOutput() = default;
  LIOOutput(size_t i, double t, const SopSE3& p, CloudPtr sc_imu, CloudPtr sc_w)
      :id(i), timestamp(t), pose(p), scan_imu(std::move(sc_imu)),
        scan_world(std::move(sc_w)){};

  size_t id = 0;
  double timestamp = 0.0;
  SopSE3 pose;
  CloudPtr scan_imu = nullptr;
  CloudPtr scan_world = nullptr;
};

struct LIOConfig {
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

  // frame info params
  std::string body_frame;
  std::string init_frame;
  std::string odom_topic;
  std::string path_topic;
  std::string cloud_world_topic;
  std::string cloud_imu_topic;
};


class LioBase{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LioBase() = default;

  /// init
  virtual bool Init(ros::NodeHandle &nh);

  virtual void Run()  = 0;

  // define the LIO callback function
  using LIOProcessFuncType = std::function<void(const LIOOutput &)>;

  LioBase &SetLIOProcessFunc(LIOProcessFuncType lio_proc){
    lio_process_func_ = std::move(lio_proc);
    return *this;
  }

private:
  bool LoadParams(ros::NodeHandle &nh);

  std::mutex mtx_buffer_;
  std::deque<double> time_buffer_;
  std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
  std::deque<PointCloudType::Ptr> lidar_buffer_;

  int scan_num_ = 0;

  double lidar_mean_scantime_ = 0.0;
  double last_timestamp_imu_ = -1.0;
  double last_timestamp_lidar_ = 0;
  bool timediff_set_flg_ = false;
  double timediff_lidar_wrt_imu_ = 0.0;
  bool lidar_pushed_ = false;

protected:
  LIOConfig lio_config_;
  LIOProcessFuncType lio_process_func_;

  void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);
  // sync lidar with imu
  bool SyncPackages();

  template <typename T>
  void SetPoseStamp(SopSE3 pose, T &out);
  void PublishPath(const SopSE3& pose);
  void PublishOdom(const SopSE3& pose);
  void PublishPointCloud(const CloudPtr& cloud, std::string frame_id,
                         const ros::Publisher& pub);

  std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;
  std::shared_ptr<ImuProcess> p_imu_ = nullptr;

  MeasureGroup measures_; // sync IMU and lidar scan

  double lidar_end_time_ = 0;

  ros::Publisher pub_odom_;
  ros::Publisher pub_path_;
  ros::Publisher pub_laser_cloud_world_;
  ros::Publisher pub_laser_cloud_imu_;

  nav_msgs::Odometry odometry_;
  nav_msgs::Path path_;
};


}  // namespace simple_lio
