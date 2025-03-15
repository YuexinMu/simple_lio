//
// Created by myx on 2025/3/6.
//

#include "lio_base.h"
#include <tf/transform_broadcaster.h>

namespace simple_lio{

bool LioBase::Init(ros::NodeHandle &nh) {
  LoadParams(nh);

  preprocess_ = std::make_shared<PointCloudPreprocess>();
  p_imu_ = std::make_shared<ImuProcess>();

  preprocess_->Blind() = lio_config_.blind;
  preprocess_->TimeScale() = lio_config_.time_scale;
  preprocess_->NumScans() = lio_config_.scan_line;
  preprocess_->PointFilterNum() = lio_config_.point_filter_num;
  preprocess_->FeatureEnabled() = lio_config_.feature_extract_enable;

  LOG(INFO) << "lidar_type " << lio_config_.lidar_type;
  if (lio_config_.lidar_type == 1) {
    preprocess_->SetLidarType(LidarType::AVIA);
    LOG(INFO) << "Using AVIA Lidar";
  } else if (lio_config_.lidar_type == 2) {
    preprocess_->SetLidarType(LidarType::VELO32);
    LOG(INFO) << "Using Velodyne 32 Lidar";
  } else if (lio_config_.lidar_type == 3) {
    preprocess_->SetLidarType(LidarType::OUST64);
    LOG(INFO) << "Using OUST 64 Lidar";
  } else {
    LOG(WARNING) << "unknown lidar_type";
    return false;
  }

  Vec3d lidar_T_wrt_IMU = VecFromArray<double>(lio_config_.extrinsic_T);
  Mat3d lidar_R_wrt_IMU = MatFromArray<double>(lio_config_.extrinsic_R);

  p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
  p_imu_->SetGyrCov(Vec3d(lio_config_.gyr_cov, lio_config_.gyr_cov, lio_config_.gyr_cov));
  p_imu_->SetAccCov(Vec3d(lio_config_.acc_cov, lio_config_.acc_cov, lio_config_.acc_cov));
  p_imu_->SetGyrBiasCov(Vec3d(lio_config_.b_gyr_cov, lio_config_.b_gyr_cov, lio_config_.b_gyr_cov));
  p_imu_->SetAccBiasCov(Vec3d(lio_config_.b_acc_cov, lio_config_.b_acc_cov, lio_config_.b_acc_cov));

  path_.header.stamp = ros::Time::now();
  path_.header.frame_id = lio_config_.init_frame;

  odometry_.header.frame_id = lio_config_.init_frame;
  odometry_.child_frame_id = lio_config_.body_frame;

  return true;
}

bool LioBase::LoadParams(ros::NodeHandle &nh) {
  // common params
  nh.param<std::string>("lio_base/lidar_topic", lio_config_.lid_topic, "/livox/lidar");
  nh.param<std::string>("lio_base/imu_topic", lio_config_.imu_topic, "/livox/imu");
  nh.param<bool>("lio_base/time_sync_en", lio_config_.time_sync_en, false);

  nh.param<int>("lio_base/lidar_type", lio_config_.lidar_type, 1);
  nh.param<int>("lio_base/scan_line", lio_config_.scan_line, 16);
  nh.param<double>("lio_base/blind", lio_config_.blind, 0.01);
  nh.param<float>("lio_base/time_scale", lio_config_.time_scale, 1e-3);

  nh.param<double>("lio_base/gyr_cov", lio_config_.gyr_cov, 0.1);
  nh.param<double>("lio_base/acc_cov", lio_config_.acc_cov, 0.1);
  nh.param<double>("lio_base/b_gyr_cov", lio_config_.b_gyr_cov, 0.0001);
  nh.param<double>("lio_base/b_acc_cov", lio_config_.b_acc_cov, 0.0001);

  nh.param<std::vector<double>>("lio_base/extrinsic_T", lio_config_.extrinsic_T, std::vector<double>());
  nh.param<std::vector<double>>("lio_base/extrinsic_R", lio_config_.extrinsic_R, std::vector<double>());

  nh.param<int>("lio_base/point_filter_num", lio_config_.point_filter_num, 2);
  nh.param<float>("lio_base/filter_size_surf", lio_config_.filter_size_surf, 0.5);
  nh.param<bool>("lio_base/feature_extract_enable", lio_config_.feature_extract_enable, false);

  nh.param<std::string>("lio_base/body_frame", lio_config_.body_frame, "body");
  nh.param<std::string>("lio_base/init_frame", lio_config_.init_frame, "camera_init");
  nh.param<std::string>("lio_base/odom_topic", lio_config_.odom_topic, "odometry");
  nh.param<std::string>("lio_base/path_topic", lio_config_.path_topic, "path");
  nh.param<std::string>("lio_base/cloud_world_topic", lio_config_.cloud_world_topic, "cloud_registered_world");
  nh.param<std::string>("lio_base/cloud_imu_topic", lio_config_.cloud_imu_topic, "cloud_registered_imu");

  return true;
}

// callbacks of lidar and imu
void LioBase::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  mtx_buffer_.lock();
  if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
    LOG(ERROR) << "lidar loop back, clear buffer";
    lidar_buffer_.clear();
  }

  PointCloudType::Ptr ptr(new PointCloudType());
  preprocess_->Process(msg, ptr);
  lidar_buffer_.push_back(ptr);
  time_buffer_.push_back(msg->header.stamp.toSec());
  last_timestamp_lidar_ = msg->header.stamp.toSec();
  mtx_buffer_.unlock();
}

void LioBase::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  mtx_buffer_.lock();

  if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
    LOG(WARNING) << "lidar loop back, clear buffer";
    lidar_buffer_.clear();
  }

  last_timestamp_lidar_ = msg->header.stamp.toSec();

  if (!lio_config_.time_sync_en && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
      !lidar_buffer_.empty()) {
    LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
              << ", lidar header time: " << last_timestamp_lidar_;
  }

  if (lio_config_.time_sync_en && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
      !imu_buffer_.empty()) {
    timediff_set_flg_ = true;
    timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
    LOG(INFO) << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_;
  }

  PointCloudType::Ptr ptr(new PointCloudType());
  preprocess_->Process(msg, ptr);
  lidar_buffer_.emplace_back(ptr);
  time_buffer_.emplace_back(last_timestamp_lidar_);


  mtx_buffer_.unlock();
}

void LioBase::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  if (abs(timediff_lidar_wrt_imu_) > 0.1 && lio_config_.time_sync_en) {
    msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu_ + msg_in->header.stamp.toSec());
  }

  double timestamp = msg->header.stamp.toSec();

  mtx_buffer_.lock();
  if (timestamp < last_timestamp_imu_) {
    LOG(WARNING) << "imu loop back, clear buffer";
    imu_buffer_.clear();
  }

  last_timestamp_imu_ = timestamp;
  imu_buffer_.emplace_back(msg);
  mtx_buffer_.unlock();
}


// sync lidar with imu
bool LioBase::SyncPackages() {
  if (lidar_buffer_.empty() || imu_buffer_.empty()) {
    return false;
  }

  /*** push a lidar scan ***/
  if (!lidar_pushed_) {
    measures_.lidar_ = lidar_buffer_.front();
    measures_.lidar_beg_time_ = time_buffer_.front();

    if (measures_.lidar_->points.size() <= 1) {
      LOG(WARNING) << "Too few input point cloud!";
      lidar_end_time_ = measures_.lidar_beg_time_ + lidar_mean_scantime_;
    } else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_) {
      lidar_end_time_ = measures_.lidar_beg_time_ + lidar_mean_scantime_;
    } else {
      scan_num_++;
      lidar_end_time_ = measures_.lidar_beg_time_ + measures_.lidar_->points.back().curvature / double(1000);
      lidar_mean_scantime_ +=
          (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
    }

    measures_.lidar_end_time_ = lidar_end_time_;
    lidar_pushed_ = true;
  }

  if (last_timestamp_imu_ < lidar_end_time_) {
    return false;
  }

  /*** push imu_ data, and pop from imu_ buffer ***/
  double imu_time = imu_buffer_.front()->header.stamp.toSec();
  measures_.imu_.clear();
  while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
    imu_time = imu_buffer_.front()->header.stamp.toSec();
    if (imu_time > lidar_end_time_) break;
    measures_.imu_.push_back(imu_buffer_.front());
    imu_buffer_.pop_front();
  }

  lidar_buffer_.pop_front();
  time_buffer_.pop_front();
  lidar_pushed_ = false;
  return true;
}

template <typename T>
void LioBase::SetPoseStamp(SopSE3 pose, T &out) {
  out.pose.position.x = pose.translation().x();
  out.pose.position.y = pose.translation().y();
  out.pose.position.z = pose.translation().z();
  out.pose.orientation.x = pose.so3().unit_quaternion().x();
  out.pose.orientation.y = pose.so3().unit_quaternion().y();
  out.pose.orientation.z = pose.so3().unit_quaternion().z();
  out.pose.orientation.w = pose.so3().unit_quaternion().w();
}

void LioBase::PublishPath(const SopSE3& pose) {
  geometry_msgs::PoseStamped msg_body_pose;
  SetPoseStamp(pose, msg_body_pose);
  msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time_);
  msg_body_pose.header.frame_id = lio_config_.init_frame;

  /*** if path is too large, the rvis will crash ***/
  path_.poses.push_back(msg_body_pose);

  pub_path_.publish(path_);
}

void LioBase::PublishOdom(const SopSE3& pose) {
  odometry_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
  SetPoseStamp(pose, odometry_.pose);
  pub_odom_.publish(odometry_);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(odometry_.pose.pose.position.x,
                                  odometry_.pose.pose.position.y,
                                  odometry_.pose.pose.position.z));
  q.setW(odometry_.pose.pose.orientation.w);
  q.setX(odometry_.pose.pose.orientation.x);
  q.setY(odometry_.pose.pose.orientation.y);
  q.setZ(odometry_.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odometry_.header.stamp,
                                        lio_config_.init_frame, lio_config_.body_frame));
}

void LioBase::PublishPointCloud(const CloudPtr& cloud, std::string frame_id,
                                const ros::Publisher& pub) {
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*cloud, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
  laserCloudmsg.header.frame_id = std::move(frame_id);
  pub.publish(laserCloudmsg);
}


}  // namespace simple_lio

