//
// Created by myx on 2024/8/17.
//

#include <pcl/common/transforms.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/robust_kernel.h>
#include <filesystem>

#include "lio.h"
#include "common/lidar_utils.h"


namespace simple_lio{

// public
lio_test::lio_test() {
  preprocess_.reset(new PointCloudPreprocess());
  p_imu_.reset(new ImuProcess());
}

lio_test::~lio_test() {
  scan_down_body_ = nullptr;
  scan_undistort_ = nullptr;
  scan_down_world_ = nullptr;
  LOG(INFO) << "lio_test deconstruct";
}

// init with ros
bool lio_test::InitROS(ros::NodeHandle &nh) {
  if(!InitParamsFromROS(nh)){
    LOG(ERROR) << "InitParamsFromROS ERROR!";
    return false;
  }

  SubAndPubToROS(nh);

  if(nn_type_ == nearest_neighbor_type::IVOX){
    ivox_ = std::make_shared<IVoxType>(ivox_options_);
  }

  // esekf init
  std::vector<double> epsi(23, 0.001);
  kf_.init_dyn_share(
      get_f, df_dx, df_dw,
      [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
      3, epsi.data());
  return true;
}

void lio_test::Run(){
  if(opt_pose_update_){
    PublishOptPath();
    opt_pose_update_ = false;
  }
  if (!SyncPackages()) {
    return;
  }

  bool return_flag = false;
  Timer::Evaluate([&](){
    // IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);

    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
      LOG(WARNING) << "No point, skip this scan!";
      return_flag = true;
      return;
    }

    // down_sample
    voxel_scan_.setInputCloud(scan_undistort_);
    voxel_scan_.filter(*scan_down_body_);
    unsigned long cur_pts = scan_down_body_->size();
    if (cur_pts <= 5) {
      LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
      return_flag = true;
      return;
    }
    if(flg_first_scan_){
      if(nn_type_ == nearest_neighbor_type::IVOX){
        ivox_->AddPoints(scan_undistort_->points);
        first_lidar_time_ = measures_.lidar_beg_time_;

        scan_down_world_->resize(cur_pts);
        for(int i = 0; i < cur_pts; i++)
        {
          PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));
        }
        flg_first_scan_ = false;
      } else if(nn_type_ == nearest_neighbor_type::IKD_TREE){
        if(ikd_tree_.Root_Node == nullptr){
          ikd_tree_.set_downsample_param(filter_size_map_min_);
          first_lidar_time_ = measures_.lidar_beg_time_;

          scan_down_world_->resize(cur_pts);
          for(int i = 0; i < cur_pts; i++)
          {
            PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));
          }
          ikd_tree_.Build(scan_down_world_->points);

          flg_first_scan_ = false;
        }
      } else{
        LOG(ERROR) << "No nearst type is specified.";
      }

      frame_id_++;

      return_flag = true;
      return;
    }

    flg_EKF_inited_ = (measures_.lidar_beg_time_ - first_lidar_time_) >= INIT_TIME;

    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    residuals_.resize(cur_pts, 0);
    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, Vec4f::Zero());

    double solve_H_time = 0;
    // update the observation model, will call nn and point-to-plane residual computation
//    paper10_test_mtx_.lock();
    kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

    // save the state
    state_point_ = kf_.get_x();
  }, "Process Time Per Scan:");
  if(return_flag){
    return;
  }

  // update local map
  MapIncremental();

  SopSE3 current_pose = State2SE3(state_point_);
  SopSE3 delta_pose = last_pose_.inverse() * current_pose;


//  *pcl_wait_save_ += *scan_undistort_;
//  *pcl_wait_save_ += *scan_down_world_;
  static int scan_wait_num = 0;
  if(scan_wait_num == 0){
    pcd_cur_pose_ = current_pose;
  }

  // publish path and point cloud
//  PublishOptPath();
  PublishRosInfo();

}

// callbacks of lidar and imu
void lio_test::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
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

void lio_test::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  mtx_buffer_.lock();

  if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
    LOG(WARNING) << "lidar loop back, clear buffer";
    lidar_buffer_.clear();
  }

  last_timestamp_lidar_ = msg->header.stamp.toSec();

  if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
      !lidar_buffer_.empty()) {
    LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
              << ", lidar header time: " << last_timestamp_lidar_;
  }

  if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
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

void lio_test::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
  publish_count_++;
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  if (abs(timediff_lidar_wrt_imu_) > 0.1 && time_sync_en_) {
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
bool lio_test::SyncPackages() {
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

// interface of mtk, customized observation model
void lio_test::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data){
  unsigned long cnt_pts = scan_down_body_->size();

  std::vector<size_t> index(cnt_pts);
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
  }

  auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
  auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

  /** closest surface search and residual computation **/
  std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
    PointType &point_body = scan_down_body_->points[i];
    PointType &point_world = scan_down_world_->points[i];

    /* transform to world frame */
    Vec3f p_body = point_body.getVector3fMap();
    point_world.getVector3fMap() = R_wl * p_body + t_wl;
    point_world.intensity = point_body.intensity;

    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
    auto &points_near = nearest_points_[i];
    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      if(nn_type_ == nearest_neighbor_type::IKD_TREE){
        ikd_tree_.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
        point_selected_surf_[i] = points_near.size() >= NUM_MATCH_POINTS && pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 5.0;
      } else if(nn_type_ == nearest_neighbor_type::IVOX){
        ivox_->GetClosestPoint(point_world, points_near, NUM_MATCH_POINTS);
        point_selected_surf_[i] = points_near.size() >= MIN_NUM_MATCH_POINTS;
      } else{
        ROS_ERROR("nearest neighbor type is error!");
        return ;
      }
      if (point_selected_surf_[i]) {
        point_selected_surf_[i] = esti_plane(plane_coef_[i], points_near, ESTI_PLANE_THRESHOLD_);
      }
    }

    if (point_selected_surf_[i]) {
      auto temp = point_world.getVector4fMap();
      temp[3] = 1.0;
      float pd2 = plane_coef_[i].dot(temp);

      bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
      if (valid_corr) {
        point_selected_surf_[i] = true;
        residuals_[i] = pd2;
      } else{
        point_selected_surf_[i] = false;
      }
    }
  });

  effect_feat_num_ = 0;
  corr_pts_.resize(cnt_pts);
  corr_norm_.resize(cnt_pts);
  for (unsigned long i = 0; i < cnt_pts; i++) {
    if (point_selected_surf_[i]) {
      corr_norm_[effect_feat_num_] = plane_coef_[i];
      corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
      corr_pts_[effect_feat_num_][3] = residuals_[i];

      effect_feat_num_++;
    }
  }
  corr_pts_.resize(effect_feat_num_);
  corr_norm_.resize(effect_feat_num_);

  /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
  ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
  ekfom_data.h.resize(effect_feat_num_);

  index.resize(effect_feat_num_);
  const Mat3f off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
  const Vec3f off_t = s.offset_T_L_I.cast<float>();
  const Mat3f Rt = s.rot.toRotationMatrix().transpose().cast<float>();

  std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
    // have be is lidar frame pose
    Vec3f point_this_be = corr_pts_[i].head<3>();
    Mat3f point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
    // haven't be is imu frame pose
    Vec3f point_this = off_R * point_this_be + off_t;
    Mat3f point_crossmat = SKEW_SYM_MATRIX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    Vec3f norm_vec = corr_norm_[i].head<3>();

    /*** calculate the Measurement Jacobian matrix H ***/
    Vec3f C(Rt * norm_vec);
    Vec3f A(point_crossmat * C);

    if (extrinsic_est_en_) {
      Vec3f B(point_be_crossmat * off_R.transpose() * C);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
          B[1], B[2], C[0], C[1], C[2];
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measurement: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -corr_pts_[i][3];
  });
}

// debug save / show
void lio_test::PublishPath() {
  SetPoseStamp(msg_body_pose_);
  msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
  msg_body_pose_.header.frame_id = world_frame_id_;

  /*** if path is too large, the rvis will crash ***/
  path_.poses.push_back(msg_body_pose_);

  pub_path_.publish(path_);
}

void lio_test::PublishOptPath(){
  if(opt_pose_size_ == 0){
    return;
  }
  opt_pose_mtx_.try_lock();
  opt_path_.poses.clear();
  std::vector<size_t> index(opt_pose_size_);
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
  }
  opt_path_.poses.resize(opt_pose_size_);
  std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i){
    geometry_msgs::PoseStamped bp_pose_msg;
    SopSE3 bp_se3;
    bp_se3 = opt_pose_time_[i].opti_pose;
    SopSE3ToPose(bp_se3, bp_pose_msg.pose);
    bp_pose_msg.header.stamp = ros::Time().fromSec(opt_pose_time_[i].timestamp);
    bp_pose_msg.header.frame_id = imu_frame_id_;

    opt_path_.poses[i] = bp_pose_msg;
  });

  pub_opt_path_.publish(opt_path_);
  opt_pose_mtx_.unlock();
}

void lio_test::PublishOdom() {
  odometry_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
  SetPoseStamp(odometry_.pose);
  pub_odom_.publish(odometry_);
  auto P = kf_.get_P();
  for (int i = 0; i < 6; i++) {
    int k = i < 3 ? i + 3 : i - 3;
    odometry_.pose.covariance[i * 6 + 0] = P(k, 3);
    odometry_.pose.covariance[i * 6 + 1] = P(k, 4);
    odometry_.pose.covariance[i * 6 + 2] = P(k, 5);
    odometry_.pose.covariance[i * 6 + 3] = P(k, 0);
    odometry_.pose.covariance[i * 6 + 4] = P(k, 1);
    odometry_.pose.covariance[i * 6 + 5] = P(k, 2);
  }

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
                                        world_frame_id_, imu_frame_id_));
}

void lio_test::PublishPointCloudWorld() {
  PointCloudType::Ptr laserCloudWorld;
  if (dense_pub_en_) {
    PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
    unsigned long size = laserCloudFullRes->points.size();
    laserCloudWorld.reset(new PointCloudType(size, 1));
    for (int i = 0; i < size; i++) {
      PointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }
  } else {
    laserCloudWorld = scan_down_world_;
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
  laserCloudmsg.header.frame_id = world_frame_id_;
  pub_laser_cloud_world_.publish(laserCloudmsg);
}

void lio_test::PublishPointCloudBody() {
  unsigned long size = scan_undistort_->points.size();
  PointCloudType::Ptr laser_cloud_imu_body(new PointCloudType(size, 1));

//  pcl::transformPointCloud(*laser_cloud_imu_body, *scan_undistort_, current_pose.matrix());

  for (int i = 0; i < size; i++) {
    PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laser_cloud_imu_body, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
  laserCloudmsg.header.frame_id = imu_frame_id_;
  pub_laser_cloud_body_.publish(laserCloudmsg);
}

void lio_test::PublishRosInfo(){
  PublishOdom();
  PublishPath();

  PublishPointCloudWorld();

  if (scan_body_pub_en_) {
    PublishPointCloudBody();
  }
}

void lio_test::SavePathTum(const std::string &traj_file, nav_msgs::Path path) {
  std::ofstream ofs;
  ofs.open(traj_file, std::ios::out);
  if (!ofs.is_open()) {
    LOG(ERROR) << "Failed to open traj_file: " << traj_file;
    return;
  }

  // tum trajectory format
//  ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
  for (const auto &p : path.poses) {
    ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
        << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
        << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
  }
  LOG(INFO) << "Write path into: " << traj_file;
  ofs.close();
}


void lio_test::Finish() {
  /**************** save map ****************/
  /* 1. make sure you have enough memories **/
  /* 2. pcd save will largely influence the real-time performances **/
//  if(pcd_save_en_){
//    SavePcd();
//  }

  /**************** save path ****************/
  std::string log_dir = std::string(ROOT_DIR) + "log/" + dataset_;
  std::filesystem::path log_path(log_dir);
  // check the log path whether exist
  if (!std::filesystem::exists(log_path)) {
    // creat log path
    if (!std::filesystem::create_directories(log_path)) {
      std::cout << "Failed to create path: " << log_path << std::endl;
    }
  }

  if (log_dir.back() != '/') {
    log_dir += '/';
  }

  Timer::PrintAll();
  std::string run_log_name_ = log_dir + "run_time_log.csv";
  Timer::DumpIntoFile(run_log_name_);

  std::string path_save_name_ = log_dir + "ori_path.txt";
  std::string opt_path_save_name_ = log_dir + "opt_path.txt";
  std::string rtk_path_save_name_ = log_dir + "rtk_path.txt";
  std::string loop_path_save_name_ = log_dir + "loop_path.txt";

  SavePathTum(loop_path_save_name_, loop_path_);
  /**************** print and save timer log ****************/
//  Timer::PrintAll();
//  std::string run_log_name_ = log_dir + "run_time_log.csv";
//  Timer::DumpIntoFile(run_log_name_);

  LOG(INFO) << "finish done";
}

// private
template <typename T>
void lio_test::SetPoseStamp(T &out) {
  out.pose.position.x = state_point_.pos(0);
  out.pose.position.y = state_point_.pos(1);
  out.pose.position.z = state_point_.pos(2);
  out.pose.orientation.x = state_point_.rot.coeffs()[0];
  out.pose.orientation.y = state_point_.rot.coeffs()[1];
  out.pose.orientation.z = state_point_.rot.coeffs()[2];
  out.pose.orientation.w = state_point_.rot.coeffs()[3];
}

SopSE3 lio_test::State2SE3(state_ikfom state){
  SopSE3 se_3 = SopSE3(state.rot, state.pos);
  return se_3;
}

void lio_test::PointBodyToWorld(const PointType *pi, PointType *const po) {
  Vec3d p_body(pi->x, pi->y, pi->z);
  Vec3d p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                       state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void lio_test::PointBodyToWorld(const Vec3f &pi, PointType *const po) {
  Vec3d p_body(pi.x(), pi.y(), pi.z());
  Vec3d p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                       state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = std::abs(po->z);
}

void lio_test::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
  Vec3d p_body_lidar(pi->x, pi->y, pi->z);
  Vec3d p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void lio_test::SopSE3ToTransOrientation(SopSE3 &se_3, Vec3d &trans, Vec4d &ori){
  trans.x() = se_3.translation().x();
  trans.y() = se_3.translation().y();
  trans.z() = se_3.translation().z();
  ori.x() = se_3.so3().unit_quaternion().x();
  ori.y() = se_3.so3().unit_quaternion().y();
  ori.z() = se_3.so3().unit_quaternion().z();
  ori.w() = se_3.so3().unit_quaternion().w();
}

void lio_test::SopSE3ToPose(SopSE3 se_3, geometry_msgs::Pose &pose){
  pose.position.x = se_3.translation().x();
  pose.position.y = se_3.translation().y();
  pose.position.z = se_3.translation().z();
  pose.orientation.x = se_3.so3().unit_quaternion().x();
  pose.orientation.y = se_3.so3().unit_quaternion().y();
  pose.orientation.z = se_3.so3().unit_quaternion().z();
  pose.orientation.w = se_3.so3().unit_quaternion().w();
}

void lio_test::MapIncremental(){
  PointVector points_to_add;
  PointVector point_no_need_down_sample;

  unsigned long cur_pts = scan_down_body_->size();
  points_to_add.reserve(cur_pts);
  point_no_need_down_sample.reserve(cur_pts);

  std::vector<size_t> index(cur_pts);
  for (size_t i = 0; i < cur_pts; ++i) {
    index[i] = i;
  }

  std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
    /* transform to world frame */
    PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

    /* decide if need add to map */
    PointType &point_world = scan_down_world_->points[i];
    if (!nearest_points_[i].empty() && flg_EKF_inited_) {
      const PointVector &points_near = nearest_points_[i];

      Eigen::Vector3f center =
          ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

      Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

      if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
          fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
          fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
        point_no_need_down_sample.emplace_back(point_world);
        return;
      }

      bool need_add = true;
      float dist = calc_dist(point_world.getVector3fMap(), center);
      if (points_near.size() >= NUM_MATCH_POINTS) {
        for (int read_i = 0; read_i < NUM_MATCH_POINTS; read_i++) {
          if (calc_dist(points_near[read_i].getVector3fMap(), center) < dist + 1e-6) {
            need_add = false;
            break;
          }
        }
      }
      if (need_add) {
        points_to_add.emplace_back(point_world);
      }
    } else {
      points_to_add.emplace_back(point_world);
    }
  });

  if(nn_type_ == nearest_neighbor_type::IKD_TREE){
    ikd_tree_.Add_Points(points_to_add, true);
    ikd_tree_.Add_Points(point_no_need_down_sample, false);
  } else if(nn_type_ == nearest_neighbor_type::IVOX){
    ivox_->AddPoints(points_to_add);
    ivox_->AddPoints(point_no_need_down_sample);
  }
}

void lio_test::SubAndPubToROS(ros::NodeHandle &nh){
  // ROS subscribe initialization
  std::string lidar_topic, imu_topic, rtk_topic;
  nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
  nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<std::string>("common/rtk_topic", rtk_topic, "/fix");

  if (preprocess_->GetLidarType() == LidarType::AVIA) {
    sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
  } else {
    sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
  }
  sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,[this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });


  // ROS publisher init
  path_.header.stamp = ros::Time::now();
  path_.header.frame_id = world_frame_id_;
  opt_path_.header.frame_id = world_frame_id_;
  odometry_.header.frame_id = world_frame_id_;
  odometry_.child_frame_id = imu_frame_id_;

  pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
  pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
  pub_odom_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);
  pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100);
  pub_opt_path_ = nh.advertise<nav_msgs::Path>("/opt_path", 100);
  pub_rtk_path_ = nh.advertise<nav_msgs::Path>("/rtk_path", 100);
  pub_loop_ = nh.advertise<visualization_msgs::MarkerArray>("/loop_vis", 100);

  cur_loop_map_ = nh.advertise<sensor_msgs::PointCloud2>("/cur_loop_map", 100);
  pre_loop_map_ = nh.advertise<sensor_msgs::PointCloud2>("/pre_loop_map", 100);
}

bool lio_test::InitParamsFromROS(ros::NodeHandle &nh){
  // get params from param server
  int lidar_type, nn_type, ivox_nearby_type;
  double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
  std::vector<double> rtk_ext_t = {0.0, 0.0, 0.0}, rtk_ext_r = {0.0, 0.0, 0.0};

  nh.param<bool>("common/time_sync_en", time_sync_en_, false);
  nh.param<std::string>("common/dataset", dataset_, "unset_dataset_name");
  nh.param<bool>("path_save_en", traj_save_en_, true);
  nh.param<bool>("publish/scan_publish_en", scan_pub_en_, true);
  nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
  nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en_, true);
  nh.param<std::string>("publish/tf_imu_frame", imu_frame_id_, "body");
  nh.param<std::string>("publish/tf_world_frame", world_frame_id_, "camera_init");
  nh.param<bool>("output_debug_en", output_debug_en_, false);

  nh.param<int>("max_iteration", NUM_MAX_ITERATIONS_, 4);
  nh.param<std::string>("map_file_path", map_file_path_, "");
  nh.param<float>("filter_size_surf", filter_size_surf_min_, 0.5);
  nh.param<float>("filter_size_map", filter_size_map_min_, 0.0);
  nh.param<double>("cube_side_length", cube_len_, 200);
  nh.param<float>("mapping/det_range", det_range_, 300.f);
  nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
  nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
  nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
  nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
  nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
  nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-3);
  nh.param<int>("preprocess/lidar_type", lidar_type, 1);
  nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);
  nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
  nh.param<bool>("feature_extract_enable", preprocess_->FeatureEnabled(), false);
  nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en_, true);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
  nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
  nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
  nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());

  nh.param<int>("nearest_neighbor_type", nn_type, 1);
  nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
  nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);
//  nh.param<float>("esti_plane_threshold", options::ESTI_PLANE_THRESHOLD, 0.1);

  // loop closure
  nh.param("loop_closure/loop_closure_frequency", lp_closure_frequency_, 50.0);
  nh.param("loop_closure/loop_closure_enable", lp_closure_enable_, true);
  nh.param("loop_closure/loop_sparse_enable", lp_sparse_enable_, true);
  nh.param("loop_closure/graph_iter_num", graph_iter_num_, 10);
  nh.param("loop_closure/history_time_threshold", history_time_th_, 60.0);
  nh.param("loop_closure/lp_distance_threshold", lp_dis_th_, 20.0);
  nh.param("loop_closure/lp_delta_trans", lp_delta_trans_, 5.0);
  nh.param("loop_closure/lp_delta_degree", lp_delta_degree_, 45.0);
  nh.param("loop_closure/lp_ndt_trans_score_threshold", lp_ndt_trans_score_th_, 5.0);
  nh.param("loop_closure/lp_ndt_fitness_score_threshold", lp_ndt_fitness_score_th, 1.0);
  nh.param("loop_closure/lp_pose_noise", lp_pos_noise_, 0.01);
  nh.param("loop_closure/lp_angle_noise", lp_ang_noise_, 0.05);
  nh.param("loop_closure/kf_num_sub_map", kf_num_sub_map_, 10);
  nh.param("loop_closure/min_id_threshold", min_id_th_, 20);
  nh.param("loop_closure/skip_id", skip_id_, 5);

  // rtk trans
  nh.param("rtk_trans/extrinsic_T", rtk_ext_t, std::vector<double>());
  nh.param("rtk_trans/extrinsic_R", rtk_ext_r, std::vector<double>());

  LOG(INFO) << "lidar_type " << lidar_type;
  if (lidar_type == 1) {
    preprocess_->SetLidarType(LidarType::AVIA);
    LOG(INFO) << "Using AVIA Lidar";
  } else if (lidar_type == 2) {
    preprocess_->SetLidarType(LidarType::VELO32);
    LOG(INFO) << "Using Velodyne 32 Lidar";
  } else if (lidar_type == 3) {
    preprocess_->SetLidarType(LidarType::OUST64);
    LOG(INFO) << "Using OUST 64 Lidar";
  } else {
    LOG(WARNING) << "unknown lidar_type";
    return false;
  }

  if (ivox_nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (ivox_nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (ivox_nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (ivox_nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }

  if (nn_type == 1) {
    nn_type_ = nearest_neighbor_type::IKD_TREE;
    LOG(INFO) << "Using IKD_TREE as nearest neighbor type";
  } else if (nn_type == 2) {
    nn_type_ = nearest_neighbor_type::IVOX;
    LOG(INFO) << "Using IVOX as nearest neighbor type";
  } else {
    LOG(WARNING) << "unknown nearest neighbor type";
    return false;
  }

  voxel_scan_.setLeafSize(filter_size_surf_min_, filter_size_surf_min_, filter_size_surf_min_);

  Vec3d lidar_T_wrt_IMU = VecFromArray<double>(extrinT_);
  Mat3d lidar_R_wrt_IMU = MatFromArray<double>(extrinR_);

  p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
  p_imu_->SetGyrCov(Vec3d(gyr_cov, gyr_cov, gyr_cov));
  p_imu_->SetAccCov(Vec3d(acc_cov, acc_cov, acc_cov));
  p_imu_->SetGyrBiasCov(Vec3d(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu_->SetAccBiasCov(Vec3d(b_acc_cov, b_acc_cov, b_acc_cov));
  return true;
}

} // namespace faster_lio