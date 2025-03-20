//
// Created by myx on 2024/8/17.
//

#include "lio.h"

#include <tf/transform_broadcaster.h>

namespace simple_lio{

// public
lio::lio() {
  preprocess_.reset(new PointCloudPreprocess());
  p_imu_.reset(new ImuProcess());
}

lio::~lio() {
  scan_down_body_ = nullptr;
  scan_undistort_ = nullptr;
  scan_down_world_ = nullptr;
  LOG(INFO) << "lio deconstruct";
}

// init with ros
bool lio::Init(ros::NodeHandle &nh) {
  LoadParams(nh);


  preprocess_.reset(new PointCloudPreprocess());
  p_imu_.reset(new ImuProcess());
  
  preprocess_->Blind() = config_.blind;
  preprocess_->TimeScale() = config_.time_scale;
  preprocess_->NumScans() = config_.scan_line;
  preprocess_->PointFilterNum() = config_.point_filter_num;
  preprocess_->FeatureEnabled() = config_.feature_extract_enable;

  LOG(INFO) << "lidar_type " << config_.lidar_type;
  if (config_.lidar_type == 1) {
    preprocess_->SetLidarType(LidarType::AVIA);
    LOG(INFO) << "Using AVIA Lidar";
  } else if (config_.lidar_type == 2) {
    preprocess_->SetLidarType(LidarType::VELO32);
    LOG(INFO) << "Using Velodyne 32 Lidar";
  } else if (config_.lidar_type == 3) {
    preprocess_->SetLidarType(LidarType::OUST64);
    LOG(INFO) << "Using OUST 64 Lidar";
  } else {
    LOG(WARNING) << "unknown lidar_type";
    return false;
  }

  Vec3d lidar_T_wrt_IMU = VecFromArray<double>(config_.extrinsic_T);
  Mat3d lidar_R_wrt_IMU = MatFromArray<double>(config_.extrinsic_R);

  p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
  p_imu_->SetGyrCov(Vec3d(config_.gyr_cov, config_.gyr_cov, config_.gyr_cov));
  p_imu_->SetAccCov(Vec3d(config_.acc_cov, config_.acc_cov, config_.acc_cov));
  p_imu_->SetGyrBiasCov(Vec3d(config_.b_gyr_cov, config_.b_gyr_cov, config_.b_gyr_cov));
  p_imu_->SetAccBiasCov(Vec3d(config_.b_acc_cov, config_.b_acc_cov, config_.b_acc_cov));


  if (config_.nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (config_.nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (config_.nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (config_.nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }
  ivox_options_.resolution_ = config_.resolution;

  if (config_.nn_type == 1) {
    nn_type_ = nearest_neighbor_type::IKD_TREE;
    LOG(INFO) << "Using IKD_TREE as nearest neighbor type";
  } else if (config_.nn_type == 2) {
    nn_type_ = nearest_neighbor_type::IVOX;
    LOG(INFO) << "Using IVOX as nearest neighbor type";
  } else {
    LOG(WARNING) << "unknown nearest neighbor type";
    return false;
  }

  voxel_scan_.setLeafSize(config_.filter_size_surf, config_.filter_size_surf, config_.filter_size_surf);

  if(nn_type_ == nearest_neighbor_type::IVOX){
    ivox_ = std::make_shared<IVoxType>(ivox_options_);
  }

  // esekf init
  std::vector<double> epsi(23, 0.001);
  kf_.init_dyn_share(
      get_f, df_dx, df_dw,
      [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
      config_.max_iteration, epsi.data());

  SubAndPubToROS(nh);
  return true;
}

// init without ros
bool lio::InitWithoutROS(const std::string &config_yaml){
  return true;
}

void lio::Run(){
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

    if(nn_type_ == nearest_neighbor_type::IVOX){
      if (flg_first_scan_) {
        ivox_->AddPoints(scan_undistort_->points);
        first_lidar_time_ = measures_.lidar_beg_time_;
        flg_first_scan_ = false;
        return_flag = true;
        return;
      }
    }

    flg_EKF_inited_ = (measures_.lidar_beg_time_ - first_lidar_time_) >= INIT_TIME;

    // down_sample
    if(nn_type_ == nearest_neighbor_type::IKD_TREE){
      //      Timer::Evaluate([&, this]() {
      voxel_scan_.setInputCloud(scan_undistort_);
      voxel_scan_.filter(*scan_down_body_);
      //      },"Downsample PointCloud");
    } else if(nn_type_ == nearest_neighbor_type::IVOX){
      //      Timer::Evaluate([&, this]() {
      voxel_scan_.setInputCloud(scan_undistort_);
      voxel_scan_.filter(*scan_down_body_);
      //      },"Downsample PointCloud");
    }

    unsigned long cur_pts = scan_down_body_->size();
    if (cur_pts <= 5) {
      LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
      return_flag = true;
      return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    residuals_.resize(cur_pts, -1.0);
    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, Vec4f::Zero());

    if(nn_type_ == nearest_neighbor_type::IKD_TREE)
    {
      if(ikd_tree_.Root_Node == nullptr){
        if(cur_pts > 5){
          ikd_tree_.set_downsample_param(config_.filter_size_map);
          scan_down_world_->resize(cur_pts);
          for(int i = 0; i < cur_pts; i++)
          {
            PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));
          }
          ikd_tree_.Build(scan_down_world_->points);
        }
        return_flag = true;
        return;
      }
    }

    // ICP and iterated Kalman filter update
    //    Timer::Evaluate([&, this]() {
    // iterated state estimation
    double solve_H_time = 0;
    // update the observation model, will call nn and point-to-plane residual computation
    kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    // save the state
    state_point_ = kf_.get_x();
//    euler_cur_ = SO3ToEuler(state_point_.rot);
    pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
    //        },"IEKF Solve and Update");
  },  "Process Time Per Scan:");

  if(return_flag){
    return;
  }
  // update local map
  Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

  //  LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " down_sample " << cur_pts;

  //  if(nn_type_ == nearest_neighbor_type::IKD_TREE){
  //    LOG(INFO) << " Kdtree size num: " << ikd_tree_.size() << " effect feat num : " << effect_feat_num_;
  //  } else if(nn_type_ == nearest_neighbor_type::IVOX){
  //    LOG(INFO) << " Map grid num: " << ivox_->NumValidGrids() << " effect feat num : " << effect_feat_num_;
  //  }

  // publish or save map pcd
//  if (run_in_offline_) {
//    if (pcd_save_en_) {
//      PublishFrameWorld();
//    }
//    if (traj_save_en_) {
//      PublishPath(pub_path_);
//    }
//  } else {
//    if (pub_odom_aft_mapped_) {
//      PublishOdometry(pub_odom_aft_mapped_);
//    }
//    if (path_pub_en_ || traj_save_en_) {
//      PublishPath(pub_path_);
//    }
//    if (scan_pub_en_ || pcd_save_en_) {
//      PublishFrameWorld();
//    }
//    if (scan_pub_en_ && scan_body_pub_en_) {
//      PublishFrameBody();
//    }
//    if (scan_pub_en_ && scan_effect_pub_en_) {
//      PublishFrameEffectWorld(pub_laser_cloud_effect_world_);
//    }
//  }


  PublishPath(State2SE3(state_point_));
  PublishOdom(State2SE3(state_point_));

  PublishPointCloud(PointCloudLidarToIMU(scan_down_body_),
                    config_.body_frame, pub_laser_cloud_imu_);
  PublishPointCloud(scan_down_world_, config_.init_frame,
                    pub_laser_cloud_world_);

  // Debug variables
  frame_num_++;
}

// callbacks of lidar and imu
void lio::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  mtx_buffer_.lock();
  Timer::Evaluate(
      [&, this]() {
        scan_count_++;
        if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
          LOG(ERROR) << "lidar loop back, clear buffer";
          lidar_buffer_.clear();
        }

        PointCloudType::Ptr ptr(new PointCloudType());
        preprocess_->Process(msg, ptr);
        lidar_buffer_.push_back(ptr);
        time_buffer_.push_back(msg->header.stamp.toSec());
        last_timestamp_lidar_ = msg->header.stamp.toSec();
      },
      "Preprocess (Standard)");
  mtx_buffer_.unlock();
}

void lio::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  mtx_buffer_.lock();
  Timer::Evaluate(
      [&, this]() {
        scan_count_++;
        if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
          LOG(WARNING) << "lidar loop back, clear buffer";
          lidar_buffer_.clear();
        }

        last_timestamp_lidar_ = msg->header.stamp.toSec();

        if (!config_.time_sync_en && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
            !lidar_buffer_.empty()) {
          LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
                    << ", lidar header time: " << last_timestamp_lidar_;
        }

        if (config_.time_sync_en && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
            !imu_buffer_.empty()) {
          timediff_set_flg_ = true;
          timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
          LOG(INFO) << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_;
        }

        PointCloudType::Ptr ptr(new PointCloudType());
        preprocess_->Process(msg, ptr);
        lidar_buffer_.emplace_back(ptr);
        time_buffer_.emplace_back(last_timestamp_lidar_);
      },
      "Preprocess (Livox)");

  mtx_buffer_.unlock();
}

void lio::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
  publish_count_++;
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  if (abs(timediff_lidar_wrt_imu_) > 0.1 && config_.time_sync_en) {
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
bool lio::SyncPackages() {
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

// get quantile
float lio::getQuantile(vector<float>& data, float quantile) {
  size_t n = data.size();
  size_t index = static_cast<size_t>(quantile * (n - 1));
  nth_element(data.begin(), data.begin() + index, data.end());
  return data[index];
}

void lio::detectOutliers(vector<float>& data, vector<float>& normal_data, std::vector<bool> &point_selected_surf) {
  vector<float> sorted_data = data;
  sorted_data.erase(remove(sorted_data.begin(), sorted_data.end(), -1.0), sorted_data.end());
  sort(sorted_data.begin(), sorted_data.end());

  // 计算四分位数
  float Q1 = getQuantile(sorted_data, 0.1);
  float Q3 = getQuantile(sorted_data, 0.9);
  float IQR = Q3 - Q1;

  // 异常值的范围
  float lower_bound = Q1 - 1.5 * IQR;
  float upper_bound = Q3 + 1.5 * IQR;
  // 将数据分类为正常值和异常值
  for (int i = 0; i < data.size(); i++) {
    //    if (data[i] < lower_bound || data[i] > upper_bound) {
    if (data[i] < Q1 || data[i] > Q3) {
      //      point_selected_surf[i] = false;
    } else {
      normal_data.push_back(data[i]);
    }
  }
}

// interface of mtk, customized observation model
void lio::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data){
  unsigned long cnt_pts = scan_down_body_->size();

  std::vector<size_t> index(cnt_pts);
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
  }

  Timer::Evaluate([&, this](){
    auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
    auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

    //    Vec3f p_body;
    //    p_body.x() = s.pos.x();
    //    p_body.y() = s.pos.y();
    //    p_body.z() = s.pos.z();
    //    if(scan_count_ % 100 == 0){
    //      Number::Record(p_body.stableNorm(), "residuals_scan_"+to_string(scan_count_));
    //    }

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
          point_selected_surf_[i] = esti_plane(plane_coef_[i], points_near, config_.plane_threshold);
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
  },"    ObsModel (Lidar Match)");

  //  std::vector<float> normal;
  //  detectOutliers(residuals_, normal, point_selected_surf_);
  //  residuals_.swap(normal);

  effect_feat_num_ = 0;
  corr_pts_.resize(cnt_pts);
  corr_norm_.resize(cnt_pts);
  for (unsigned long i = 0; i < cnt_pts; i++) {
    if (point_selected_surf_[i]) {
      corr_norm_[effect_feat_num_] = plane_coef_[i];
      corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
      corr_pts_[effect_feat_num_][3] = residuals_[i];

      //      if(scan_count_ % 100 == 0){
      //        Number::Record(residuals_[i], "residuals_scan_"+to_string(scan_count_));
      //      }

      effect_feat_num_++;
    }
  }
  corr_pts_.resize(effect_feat_num_);
  corr_norm_.resize(effect_feat_num_);

  Timer::Evaluate(
      [&, this]() {
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
      },"    ObsModel (IEKF Build Jacobian)");
}

template <typename T>
void lio::SetPoseStamp(SopSE3 pose, T &out) {
  out.pose.position.x = pose.translation().x();
  out.pose.position.y = pose.translation().y();
  out.pose.position.z = pose.translation().z();
  out.pose.orientation.x = pose.so3().unit_quaternion().x();
  out.pose.orientation.y = pose.so3().unit_quaternion().y();
  out.pose.orientation.z = pose.so3().unit_quaternion().z();
  out.pose.orientation.w = pose.so3().unit_quaternion().w();
}

void lio::PublishPath(const SopSE3& pose) {
  geometry_msgs::PoseStamped msg_body_pose;
  SetPoseStamp(pose, msg_body_pose);
  msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time_);
  msg_body_pose.header.frame_id = config_.init_frame;

  /*** if path is too large, the rvis will crash ***/
  path_.poses.push_back(msg_body_pose);

  pub_path_.publish(path_);
}

void lio::PublishOdom(const SopSE3& pose) {
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
                                        config_.init_frame, config_.body_frame));
}

void lio::PublishPointCloud(const CloudPtr& cloud, std::string frame_id,
                                const ros::Publisher& pub) {
  CloudPtr po{new PointCloudType()};
  size_t size = cloud->points.size();
  po->points.resize(size);
  po->points = cloud->points;

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*po, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
  laserCloudmsg.header.frame_id = std::move(frame_id);
  pub.publish(laserCloudmsg);
}

SopSE3 lio::State2SE3(state_ikfom state){
  SopSE3 se_3 = SopSE3(state.rot, state.pos);
  return se_3;
}

void lio::PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world) {
  unsigned long size = corr_pts_.size();
  PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

  for (int i = 0; i < size; i++) {
    PointBodyToWorld(corr_pts_[i].head<3>(), &laser_cloud->points[i]);
  }
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laser_cloud, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
  laserCloudmsg.header.frame_id = "camera_init";
  pub_laser_cloud_effect_world.publish(laserCloudmsg);
}

void lio::SaveTrajectory(const std::string &traj_file) {
  std::ofstream ofs;
  ofs.open(traj_file, std::ios::out);
  if (!ofs.is_open()) {
    LOG(ERROR) << "Failed to open traj_file: " << traj_file;
    return;
  }

  // tum trajectory format
  ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
  for (const auto &p : path_.poses) {
    ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
        << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
        << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
  }

  ofs.close();
}

void lio::Finish() {
  /**************** save map ****************/
  /* 1. make sure you have enough memories **/
//  /* 2. pcd save will largely influence the real-time performances **/
//  if (pcl_wait_save_->size() > 0 && pcd_save_en_) {
//    std::string file_name = std::string("scans.pcd");
//    std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
//    pcl::PCDWriter pcd_writer;
//    LOG(INFO) << "current scan saved to /PCD/" << file_name;
//    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
//  }

  LOG(INFO) << "finish done";
}

// private
//template <typename T>
//void lio::SetPoseStamp(T &out) {
//  out.pose.position.x = state_point_.pos(0);
//  out.pose.position.y = state_point_.pos(1);
//  out.pose.position.z = state_point_.pos(2);
//  out.pose.orientation.x = state_point_.rot.coeffs()[0];
//  out.pose.orientation.y = state_point_.rot.coeffs()[1];
//  out.pose.orientation.z = state_point_.rot.coeffs()[2];
//  out.pose.orientation.w = state_point_.rot.coeffs()[3];
//}

void lio::PointBodyToWorld(const PointType *pi, PointType *const po) {
  Vec3d p_body(pi->x, pi->y, pi->z);
  Vec3d p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                       state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void lio::PointBodyToWorld(const Vec3f &pi, PointType *const po) {
  Vec3d p_body(pi.x(), pi.y(), pi.z());
  Vec3d p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                       state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = std::abs(po->z);
}

void lio::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
  Vec3d p_body_lidar(pi->x, pi->y, pi->z);
  Vec3d p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

CloudPtr lio::PointCloudLidarToIMU(CloudPtr &pi) {
  CloudPtr po{new PointCloudType()};
  size_t size = pi->points.size();
  po->points.resize(size);

  for(int i = 0; i < size; i++) {
    Vec3d p_body_lidar(pi->points[i].x, pi->points[i].y, pi->points[i].z);
    Vec3d p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->points[i].x = (float)p_body_imu[0];
    po->points[i].y = (float)p_body_imu[1];
    po->points[i].z = (float)p_body_imu[2];
    po->points[i].intensity = pi->points[i].intensity;
  }

  return po;
}

void lio::MapIncremental(){
  PointVector points_to_add;
  PointVector point_no_need_down_sample;

  unsigned long cur_pts = scan_down_body_->size();
  //  Number::Record((unsigned long)cur_pts, "ori_point_size");
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
          ((point_world.getVector3fMap() / config_.filter_size_map).array().floor() + 0.5) * config_.filter_size_map;

      Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

      if (fabs(dis_2_center.x()) > 0.5 * config_.filter_size_map &&
          fabs(dis_2_center.y()) > 0.5 * config_.filter_size_map &&
          fabs(dis_2_center.z()) > 0.5 * config_.filter_size_map) {
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

//  unsigned long size = points_to_add.size();
//  if(size != 0){
//    PointCloudType::Ptr match_cloud(new PointCloudType);
//    for (int i = 0; i < size; i++) {
//      match_cloud->points.push_back(points_to_add[i]);
//    }
//
//    sensor_msgs::PointCloud2 laserCloudmsg;
//    pcl::toROSMsg(*match_cloud, laserCloudmsg);
//    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
//    laserCloudmsg.header.frame_id = "camera_init";
//    pub_match_cloud_world_.publish(laserCloudmsg);
//  }
//
//  unsigned long size1 = point_no_need_down_sample.size();
//  if(size1 != 0){
//    PointCloudType::Ptr match_cloud1(new PointCloudType);
//    for (int i = 0; i < size1; i++) {
//      match_cloud1->points.push_back(point_no_need_down_sample[i]);
//    }
//
//    sensor_msgs::PointCloud2 laserCloudmsg1;
//    pcl::toROSMsg(*match_cloud1, laserCloudmsg1);
//    laserCloudmsg1.header.stamp = ros::Time().fromSec(lidar_end_time_);
//    laserCloudmsg1.header.frame_id = "camera_init";
//    pub_match_cloud_world_.publish(laserCloudmsg1);
//  }

  //  Number::Record((unsigned long)points_to_add.size(), "add_point_size");
  //  Number::Record((unsigned long)point_no_need_down_sample.size(), "no_need_ds_point_size");

  if(nn_type_ == nearest_neighbor_type::IKD_TREE){
    Timer::Evaluate([&, this](){
      ikd_tree_.Add_Points(points_to_add, true);
      ikd_tree_.Add_Points(point_no_need_down_sample, false);
    },"    Ikd_Tree Add Points");
  } else if(nn_type_ == nearest_neighbor_type::IVOX){
    Timer::Evaluate([&, this]() {
      ivox_->AddPoints(points_to_add);
      ivox_->AddPoints(point_no_need_down_sample);
    },"    IVox Add Points");
  }
}

void lio::SubAndPubToROS(ros::NodeHandle &nh){
  static ros::Subscriber sub_pcl, sub_imu;
  if (preprocess_->GetLidarType() == LidarType::AVIA) {
    sub_pcl = nh.subscribe<livox_ros_driver::CustomMsg>(config_.lid_topic,
                                                        200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg)
                                                        { LivoxPCLCallBack(msg); });
  } else {
    sub_pcl = nh.subscribe<sensor_msgs::PointCloud2>(config_.lid_topic,
                                                     200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg)
                                                     { StandardPCLCallBack(msg); });
  }
  sub_imu = nh.subscribe<sensor_msgs::Imu>(config_.imu_topic,
                                           200000,[this](const sensor_msgs::Imu::ConstPtr &msg)
                                           { IMUCallBack(msg); });
  
//  pub_odom_ = nh.advertise<nav_msgs::Odometry>(config_.odom_topic, 100);
//  pub_odom_ = nh.advertise<nav_msgs::Odometry>(config_.odom_topic, 100);
//  pub_path_ = nh.advertise<nav_msgs::Path>(config_.path_topic, 100);
//  pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>(config_.cloud_world_topic, 10000);
//  pub_laser_cloud_imu_ = nh.advertise<sensor_msgs::PointCloud2>(config_.cloud_imu_topic, 10000);
//  
  // ROS subscribe initialization
//  std::string lidar_topic, imu_topic;
//  nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
//  nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");
//
//  if (preprocess_->GetLidarType() == LidarType::AVIA) {
//    sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
//        lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
//  } else {
//    sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
//        lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
//  }
//
//  sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
//                                            [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });

  // ROS publisher init
//  path_.header.stamp = ros::Time::now();
//  path_.header.frame_id = "camera_init";

//  pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
//  pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
//  pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
//  pub_match_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/match_cloud_world", 100000);
//  pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
//  pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);

  path_.header.stamp = ros::Time::now();
  path_.header.frame_id = config_.init_frame;
  
  odometry_.header.frame_id = config_.init_frame;
  odometry_.child_frame_id = config_.body_frame;
  
  pub_odom_ = nh.advertise<nav_msgs::Odometry>(config_.odom_topic, 100);
  pub_path_ = nh.advertise<nav_msgs::Path>(config_.path_topic, 100);
  pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>(config_.cloud_world_topic, 10000);
  pub_laser_cloud_imu_ = nh.advertise<sensor_msgs::PointCloud2>(config_.cloud_imu_topic, 10000);
}

bool lio::LoadParams(ros::NodeHandle &nh){
  // get params from param server
  // common params
  nh.param<std::string>("lio_base/lidar_topic", config_.lid_topic, "/livox/lidar");
  nh.param<std::string>("lio_base/imu_topic", config_.imu_topic, "/livox/imu");
  nh.param<bool>("lio_base/time_sync_en", config_.time_sync_en, false);
  
  nh.param<int>("lio_base/lidar_type", config_.lidar_type, 1);
  nh.param<int>("lio_base/scan_line", config_.scan_line, 16);
  nh.param<double>("lio_base/blind", config_.blind, 0.01);
  nh.param<float>("lio_base/time_scale", config_.time_scale, 1e-3);

  nh.param<double>("lio_base/gyr_cov", config_.gyr_cov, 0.1);
  nh.param<double>("lio_base/acc_cov", config_.acc_cov, 0.1);
  nh.param<double>("lio_base/b_gyr_cov", config_.b_gyr_cov, 0.0001);
  nh.param<double>("lio_base/b_acc_cov", config_.b_acc_cov, 0.0001);

  nh.param<std::vector<double>>("lio_base/extrinsic_T", config_.extrinsic_T, std::vector<double>());
  nh.param<std::vector<double>>("lio_base/extrinsic_R", config_.extrinsic_R, std::vector<double>());

  nh.param<int>("lio_base/point_filter_num", config_.point_filter_num, 2);
  nh.param<float>("lio_base/filter_size_surf", config_.filter_size_surf, 0.5);
  nh.param<bool>("lio_base/feature_extract_enable", config_.feature_extract_enable, false);

  // filter params
  nh.param<float>("lio_base/filter_size_map", config_.filter_size_map, 0.0);
  nh.param<bool>("lio_base/extrinsic_est_en", config_.extrinsic_est_en, true);
  nh.param<float>("lio_base/plane_threshold", config_.plane_threshold, 0.1);
  nh.param<int>("lio_base/max_iteration", config_.max_iteration, 4);

  nh.param<int>("lio_base/nearest_neighbor_type", config_.nn_type, 1);

  // frame info params
  nh.param<std::string>("lio_base/body_frame", config_.body_frame, "body");
  nh.param<std::string>("lio_base/init_frame", config_.init_frame, "camera_init");
  nh.param<std::string>("lio_base/odom_topic", config_.odom_topic, "odometry");
  nh.param<std::string>("lio_base/path_topic", config_.path_topic, "path");
  nh.param<std::string>("lio_base/cloud_world_topic", config_.cloud_world_topic, "cloud_registered_world");
  nh.param<std::string>("lio_base/cloud_imu_topic", config_.cloud_imu_topic, "cloud_registered_imu");

  // ivox params
  nh.param<float>("faster_lio/ivox_grid_resolution", config_.resolution, 0.2);
  nh.param<int>("faster_lio/ivox_nearby_type", config_.nearby_type, 18);
  return true;
}

bool lio::LoadParamsFromYAML(const std::string &yaml){
  return true;
}

void lio::PrintState(const state_ikfom &s){
  LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
            << ", off r: " << s.offset_R_L_I.coeffs().transpose() << ", t: " << s.offset_T_L_I.transpose();
}

} // namespace simple_lio