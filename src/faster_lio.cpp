//
// Created by myx on 2025/3/5.
//

#include "faster_lio.h"

#include <ros/ros.h>

namespace simple_lio{

faster_lio::~faster_lio(){
  scan_down_body_ = nullptr;
  scan_undistort_ = nullptr;
  scan_down_world_ = nullptr;
  LOG(INFO) << "lio_test deconstruct";
}


// init with ros
bool faster_lio::Init(ros::NodeHandle &nh) {
  LioBase::Init(nh);

  // ivox params
  nh.param<float>("faster_lio/ivox_grid_resolution", faster_lio_cfg_.resolution, 0.2);
  nh.param<int>("faster_lio/ivox_nearby_type", faster_lio_cfg_.nearby_type, 18);

  // filter params
  nh.param<float>("faster_lio/filter_size_map", faster_lio_cfg_.map_filter_size, 0.0);
  nh.param<bool>("faster_lio/extrinsic_est_en", faster_lio_cfg_.extrinsic_est_en, true);
  nh.param<float>("faster_lio/plane_threshold", faster_lio_cfg_.plane_threshold, 0.1);
  nh.param<int>("faster_lio/max_iteration", faster_lio_cfg_.max_iteration, 4);

  SubAndPubToROS(nh);

  ivox_options_.resolution_ = faster_lio_cfg_.resolution;
  if (faster_lio_cfg_.nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (faster_lio_cfg_.nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (faster_lio_cfg_.nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (faster_lio_cfg_.nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }
  ivox_ = std::make_shared<IVoxType>(ivox_options_);
  voxel_scan_.setLeafSize(lio_config_.filter_size_surf,
                          lio_config_.filter_size_surf,
                          lio_config_.filter_size_surf);
  // esekf init
  std::vector<double> epsi(23, 0.001);
  kf_.init_dyn_share(
      get_f, df_dx, df_dw,
      [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
      faster_lio_cfg_.max_iteration, epsi.data());
  return true;
}

void faster_lio::SubAndPubToROS(ros::NodeHandle &nh){
  // TODO: (myx) The lio will drift when those functions be set in lio_base.
  // I don't know why.
  static ros::Subscriber sub_pcl, sub_imu;
  if (preprocess_->GetLidarType() == LidarType::AVIA) {
    sub_pcl = nh.subscribe<livox_ros_driver::CustomMsg>(lio_config_.lid_topic,
                                                        200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg)
                                                        { LivoxPCLCallBack(msg); });
  } else {
    sub_pcl = nh.subscribe<sensor_msgs::PointCloud2>(lio_config_.lid_topic,
                                                     200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg)
                                                     { StandardPCLCallBack(msg); });
  }
  sub_imu = nh.subscribe<sensor_msgs::Imu>(lio_config_.imu_topic,
                                           200000,[this](const sensor_msgs::Imu::ConstPtr &msg)
                                           { IMUCallBack(msg); });

  pub_odom_ = nh.advertise<nav_msgs::Odometry>(lio_config_.odom_topic, 100);
  pub_path_ = nh.advertise<nav_msgs::Path>(lio_config_.path_topic, 100);
  pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>(lio_config_.cloud_world_topic, 10000);
  pub_laser_cloud_imu_ = nh.advertise<sensor_msgs::PointCloud2>(lio_config_.cloud_imu_topic, 10000);
}


void faster_lio::Run(){
  if (!SyncPackages()) {
    return;
  }

  // IMU process, kf prediction, undistortion
  p_imu_->Process(measures_, kf_, scan_undistort_);

  if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
    LOG(WARNING) << "No point, skip this scan!";
    return;
  }

  // down_sample
  voxel_scan_.setInputCloud(scan_undistort_);
  voxel_scan_.filter(*scan_down_body_);
  unsigned long cur_pts = scan_down_body_->size();
  if (cur_pts <= 5) {
    LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size()
                 << ", " << scan_down_body_->size();
    return;
  }
  if(flg_first_scan_){
    ivox_->AddPoints(scan_undistort_->points);
    first_lidar_time_ = measures_.lidar_beg_time_;

    scan_down_world_->resize(cur_pts);
    for(int i = 0; i < cur_pts; i++)
    {
      PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));
    }
    flg_first_scan_ = false;

    LIOOutput lio_output(frame_id_, lidar_end_time_, State2SE3(state_point_),
                         PointCloudLidarToIMU(scan_down_body_), scan_down_body_);
    if(lio_process_func_){
      lio_process_func_(lio_output);
    }

    frame_id_++;
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
  kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
  // save the state
  state_point_ = kf_.get_x();

  // update local map
  MapIncremental();

  LIOOutput lio_output(frame_id_, lidar_end_time_, State2SE3(state_point_),
                       scan_down_body_, PointCloudLidarToIMU(scan_down_body_));
  if(lio_process_func_){
    lio_process_func_(lio_output);
  }
  frame_id_++;

  PublishPath(State2SE3(state_point_));
  PublishOdom(State2SE3(state_point_));

  PublishPointCloud(PointCloudLidarToIMU(scan_down_body_),
                    lio_config_.body_frame, pub_laser_cloud_imu_);
  PublishPointCloud(scan_down_world_, lio_config_.init_frame,
                    pub_laser_cloud_world_);
}


// interface of mtk, customized observation model
void faster_lio::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data){
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

    auto &points_near = nearest_points_[i];
    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      ivox_->GetClosestPoint(point_world, points_near, NUM_MATCH_POINTS);
      point_selected_surf_[i] = points_near.size() >= MIN_NUM_MATCH_POINTS;

      if (point_selected_surf_[i]) {
        point_selected_surf_[i] = esti_plane(plane_coef_[i], points_near, faster_lio_cfg_.plane_threshold);
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

    if (faster_lio_cfg_.extrinsic_est_en) {
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


// private
SopSE3 faster_lio::State2SE3(state_ikfom state){
  SopSE3 se_3 = SopSE3(state.rot, state.pos);
  return se_3;
}

void faster_lio::PointBodyToWorld(const PointType *pi, PointType *const po) {
  Vec3d p_body(pi->x, pi->y, pi->z);
  Vec3d p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                 state_point_.pos);

  po->x = (float)p_global(0);
  po->y = (float)p_global(1);
  po->z = (float)p_global(2);
  po->intensity = pi->intensity;
}

void faster_lio::PointBodyToWorld(const Vec3f &pi, PointType *const po) {
  Vec3d p_body(pi.x(), pi.y(), pi.z());
  Vec3d p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                 state_point_.pos);

  po->x = (float)p_global(0);
  po->y = (float)p_global(1);
  po->z = (float)p_global(2);
  po->intensity = std::abs(po->z);
}

CloudPtr faster_lio::PointCloudLidarToIMU(CloudPtr &pi) {
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

void faster_lio::MapIncremental(){
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
          ((point_world.getVector3fMap() / faster_lio_cfg_.map_filter_size).array().floor() + 0.5) * faster_lio_cfg_.map_filter_size;

      Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

      if (fabs(dis_2_center.x()) > 0.5 * faster_lio_cfg_.map_filter_size &&
          fabs(dis_2_center.y()) > 0.5 * faster_lio_cfg_.map_filter_size &&
          fabs(dis_2_center.z()) > 0.5 * faster_lio_cfg_.map_filter_size) {
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

  ivox_->AddPoints(points_to_add);
  ivox_->AddPoints(point_no_need_down_sample);
}


} // namespace simple_lio
