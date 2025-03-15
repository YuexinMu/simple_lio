#ifndef FASTER_LIO_IMU_PROCESSING_H
#define FASTER_LIO_IMU_PROCESSING_H

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <deque>
#include <fstream>

#include "common_lib.h"
#include "so3_math.h"
#include "use-ikfom.hpp"
#include "utils.h"

namespace simple_lio {
constexpr int MAX_INI_COUNT = 20;

/// IMU Process and undistortion
class ImuProcess {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void SetExtrinsic(const Vec3d &transl, const Mat3d &rot);
  void SetGyrCov(const Vec3d &scaler);
  void SetAccCov(const Vec3d &scaler);
  void SetGyrBiasCov(const Vec3d &b_g);
  void SetAccBiasCov(const Vec3d &b_a);
  void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
               PointCloudType::Ptr cur_pcl_un);

  std::ofstream fout_imu_;
  Eigen::Matrix<double, 12, 12> Q_;
  Vec3d cov_acc_;
  Vec3d cov_gyr_;
  Vec3d cov_acc_scale_;
  Vec3d cov_gyr_scale_;
  Vec3d cov_bias_gyr_;
  Vec3d cov_bias_acc_;

private:
  void IMUInit(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                    PointCloudType &pcl_out);

  PointCloudType::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  std::deque<sensor_msgs::ImuConstPtr> v_imu_;
  std::vector<Pose6D> IMUpose_;
  std::vector<Mat3d> v_rot_pcl_;
  Mat3d Lidar_R_wrt_IMU_;
  Vec3d Lidar_T_wrt_IMU_;
  Vec3d mean_acc_;
  Vec3d mean_gyr_;
  Vec3d angvel_last_;
  Vec3d acc_s_last_;
  double last_lidar_end_time_ = 0;
  int init_iter_num_ = 1;
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;
};

}  // namespace simple_lio

#endif
