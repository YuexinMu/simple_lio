#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/array.hpp>
#include <unsupported/Eigen/ArpackSupport>
#include <deque>

#include "so3_math.h"

#include "sophus/se2.hpp"
#include "sophus/se3.hpp"

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

struct FullPointType {
  PCL_ADD_POINT4D;
  float range = 0;
  float radius = 0;
  float intensity = 0;
  uint8_t ring = 0;
  uint8_t angle = 0;
  double time = 0;
  float height = 0;

  inline FullPointType() {}
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace simple_lio{
const double INIT_TIME = 0.1;
const double LASER_POINT_COV = 0.001;
const int NUM_MATCH_POINTS = 5;
const int MIN_NUM_MATCH_POINTS = 3;

using Vec2d = Eigen::Vector2d;
using Vec2f = Eigen::Vector2f;
using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Vec4d = Eigen::Vector4d;
using Vec4f = Eigen::Vector4f;
using Vec5d = Eigen::Matrix<double, 5, 1>;
using Vec5f = Eigen::Matrix<float, 5, 1>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec9d = Eigen::Matrix<double, 9, 1>;
using Vec15d = Eigen::Matrix<double, 15, 15>;
using Vec18d = Eigen::Matrix<double, 18, 1>;

using VVec3D = std::vector<Vec3d, Eigen::aligned_allocator<Vec3d>>;
using VVec3F = std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>>;
using VVec4F = std::vector<Vec4f, Eigen::aligned_allocator<Vec4f>>;
using VVec4D = std::vector<Vec4d, Eigen::aligned_allocator<Vec4d>>;
using VVec5F = std::vector<Vec5f, Eigen::aligned_allocator<Vec5f>>;
using VVec5D = std::vector<Vec5d, Eigen::aligned_allocator<Vec5d>>;

using Mat1d = Eigen::Matrix<double, 1, 1>;
using Mat2d = Eigen::Matrix<double, 2, 2>;
using Mat23d = Eigen::Matrix<double, 2, 3>;
using Mat32d = Eigen::Matrix<double, 3, 2>;
using Mat3d = Eigen::Matrix3d;
using Mat3f = Eigen::Matrix3f;
using Mat4d = Eigen::Matrix4d;
using Mat4f = Eigen::Matrix4f;
using Mat5d = Eigen::Matrix<double, 5, 5>;
using Mat5f = Eigen::Matrix<float, 5, 5>;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using Mat6f = Eigen::Matrix<float, 6, 6>;
using Mat9d = Eigen::Matrix<double, 9, 9>;
using Mat96d = Eigen::Matrix<double, 9, 6>;
using Mat15d = Eigen::Matrix<double, 15, 15>;
using Mat18d = Eigen::Matrix<double, 18, 18>;

using VecXd = Eigen::Matrix<double, -1, 1>;
using MatXd = Eigen::Matrix<double, -1, -1>;
using MatX18d = Eigen::Matrix<double, -1, 18>;

using Quatd = Eigen::Quaterniond;
using Quatf = Eigen::Quaternionf;

const Mat3d Eye3d = Mat3d::Identity();
const Mat3f Eye3f = Mat3f::Identity();
const Vec3d Zero3d(0, 0, 0);
const Vec3f Zero3f(0, 0, 0);

// pose represented as sophus structs
using SopSE2 = Sophus::SE2d;
using SopSE2f = Sophus::SE2f;
using SopSO2 = Sophus::SO2d;
using SopSE3 = Sophus::SE3d;
using SopSE3f = Sophus::SE3f;
using SopSO3 = Sophus::SO3d;

using IdType = unsigned long;

struct PoseTime {
  PoseTime() = default;
  PoseTime(double t, const SopSE3& p) : timestamp(t), opti_pose(p){}
  PoseTime(double t, const SopSE3& p1, const SopSE3& p2) : timestamp(t), opti_pose(p1), relative_pose(p2){}
  double timestamp = 0;
  SopSE3 opti_pose = SopSE3();
  SopSE3 relative_pose = SopSE3();
};

constexpr double G_m_s2 = 9.81;  // Gravity const in GuangDong/China

inline std::string DEBUG_FILE_DIR(const std::string &name) { return std::string(ROOT_DIR) + "log/" + name; }

template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const std::vector<double> &v) {
  return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}
template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const Eigen::Matrix<S, 3, 1> &v) {
  return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}
template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const boost::array<S, 3> &v) {
  return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}

template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const Eigen::Matrix<S, 9, 1> &v) {
  Eigen::Matrix<S, 3, 3> m;
  m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
  return m;
}
template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const std::vector<double> &v) {
  Eigen::Matrix<S, 3, 3> m;
  m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
  return m;
}
template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const boost::array<S, 9> &v) {
  Eigen::Matrix<S, 3, 3> m;
  m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
  return m;
}

/// xt: 经验证，此转换与ros中tf::createQuaternionFromRPY结果一致
template <typename T>
Eigen::Matrix<T, 3, 3> RpyToRotM2(const T r, const T p, const T y) {
  using AA = Eigen::AngleAxis<T>;
  using Vec3 = Eigen::Matrix<T, 3, 1>;
  return Eigen::Matrix<T, 3, 3>(AA(y, Vec3::UnitZ()) * AA(p, Vec3::UnitY()) * AA(r, Vec3::UnitX()));
}

struct Pose6D{
  Pose6D() = default;
  double offset_time = 0.0;
  Vec3d acc;
  Vec3d gyr;
  Vec3d vel;
  Vec3d pos;
  Vec9d rot;
};

/// sync imu and lidar measurements
struct MeasureGroup {
  MeasureGroup() { this->lidar_.reset(new PointCloudType()); };

  double lidar_beg_time_ = 0;
  double lidar_end_time_ = 0;
  PointCloudType::Ptr lidar_ = nullptr;
  std::deque<sensor_msgs::Imu::ConstPtr> imu_;
};

/**
 * set a pose 6d from ekf status
 * @tparam T
 * @param t
 * @param a
 * @param g
 * @param v
 * @param p
 * @param R
 * @return
 */
template <typename T>
Pose6D set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                  const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R) {
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++) {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
  }
  return rot_kp;
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec_:  normalized x0
*/
/**
 * 计算一组点的法线
 * @tparam T
 * @param normvec
 * @param point
 * @param threshold
 * @param point_num
 * @return
 */
template <typename T>
bool esti_normvector(Eigen::Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold,
                     const int &point_num) {
  Eigen::MatrixXf A(point_num, 3);
  Eigen::MatrixXf b(point_num, 1);
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < point_num; j++) {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }
  normvec = A.colPivHouseholderQr().solve(b);

  for (int j = 0; j < point_num; j++) {
    if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold) {
      return false;
    }
  }

  normvec.normalize();
  return true;
}

/**
 * squared distance
 * @param p1
 * @param p2
 * @return
 */
inline float calc_dist(const PointType &p1, const PointType &p2) {
  return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}

inline float calc_dist(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) { return (p1 - p2).squaredNorm(); }

/**
 * estimate a plane
 * @tparam T
 * @param pca_result
 * @param point
 * @param threshold
 * @return
 */
template <typename T>
inline bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold = 0.1f) {
  if (point.size() < MIN_NUM_MATCH_POINTS) {
    return false;
  }

  Eigen::Matrix<T, 3, 1> normvec;

  if (point.size() == NUM_MATCH_POINTS) {
    Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;

    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
      A(j, 0) = point[j].x;
      A(j, 1) = point[j].y;
      A(j, 2) = point[j].z;
    }

    normvec = A.colPivHouseholderQr().solve(b);
  } else {
    Eigen::MatrixXd A(point.size(), 3);
    Eigen::VectorXd b(point.size(), 1);

    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point.size(); j++) {
      A(j, 0) = point[j].x;
      A(j, 1) = point[j].y;
      A(j, 2) = point[j].z;
    }

    Eigen::MatrixXd n = A.colPivHouseholderQr().solve(b);
    normvec(0, 0) = n(0, 0);
    normvec(1, 0) = n(1, 0);
    normvec(2, 0) = n(2, 0);
  }

  T n = normvec.norm();
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;

  for (const auto &p : point) {
    Eigen::Matrix<T, 4, 1> temp = p.getVector4fMap();
    temp[3] = 1.0;
    if (fabs(pca_result.dot(temp)) > threshold) {
      return false;
    }
  }
  return true;
}

template <typename S>
inline SopSE3 Mat4ToSE3(const Eigen::Matrix<S, 4, 4>& m) {
  /// 对R做归一化，防止sophus里的检查不过
  Quatd q(m.template block<3, 3>(0, 0).template cast<double>());
  q.normalize();
  return SopSE3(q, m.template block<3, 1>(0, 3).template cast<double>());
}

} // namespace simple_lio

namespace simple_lio::math{
// 常量定义
constexpr double kDEG2RAD = M_PI / 180.0;  // deg->rad
constexpr double kRAD2DEG = 180.0 / M_PI;  // rad -> deg
constexpr double G_m_s2 = 9.81;            // 重力大小

template <typename T>
T rad2deg(const T &radians) {
  return radians * 180.0 / M_PI;
}

template <typename T>
T deg2rad(const T &degrees) {
  return degrees * M_PI / 180.0;
}


}


#endif