#pragma once

#include <memory>
#include <iostream>
#include <Eigen/Core>
#include <camera/traits.hpp>
#include <camera/generic_camera_base.hpp>

namespace ceres {
template <typename T, int N>
struct Jet;
}  // namespace ceres

namespace camera {

template <typename Projection>
class GenericCamera : public GenericCameraBase {
public:
  GenericCamera(const Eigen::VectorXd& intrinsics, const Eigen::VectorXd& distortion_coeffs)
      : intrinsic(intrinsics), distortion(distortion_coeffs)
      {}
  
  // shen 实现获取内参和畸变系数的方法
  virtual Eigen::VectorXd get_intrinsics() const override {
    // std::cout << "get_intrinsics called, size: " << intrinsic.size() << std::endl;
    return intrinsic;
  }

  virtual Eigen::VectorXd get_distortion_coeffs() const override {
      return distortion;
  }

  virtual Eigen::Vector2d project(const Eigen::Vector3d& point_3d) const override { //
    return (*this)(point_3d);
  }

  virtual Eigen::Vector2d operator()(const Eigen::Vector3d& point_3d) const override {
    Projection proj;
    return proj(intrinsic.data(), distortion.data(), point_3d);
  }

  virtual Eigen::Matrix<ceres::Jet<double, 7>, 2, 1> operator()(const Eigen::Matrix<ceres::Jet<double, 7>, 3, 1>& point_3d) const override {
    Projection proj;
    return proj(intrinsic.data(), distortion.data(), point_3d);
  }

private:
  Eigen::VectorXd intrinsic;
  Eigen::VectorXd distortion;
};

}  // namespace camera