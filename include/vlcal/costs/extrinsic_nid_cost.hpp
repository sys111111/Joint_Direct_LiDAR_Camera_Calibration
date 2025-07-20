#pragma once

#include <algorithm>
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <vlcal/common/frame.hpp>
#include <camera/generic_camera_base.hpp>

namespace vlcal {

template <typename T>
inline double get_real(const T& x) {
  return x.a;
}

template <>
inline double get_real(const double& x) {
  return x;
}

class EXNIDCost {
public:
  EXNIDCost(const camera::GenericCameraBase::ConstPtr& proj,
            const cv::Mat& normalized_image,
            const Frame::ConstPtr& points,
            const int bins = 16)
    : proj_(proj),
      normalized_image_(normalized_image.clone()),
      points_(points),
      bins_(bins)
  {
    spline_coeffs.row(0) << 1.0, -3.0, 3.0, -1.0;
    spline_coeffs.row(1) << 4.0,  0.0, -6.0, 3.0;
    spline_coeffs.row(2) << 1.0,  3.0,  3.0, -3.0;
    spline_coeffs.row(3) << 0.0,  0.0,  0.0,  1.0;
    spline_coeffs /= 6.0;
  }

  template <typename T>
  bool operator()(const T* T_camera_lidar_params, T* residual) const {
    const Eigen::Map<Sophus::SE3<T> const> T_camera_lidar(T_camera_lidar_params);

    Eigen::Matrix<T, -1, -1> hist = Eigen::Matrix<T, -1, -1>::Zero(bins_, bins_);
    Eigen::Matrix<T, -1, 1>   hist_image  = Eigen::Matrix<T, -1, 1>::Zero(bins_);
    Eigen::VectorXd           hist_points = Eigen::VectorXd::Zero(bins_);

    int num_outliers = 0;
    for (int i = 0; i < points_->size(); i++) {
      const Eigen::Matrix<T, 3, 1> pt_camera =
        T_camera_lidar * points_->points[i].head<3>();
      const double intensity = points_->intensities[i];
      const int bin_points = std::max<int>(0, std::min<int>(bins_ - 1, intensity * bins_));

      const Eigen::Matrix<T, 2, 1> projected = (*proj_)(pt_camera);
      const Eigen::Vector2i knot_i(
        std::floor(get_real(projected[0])),
        std::floor(get_real(projected[1])));
      const Eigen::Matrix<T, 2, 1> s = projected - knot_i.cast<double>();

      if ((knot_i.array() < Eigen::Array2i(0, 0)).any() ||
          (knot_i.array() >= Eigen::Array2i(
            normalized_image_.cols,
            normalized_image_.rows)).any()) {
        num_outliers++;
        continue;
      }

      hist_points[bin_points]++;

      Eigen::Matrix<T, 4, 2> se;
      se.row(0).setOnes();
      se.row(1) = s.transpose();
      se.row(2) = s.array().square().transpose();
      se.row(3) = (s.array().square() * s.array()).transpose();

      const Eigen::Matrix<T, 4, 2> beta = spline_coeffs * se;

      Eigen::Array4i knots_x(knot_i.x() - 1, knot_i.x(), knot_i.x() + 1, knot_i.x() + 2);
      Eigen::Array4i knots_y(knot_i.y() - 1, knot_i.y(), knot_i.y() + 1, knot_i.y() + 2);
      knots_x = knots_x.max(0).min(normalized_image_.cols - 1);
      knots_y = knots_y.max(0).min(normalized_image_.rows - 1);

      for (int ii = 0; ii < 4; ii++) {
        for (int jj = 0; jj < 4; jj++) {
          const T w = beta(ii, 0) * beta(jj, 1);
          const double pix = normalized_image_.at<double>(knots_y[jj], knots_x[ii]);
          const int bin_image = std::min<int>(pix * bins_, bins_ - 1);
          hist(bin_image, bin_points) += w;
          hist_image[bin_image]       += w;
        }
      }
    }

    const double sum = hist_points.sum();

    hist_image = hist_image / sum;
    hist_points = hist_points / sum;
    hist = hist / sum;

    const T H_image = -(hist_image.array() * (hist_image.array() + 1e-6).log()).sum();
    const double H_points = -(hist_points.array() * (hist_points.array() + 1e-6).log()).sum();
    const T H_image_points = -(hist.array() * (hist.array() + 1e-6).log()).sum();
    const T MI = H_image + H_points - H_image_points;
    const T NID = (H_image_points - MI) / H_image_points;

    if (!std::isfinite(get_real(NID))) {
      std::cout << get_real(H_image_points) << " "
                << get_real(MI)            << " "
                << get_real(NID)           << std::endl;
      return false;
    }

    residual[0] = NID;
    return true;
  }

private:
  const camera::GenericCameraBase::ConstPtr proj_;
  const cv::Mat normalized_image_;
  const Frame::ConstPtr points_;
  const int bins_;
  Eigen::Matrix<double, 4, 4> spline_coeffs;
};

}  // namespace vlcal
