#include <vlcal/calib/extrinsic_visual_camera_calibration.hpp>

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <ceres/autodiff_first_order_function.h>

#include <sophus/se3.hpp>
#include <sophus/ceres_manifold.hpp>

#include <dfo/nelder_mead.hpp>
#include <gtsam/geometry/Pose3.h>

#include <vlcal/costs/extrinsic_nid_cost.hpp>
#include <vlcal/calib/view_culling.hpp>
#include <vlcal/calib/cost_calculator_nid.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <memory>             
#include <Eigen/Core>

namespace vlcal {

EXVisualCameraCalibration::EXVisualCameraCalibration(
  const camera::GenericCameraBase::ConstPtr& proj,
  const std::vector<VisualLiDARData::ConstPtr>& dataset,
  const EXVisualCameraCalibrationParams& params)
: params_(params),
  proj_(proj),
  dataset_(dataset) 
  {

  }

Eigen::Isometry3d EXVisualCameraCalibration::extrinsic_calibrate(const Eigen::Isometry3d& init_T_camera_lidar) {
  Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar;

  // Outer loop
  for (int i = 0; i < params_.max_outer_iterations; i++) {
    Eigen::Isometry3d new_T_camera_lidar;
        
    new_T_camera_lidar = estimate_pose_bfgs(T_camera_lidar);

    const Eigen::Isometry3d delta = new_T_camera_lidar.inverse() * T_camera_lidar;
    T_camera_lidar = new_T_camera_lidar;

    const double delta_t = delta.translation().norm();
    const double delta_r = Eigen::AngleAxisd(delta.linear()).angle();
    const bool converged = delta_t < params_.delta_trans_thresh && delta_r < params_.delta_rot_thresh;

    std::stringstream sst;
    sst << boost::format("delta_t: %.3f [m]  delta_r: %.3f [rad]") % delta_t % delta_r << std::endl;
    sst << (converged ? "Outer loop converged" : "Re-run inner optimization with the new viewpoint");
    guik::LightViewer::instance()->append_text(sst.str());

    if (converged) {
      break;
    }
  }

  return T_camera_lidar;
}

struct MultiNIDCost {
public:
  MultiNIDCost(const Sophus::SE3d& init_T_camera_lidar) : init_T_camera_lidar(init_T_camera_lidar) {}

  void add(const std::shared_ptr<EXNIDCost>& cost) { costs.emplace_back(cost); }

  template <typename T>
  bool operator()(const T* params, T* residual) const {
    std::vector<double> values(Sophus::SE3d::num_parameters);
    std::transform(params, params + Sophus::SE3d::num_parameters, values.begin(), [](const auto& x) { return get_real(x); });
    const Eigen::Map<const Sophus::SE3d> T_camera_lidar(values.data());
    const Sophus::SE3d delta = init_T_camera_lidar.inverse() * T_camera_lidar;

    if (delta.translation().norm() > 0.2 || Eigen::AngleAxisd(delta.rotationMatrix()).angle() > 2.0 * M_PI / 180.0) {
      return false;
    }

    std::vector<bool> results(costs.size());
    std::vector<T> residuals(costs.size());

#pragma omp parallel for
    for (int i = 0; i < costs.size(); i++) {
      results[i] = (*costs[i])(params, &residuals[i]);
    }

    for (int i = 1; i < costs.size(); i++) {
      residuals[0] += residuals[i];
    }

    *residual = residuals[0];

    return std::count(results.begin(), results.end(), false) == 0;
  }

private:
  Sophus::SE3d init_T_camera_lidar;
  std::vector<std::shared_ptr<EXNIDCost>> costs;
};

struct IterationCallbackWrapper : public ceres::IterationCallback {
public:
  IterationCallbackWrapper(const std::function<ceres::CallbackReturnType(const ceres::IterationSummary&)>& callback) : callback(callback) {}

  virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) { return callback(summary); }

private:
  std::function<ceres::CallbackReturnType(const ceres::IterationSummary&)> callback;
};

Eigen::Isometry3d EXVisualCameraCalibration::estimate_pose_bfgs(const Eigen::Isometry3d& init_T_camera_lidar) {
  ViewCullingParams view_culling_params;
  view_culling_params.enable_depth_buffer_culling = !params_.disable_z_buffer_culling;

  Eigen::Vector2i image_size(dataset_.front()->image.cols, dataset_.front()->image.rows);

  ViewCulling view_culling(proj_, image_size, view_culling_params);

  Sophus::SE3d T_camera_lidar(init_T_camera_lidar.matrix());

  std::vector<std::shared_ptr<EXNIDCost>> nid_costs;

  for (int i = 0; i < dataset_.size(); i++) {
    // Remove hidden points
    auto culled_points = view_culling.cull(dataset_[i]->points, init_T_camera_lidar);

    cv::Mat normalized_image;
    dataset_[i]->image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0);

    auto nid_cost = std::make_shared<EXNIDCost>(proj_, normalized_image, culled_points, params_.nid_bins);
    nid_costs.emplace_back(nid_cost);
  }

    auto sum_nid = new MultiNIDCost(T_camera_lidar);
  for (const auto& nid_cost : nid_costs) {
  sum_nid->add(nid_cost);
  }

  auto complete_cost = std::make_unique<MultiNIDCost>(T_camera_lidar);
  *complete_cost = *sum_nid;  
  delete sum_nid;  

  // auto cost = new ceres::AutoDiffFirstOrderFunction<MultiNIDCost, Sophus::SE3d::num_parameters>(
  //   std::move(complete_cost)
  // );

  MultiNIDCost* raw_functor = complete_cost.release();
  auto* cost = new ceres::AutoDiffFirstOrderFunction<
    MultiNIDCost,
    Sophus::SE3d::num_parameters
  >(raw_functor);

  ceres::GradientProblem problem(
    cost,
    new Sophus::Manifold<Sophus::SE3>()
  );

  ceres::GradientProblemSolver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.update_state_every_iteration = true;
  options.line_search_direction_type = ceres::BFGS;

  options.callbacks.emplace_back(new IterationCallbackWrapper([&](const ceres::IterationSummary& summary) {
    params_.callback(Eigen::Isometry3d(T_camera_lidar.matrix()));
    return ceres::CallbackReturnType::SOLVER_CONTINUE;
  }));

  //shen BFGS
  ceres::GradientProblemSolver::Summary summary;
  ceres::Solve(options, problem, T_camera_lidar.data(), &summary);

  std::stringstream sst;
  sst << boost::format("Inner optimization (BFGS) terminated after %d iterations") % summary.iterations.size() << std::endl;
  sst << boost::format("Final cost: %.3f") % summary.final_cost << std::endl;
  sst << "--- T_camera_lidar ---" << std::endl << T_camera_lidar.matrix();
  guik::LightViewer::instance()->append_text(sst.str());

  return Eigen::Isometry3d(T_camera_lidar.matrix());
}

}  // namespace vlcal
