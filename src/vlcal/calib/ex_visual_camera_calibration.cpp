#include <iostream>
#include <typeinfo>

#include <vlcal/calib/visual_camera_calibration.hpp>

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

#include <vlcal/costs/nid_cost.hpp>
#include <vlcal/calib/view_culling.hpp>
#include <vlcal/calib/cost_calculator_nid.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace vlcal {

VisualCameraCalibration::VisualCameraCalibration(
  const camera::GenericCameraBase::ConstPtr& proj,
  const std::vector<VisualLiDARData::ConstPtr>& dataset,
  const VisualCameraCalibrationParams& params)
: params(params),
  proj(proj),
  dataset(dataset) {}

Eigen::Isometry3d VisualCameraCalibration::calibrate(const Eigen::Isometry3d& init_T_camera_lidar) { 
  Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar;
  // Outer loop
  for (int i = 0; i < params.max_outer_iterations; i++) {
    Eigen::Isometry3d new_T_camera_lidar;
    // 0TODO 修改其中一个 加一个 内参 函数
    std::cout << "start estimate_intrinsic_bfgs" << std::endl;
    new_T_camera_lidar = estimate_intrinsic_bfgs(T_camera_lidar);

    const Eigen::Isometry3d delta = new_T_camera_lidar.inverse() * T_camera_lidar;
    T_camera_lidar = new_T_camera_lidar;

    const double delta_t = delta.translation().norm();
    const double delta_r = Eigen::AngleAxisd(delta.linear()).angle();
    const bool converged = delta_t < params.delta_trans_thresh && delta_r < params.delta_rot_thresh;

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

//shen nelder_mead优化
Eigen::Isometry3d VisualCameraCalibration::estimate_pose_nelder_mead(const Eigen::Isometry3d& init_T_camera_lidar) {
  ViewCullingParams view_culling_params;
  view_culling_params.enable_depth_buffer_culling = !params.disable_z_buffer_culling;
  ViewCulling view_culling(proj, {dataset.front()->image.cols, dataset.front()->image.rows}, view_culling_params);

  std::vector<CostCalculator::Ptr> costs;
  for (const auto& data : dataset) {
    // Remove hidden points
    auto culled_points = view_culling.cull(data->points, init_T_camera_lidar);
    auto new_data = std::make_shared<VisualLiDARData>(data->image, culled_points);

    // Create NID cost
    NIDCostParams nid_params;
    nid_params.bins = params.nid_bins;

    costs.emplace_back(std::make_shared<CostCalculatorNID>(proj, new_data, nid_params));

  }

  double best_cost = std::numeric_limits<double>::max();

  const auto f = [&](const gtsam::Vector6& x) {
    const Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar * Eigen::Isometry3d(gtsam::Pose3::Expmap(x).matrix());
    double sum_costs = 0.0;

#pragma omp parallel for reduction(+ : sum_costs)
    for (int i = 0; i < costs.size(); i++) {
      sum_costs += costs[i]->calculate(T_camera_lidar);
    }

    if (sum_costs < best_cost) {
      best_cost = sum_costs;
      params.callback(T_camera_lidar);
      std::cout << "cost:" << best_cost << std::endl;
    }

    return sum_costs;
  };

  // Optimize
  dfo::NelderMead<6>::Params nelder_mead_params;
  nelder_mead_params.init_step = params.nelder_mead_init_step;
  nelder_mead_params.convergence_var_thresh = params.nelder_mead_convergence_criteria;
  nelder_mead_params.max_iterations = params.max_inner_iterations;

  // shen 单纯形法 优化（）
  dfo::NelderMead<6> optimizer(nelder_mead_params);
  auto result = optimizer.optimize(f, gtsam::Vector6::Zero());

  const Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar * Eigen::Isometry3d(gtsam::Pose3::Expmap(result.x).matrix());

  std::stringstream sst;
  sst << boost::format("Inner optimization (Nelder-Mead) terminated after %d iterations") % result.num_iterations << std::endl;
  sst << boost::format("Final cost: %.3f") % result.y << std::endl;
  sst << "--- T_camera_lidar ---" << std::endl << T_camera_lidar.matrix();

  guik::LightViewer::instance()->append_text(sst.str());

  return T_camera_lidar;
}

struct MultiNIDCost {
public:
  MultiNIDCost(const Sophus::SE3d& init_T_camera_lidar) : init_T_camera_lidar(init_T_camera_lidar) {}

  void add(const std::shared_ptr<NIDCost>& cost) { costs.emplace_back(cost); }

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
  std::vector<std::shared_ptr<NIDCost>> costs;
};

struct IterationCallbackWrapper : public ceres::IterationCallback {
public:
  IterationCallbackWrapper(const std::function<ceres::CallbackReturnType(const ceres::IterationSummary&)>& callback) : callback(callback) {}

  virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) { return callback(summary); }

private:
  std::function<ceres::CallbackReturnType(const ceres::IterationSummary&)> callback;
};

 // 0TODO 这边需要这个重写 estimate_pose_bfgs 叫estimate_intrinsic_bfgs吧？
Eigen::Isometry3d VisualCameraCalibration::estimate_intrinsic_bfgs(const Eigen::Isometry3d& init_T_camera_lidar) {
  ViewCullingParams view_culling_params;
  view_culling_params.enable_depth_buffer_culling = !params.disable_z_buffer_culling;

  // sheng 内参 在proj 已经传进来了
  Eigen::VectorXd intrinsics_out = proj->get_intrinsics();
  Eigen::VectorXd distortion_out = proj->get_distortion_coeffs();

  ViewCulling view_culling(proj, {dataset.front()->image.cols, dataset.front()->image.rows}, view_culling_params);
  
  // sheng 外参初始化 在这边
  Sophus::SE3d T_camera_lidar(init_T_camera_lidar.matrix());
  std::vector<std::shared_ptr<NIDCost>> nid_costs;

   for (int i = 0; i < dataset.size(); i++) {
    // Remove hidden points
    auto culled_points = view_culling.cull(dataset[i]->points, init_T_camera_lidar);

    cv::Mat normalized_image;
    dataset[i]->image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0);

    std::shared_ptr<NIDCost> nid_cost(new NIDCost(proj, normalized_image, culled_points, params.nid_bins));
    nid_costs.emplace_back(nid_cost);
  }
  
  auto sum_nid = new MultiNIDCost(T_camera_lidar);
  for (const auto& nid_cost : nid_costs) {
  sum_nid->add(nid_cost);
  }

  // 创建完备的代价函数拷贝
  auto complete_cost = std::make_unique<MultiNIDCost>(T_camera_lidar);
  // auto complete_cost = std::make_unique<MultiNIDCost>(init_intrinsics, init_T_camera_lidar);
  *complete_cost = *sum_nid;  // 拷贝所有数据
  delete sum_nid;  // 释放临时对象

  // 创建自动求导的一阶函数
  // auto cost = new ceres::AutoDiffFirstOrderFunction<MultiNIDCost, Sophus::SE3d::num_parameters>(
  //   std::move(complete_cost)
  // );

  auto cost = new ceres::AutoDiffFirstOrderFunction<MultiNIDCost, Sophus::SE3d::num_parameters>(
    complete_cost.release()
  );

  // 构造梯度问题
  ceres::GradientProblem problem(
    cost,
    new Sophus::Manifold<Sophus::SE3>()
  );

  ceres::GradientProblemSolver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.update_state_every_iteration = false;
  options.line_search_direction_type = ceres::BFGS;
  // 0TODO 修改这边的待优化变量    params.callback(Eigen::Isometry3d(T_camera_lidar.matrix()));
 
  options.callbacks.emplace_back(new IterationCallbackWrapper([&](const ceres::IterationSummary& summary) {
    params.callback(Eigen::Isometry3d(T_camera_lidar.matrix()));
    return ceres::CallbackReturnType::SOLVER_CONTINUE;
  }));

  //shen BFGS优化
  ceres::GradientProblemSolver::Summary summary;
  ceres::Solve(options, problem, T_camera_lidar.data(), &summary);

  std::cout << "BFGS Finish!" << std::endl;

  std::stringstream sst;
  sst << boost::format("Inner optimization (BFGS) terminated after %d iterations") % summary.iterations.size() << std::endl;
  sst << boost::format("Final cost: %.3f") % summary.final_cost << std::endl;
  sst << "--- T_camera_lidar ---" << std::endl << T_camera_lidar.matrix();
  guik::LightViewer::instance()->append_text(sst.str());

  std::cout << "T_camera_lidar" << T_camera_lidar.matrix() << std::endl;

  return Eigen::Isometry3d(T_camera_lidar.matrix());
}

}  // namespace vlcal
