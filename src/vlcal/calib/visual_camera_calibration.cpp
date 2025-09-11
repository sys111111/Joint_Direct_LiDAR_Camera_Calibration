#include <iostream>
#include <typeinfo>
#include <sstream>
#include <iomanip>

#include <fstream>
#include <chrono>

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

#include <vlcal/calib/view_culling.hpp>
#include <vlcal/calib/cost_calculator_nid.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <camera/create_camera.hpp>
#include <vlcal/costs/nid_cost.hpp>
#include <vlcal/common/console_colors.hpp>
#include <vlcal/common/visual_lidar_data.hpp>
#include <vlcal/common/points_color_updater.hpp>
#include <vlcal/common/visual_lidar_visualizer.hpp>

#include <ceres/autodiff_cost_function.h> 

namespace vlcal {

VisualCameraCalibration::VisualCameraCalibration(
  const camera::GenericCameraBase::ConstPtr& proj,
  const std::vector<VisualLiDARData::ConstPtr>& dataset,
  const VisualCameraCalibrationParams& params)
: params(params),
  proj(proj),
  dataset(dataset) {
    Eigen::VectorXd intrinsics = proj->get_intrinsics();
    Eigen::VectorXd distortion = proj->get_distortion_coeffs();
    
    std::vector<double> intrinsics_vec = toStdVector(intrinsics);
    std::vector<double> distortion_vec = toStdVector(distortion);
    
    proj_update = std::const_pointer_cast<camera::GenericCameraBase>(
        camera::create_camera("plumb_bob", intrinsics_vec, distortion_vec));
  }

std::tuple<Eigen::Vector4d, Eigen::VectorXd, Eigen::Isometry3d> VisualCameraCalibration::calibrate(
    const Eigen::Isometry3d& init_T_camera_lidar){

    std::cout << init_T_camera_lidar.matrix() << std::endl;
    Eigen::Isometry3d T_camera_lidar = init_T_camera_lidar;
    Eigen::Vector4d final_intrinsics;
    Eigen::Vector4d prev_intrinsics = Eigen::Vector4d::Zero();

    Eigen::VectorXd new_distortion(5);
    bool converged = false;

    for (int i = 0; i < params.max_outer_iterations; i++) {
        std::cout << "Iteration " << i + 1 << " of " << params.max_outer_iterations << std::endl;

        if (i < params.max_outer_iterations / 2 + 1) {
            std::cout << "Using BFGS optimization" << std::endl;
            auto [intrinsics, distortion, new_T_camera_lidar] = estimate_intrinsic_bfgs(init_T_camera_lidar);
            final_intrinsics = intrinsics;
            new_distortion = distortion;
        } else {
            std::cout << "Using LM optimization" << std::endl;
            auto [intrinsics, distortion, new_T_camera_lidar] = estimate_intrinsic_LM(init_T_camera_lidar);
            final_intrinsics = intrinsics;
            new_distortion = distortion;
        }

        OptimizationData data;
        data.outer_iteration = i + 1;
        data.cost = current_cost;

        Eigen::VectorXd current_intrinsics = proj_update->get_intrinsics();
        new_distortion = proj_update->get_distortion_coeffs();

        std::cout << "\n=== Optimization Results ===" << std::endl;
        std::cout << "\nFinal parameters after LM optimization:" << std::endl;
        std::cout << "fx: " << final_intrinsics[0] << std::endl;
        std::cout << "fy: " << final_intrinsics[1] << std::endl;
        std::cout << "cx: " << final_intrinsics[2] << std::endl;
        std::cout << "cy: " << final_intrinsics[3] << std::endl;
        std::cout << "k1: " << new_distortion[0] << std::endl;
        std::cout << "k2: " << new_distortion[1] << std::endl;
        std::cout << "p1: " << new_distortion[2] << std::endl;
        std::cout << "p2: " << new_distortion[3] << std::endl;
        std::cout << "k3: " << new_distortion[4] << std::endl;

        std::stringstream sst;
        sst << "--- Camera Intrinsics ---" << std::endl;
        sst << "fx: " << final_intrinsics[0] << std::endl;
        sst << "fy: " << final_intrinsics[1] << std::endl;
        sst << "cx: " << final_intrinsics[2] << std::endl;
        sst << "cy: " << final_intrinsics[3] << std::endl;
        sst << "k1: " << new_distortion[0] << std::endl;
        sst << "k2: " << new_distortion[1] << std::endl;
        sst << "p1: " << new_distortion[2] << std::endl;
        sst << "p2: " << new_distortion[3] << std::endl;
        sst << "k3: " << new_distortion[4] << std::endl;

        data.fx = current_intrinsics[0];
        data.fy = current_intrinsics[1];
        data.cx = current_intrinsics[2];
        data.cy = current_intrinsics[3];
        data.k1 = new_distortion[0];
        data.k2 = new_distortion[1];
        data.p1 = new_distortion[2];
        data.p2 = new_distortion[3];
        data.k3 = new_distortion[4];

        optimization_data.push_back(data);

        double intrinsics_delta = (final_intrinsics - prev_intrinsics).norm();
        prev_intrinsics = final_intrinsics;

        Eigen::VectorXd new_intrinsics(4);
        new_intrinsics << final_intrinsics;

        std::vector<double> intrinsics_vec = toStdVector(new_intrinsics);
        std::vector<double> distortion_vec = toStdVector(new_distortion);

        proj_update = std::const_pointer_cast<camera::GenericCameraBase>(
            camera::create_camera("plumb_bob", intrinsics_vec, distortion_vec));

        if (intrinsics_delta < 1e-9) {
            std::cout << "Optimization converged after " << i + 1 << " iterations." << std::endl;
            converged = true;
            break;
        }
    }

    if (!converged) {
        std::cout << "Optimization reached maximum iterations without convergence." << std::endl;
    }

    return std::make_tuple(final_intrinsics, new_distortion, T_camera_lidar);
}

class OptimizationCallback : public ceres::IterationCallback {
    public:
        explicit OptimizationCallback(double* parameters) 
            : parameters_(parameters) {}
        
        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
            return ceres::SOLVER_CONTINUE;
        }
    
    private:
        double* parameters_;
};

class MultiNIDCost {
public:
    MultiNIDCost(const CameraIntrinsics<double>& init_intrinsics, 
                 const Eigen::Isometry3d& T_camera_lidar)
        : init_intrinsics(init_intrinsics), T_camera_lidar(T_camera_lidar) {}

    void add(const std::shared_ptr<NIDCost>& cost) {
        costs.push_back(cost);
    }

    template <typename T>
    bool operator()(const T* params, T* residual) const 
    {
        CameraIntrinsics<T> curr_intrinsics;
        curr_intrinsics.f_x = T(params[0]);
        curr_intrinsics.f_y = T(params[1]);
        curr_intrinsics.c_x = T(params[2]);
        curr_intrinsics.c_y = T(params[3]);
        curr_intrinsics.k1 = T(params[4]);
        curr_intrinsics.k2 = T(params[5]);
        curr_intrinsics.p1 = T(params[6]);
        curr_intrinsics.p2 = T(params[7]);
        curr_intrinsics.k3 = T(params[8]);

        std::vector<bool> results(costs.size());
        std::vector<T> residuals(costs.size());

        #pragma omp parallel for
        for (int i = 0; i < costs.size(); i++) 
        {
            results[i] = (*costs[i])(curr_intrinsics, T_camera_lidar, &residuals[i]);
        }

        for (int i = 1; i < costs.size(); i++) 
        {
        residuals[0] += residuals[i];
        }

        *residual = residuals[0];

        return std::count(results.begin(), results.end(), false) == 0;
    }

private:
    CameraIntrinsics<double> init_intrinsics;
    Eigen::Isometry3d T_camera_lidar;
    std::vector<std::shared_ptr<NIDCost>> costs;
};

std::tuple<Eigen::Vector4d, Eigen::VectorXd, Eigen::Isometry3d> 
VisualCameraCalibration::estimate_intrinsic_bfgs(const Eigen::Isometry3d& init_T_camera_lidar) {
    std::cout << "=== BFGS Optimization ===" << std::endl;
    ViewCullingParams view_culling_params;
    view_culling_params.enable_depth_buffer_culling = !this->params.disable_z_buffer_culling;
    ViewCulling view_culling(proj_update,  
        {this->dataset.front()->image.cols, this->dataset.front()->image.rows},
        view_culling_params);
    
    Eigen::VectorXd intrinsics_out = proj_update->get_intrinsics(); 
    Eigen::VectorXd distortion_out = proj_update->get_distortion_coeffs();
    
    double* parameters = new double[9];
    std::copy(intrinsics_out.data(), intrinsics_out.data() + 4, parameters);
    std::copy(distortion_out.data(), distortion_out.data() + 5, parameters + 4);

    std::cout << "\nInitial parameters:" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "fx: " << parameters[0] << std::endl;
    std::cout << "fy: " << parameters[1] << std::endl;
    std::cout << "cx: " << parameters[2] << std::endl;
    std::cout << "cy: " << parameters[3] << std::endl;
    std::cout << "k1: " << parameters[4] << std::endl;
    std::cout << "k2: " << parameters[5] << std::endl;
    std::cout << "p1: " << parameters[6] << std::endl;
    std::cout << "p2: " << parameters[7] << std::endl;
    std::cout << "k3: " << parameters[8] << std::endl;
    
    CameraIntrinsics<double> init_intrinsics;
    init_intrinsics.f_x = intrinsics_out(0);
    init_intrinsics.f_y = intrinsics_out(1);
    init_intrinsics.c_x = intrinsics_out(2);
    init_intrinsics.c_y = intrinsics_out(3);
    init_intrinsics.k1 = distortion_out(0);
    init_intrinsics.k2 = distortion_out(1);
    init_intrinsics.p1 = distortion_out(2);
    init_intrinsics.p2 = distortion_out(3);
    init_intrinsics.k3 = distortion_out(4);

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;
    ceres::Problem problem(problem_options);

    problem.AddParameterBlock(parameters, 9);

    std::vector<std::shared_ptr<NIDCost>> nid_costs;

    for (int i = 0; i < dataset.size(); i++) {
        auto culled_points = view_culling.cull(dataset[i]->points, init_T_camera_lidar);
        cv::Mat normalized_image;
        dataset[i]->image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0);
        // auto nid_cost = std::make_shared<NIDCost>(normalized_image, culled_points, params.nid_bins);
        std::shared_ptr<NIDCost> nid_cost(new NIDCost(normalized_image, culled_points, params.nid_bins));
        nid_costs.emplace_back(nid_cost);
    }

    auto sum_nid = std::make_unique<MultiNIDCost>(init_intrinsics, init_T_camera_lidar);

    for (const auto& nid_cost : nid_costs) {
        sum_nid->add(nid_cost);
    }

    std::unique_ptr<ceres::FirstOrderFunction> first_order_function = std::make_unique<ceres::AutoDiffFirstOrderFunction<MultiNIDCost, 9>>(std::move(sum_nid));

    ceres::GradientProblemSolver::Options options;
    options.update_state_every_iteration = true;
    options.minimizer_progress_to_stdout = true;
    options.line_search_direction_type = ceres::BFGS;

    ceres::GradientProblem gradient_problem(std::move(first_order_function));
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, gradient_problem, parameters, &summary);

    current_cost = summary.final_cost;

    std::cout << "\n=== Optimization Results ===" << std::endl;
    std::cout << "\nFinal parameters:" << std::endl;
    std::cout << "fx: " << parameters[0] << std::endl;
    std::cout << "fy: " << parameters[1] << std::endl;
    std::cout << "cx: " << parameters[2] << std::endl;
    std::cout << "cy: " << parameters[3] << std::endl;
    std::cout << "k1: " << parameters[4] << std::endl;
    std::cout << "k2: " << parameters[5] << std::endl;
    std::cout << "p1: " << parameters[6] << std::endl;
    std::cout << "p2: " << parameters[7] << std::endl;
    std::cout << "k3: " << parameters[8] << std::endl;

    Eigen::Vector4d final_intrinsics;
    final_intrinsics << parameters[0], parameters[1], parameters[2], parameters[3];

    Eigen::VectorXd new_intrinsics(4);
    new_intrinsics << parameters[0], parameters[1], parameters[2], parameters[3];
    Eigen::VectorXd new_distortion(5);
    new_distortion << parameters[4], parameters[5], parameters[6], parameters[7], parameters[8];

    std::cout << "summary_cost: " << summary.final_cost << std::endl;
    
    std::vector<double> intrinsics_vec = toStdVector(new_intrinsics);
    std::vector<double> distortion_vec = toStdVector(new_distortion);
    
    proj_update = std::const_pointer_cast<camera::GenericCameraBase>(
        camera::create_camera("plumb_bob", intrinsics_vec, distortion_vec));

    delete[] parameters;

    return std::make_tuple(final_intrinsics, new_distortion, init_T_camera_lidar);
}

std::tuple<Eigen::Vector4d, Eigen::VectorXd, Eigen::Isometry3d> 
VisualCameraCalibration::estimate_intrinsic_LM(const Eigen::Isometry3d& init_T_camera_lidar) {
    std::cout << "=== LM Optimization ===" << std::endl;
    Eigen::VectorXd intrinsics_out = proj_update->get_intrinsics();
    Eigen::VectorXd distortion_out = proj_update->get_distortion_coeffs();

    double parameters[9];
    std::copy(intrinsics_out.data(), intrinsics_out.data() + 4, parameters);
    std::copy(distortion_out.data(), distortion_out.data() + 5, parameters + 4);

    CameraIntrinsics<double> init_intrinsics;
    init_intrinsics.f_x = intrinsics_out(0);
    init_intrinsics.f_y = intrinsics_out(1);
    init_intrinsics.c_x = intrinsics_out(2);
    init_intrinsics.c_y = intrinsics_out(3);
    init_intrinsics.k1 = distortion_out(0);
    init_intrinsics.k2 = distortion_out(1);
    init_intrinsics.p1 = distortion_out(2);
    init_intrinsics.p2 = distortion_out(3);
    init_intrinsics.k3 = distortion_out(4);

    ViewCullingParams view_culling_params;
    view_culling_params.enable_depth_buffer_culling = !this->params.disable_z_buffer_culling;
    ViewCulling view_culling(proj_update,
        {this->dataset.front()->image.cols, this->dataset.front()->image.rows},
        view_culling_params);

    std::vector<std::shared_ptr<NIDCost>> nid_costs;
    for (const auto& data : dataset) {
        cv::Mat normalized_image;
        data->image.convertTo(normalized_image, CV_64FC1, 1.0 / 255.0);
        auto culled_points = view_culling.cull(data->points, init_T_camera_lidar);
        auto nid_cost = std::make_shared<NIDCost>(normalized_image, culled_points, params.nid_bins);
        nid_costs.emplace_back(nid_cost);
    }

    ceres::Problem problem;
    problem.AddParameterBlock(parameters, 9);

    auto multi_nid_cost = new MultiNIDCost(init_intrinsics, init_T_camera_lidar);
    for (const auto& cost : nid_costs) {
        multi_nid_cost->add(cost);
    }

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<MultiNIDCost, 1, 9>(multi_nid_cost),
        nullptr,
        parameters);

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.max_num_iterations = 100;
    // options.function_tolerance = 1e-12;
    // options.gradient_tolerance = 1e-12;
    // options.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector4d final_intrinsics;
    final_intrinsics << parameters[0], parameters[1], parameters[2], parameters[3];

    Eigen::VectorXd new_intrinsics(4);
    new_intrinsics << parameters[0], parameters[1], parameters[2], parameters[3];
    Eigen::VectorXd new_distortion(5);
    new_distortion << parameters[4], parameters[5], parameters[6], parameters[7], parameters[8];

    std::vector<double> intrinsics_vec = toStdVector(new_intrinsics);
    std::vector<double> distortion_vec = toStdVector(new_distortion);

    proj_update = std::const_pointer_cast<camera::GenericCameraBase>(
        camera::create_camera("plumb_bob", intrinsics_vec, distortion_vec));

    return std::make_tuple(final_intrinsics, new_distortion, init_T_camera_lidar);
}

}  // namespace vlcal