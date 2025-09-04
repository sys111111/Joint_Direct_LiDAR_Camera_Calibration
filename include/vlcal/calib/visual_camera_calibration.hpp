#pragma once

#include <camera/generic_camera_base.hpp>
#include <vlcal/common/visual_lidar_data.hpp>

namespace vlcal {

enum class RegistrationType { NID_BFGS };

struct VisualCameraCalibrationParams {
public:
  VisualCameraCalibrationParams() {
    max_outer_iterations = 10;
    max_inner_iterations = 256;

    delta_trans_thresh = 0.1;
    delta_rot_thresh = 0.5 * M_PI / 180.0;

    disable_z_buffer_culling = false;

    nid_bins = 16;

    registration_type = RegistrationType::NID_BFGS;
    nelder_mead_init_step = 1e-3;
    nelder_mead_convergence_criteria = 1e-8;
  }

  int max_outer_iterations;   
  int max_inner_iterations;   
  double delta_trans_thresh;  
  double delta_rot_thresh;    

  bool disable_z_buffer_culling;  
  int nid_bins;                   

  RegistrationType registration_type;
  double nelder_mead_init_step;
  double nelder_mead_convergence_criteria;

  std::function<void(const Eigen::Isometry3d& T_camera_lidar)> callback;

  std::function<void(const Eigen::Vector4d& intrinsics, const Eigen::VectorXd& distortion)> intrinsic_callback;
};

class VisualCameraCalibration {
public:
  VisualCameraCalibration(const camera::GenericCameraBase::ConstPtr& proj, 
                          const std::vector<VisualLiDARData::ConstPtr>& dataset, 
                          const VisualCameraCalibrationParams& params = VisualCameraCalibrationParams());

  std::tuple<Eigen::Vector4d, Eigen::VectorXd, Eigen::Isometry3d> calibrate(const Eigen::Isometry3d& init_T_camera_lidar);
  Eigen::Isometry3d calibrate_ex(const Eigen::Isometry3d& init_T_camera_lidar);

  std::tuple<Eigen::Vector4d, Eigen::VectorXd, Eigen::Isometry3d> estimate_intrinsic_bfgs(const Eigen::Isometry3d& init_T_camera_lidar);
  std::tuple<Eigen::Vector4d, Eigen::VectorXd, Eigen::Isometry3d> estimate_intrinsic_LM(const Eigen::Isometry3d& init_T_camera_lidar);

  Eigen::Isometry3d estimate_pose_nelder_mead(const Eigen::Isometry3d& init_T_camera_lidar);
  Eigen::Isometry3d estimate_pose_bfgs(const Eigen::Isometry3d& init_T_camera_lidar);

  struct OptimizationData {
    int outer_iteration;  
    double cost;
    double fx, fy, cx, cy;
    double k1, k2, p1, p2, k3;
  };

private:
  const VisualCameraCalibrationParams params;

  std::vector<double> toStdVector(const Eigen::VectorXd& vec) const {
      return std::vector<double>(vec.data(), vec.data() + vec.size());
  }

  camera::GenericCameraBase::Ptr proj_update;  

  const camera::GenericCameraBase::ConstPtr proj;  
  const std::vector<VisualLiDARData::ConstPtr> dataset;  
  std::vector<OptimizationData> optimization_data;  
  double current_cost = 0.0; 

};

}  // namespace vlcal
