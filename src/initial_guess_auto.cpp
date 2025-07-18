// file: src/initial_guess_auto.cpp

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <ceres/ceres.h>
#include <ceres/loss_function.h>           // for CauchyLoss
#include <sophus/se3.hpp>                  // for Sophus::SE3d
#include <sophus/ceres_manifold.hpp>

#include <vlcal/common/console_colors.hpp>
#include <camera/create_camera.hpp>
#include <vlcal/common/visual_lidar_data.hpp>

namespace vlcal {
//
// 重投影残差，只优化 fx, fy, cx, cy
//
struct ReprojIntrinsicsCost {
  ReprojIntrinsicsCost(
      const Eigen::Vector3d& Xl,
      const Eigen::Vector2d& uv,
      const Sophus::SE3d&   T_Camera_Lidar)
    : Xl_(Xl), uv_(uv), T_Camera_Lidar(T_Camera_Lidar) {}

  template <typename T>
  bool operator()(T const* const intr, T* residual) const {
    // 1) LiDAR 点变换到相机系：R * Xl + t
    Eigen::Matrix<T,3,1> Xc =
        T_Camera_Lidar.rotationMatrix().template cast<T>() * Xl_.template cast<T>()
      + T_Camera_Lidar.translation().template cast<T>();

    // 2) 针对 fx,fy,cx,cy 做针孔投影
    T fx = intr[0], fy = intr[1], cx = intr[2], cy = intr[3];
    T u_hat = fx * Xc.x() / Xc.z() + cx;
    T v_hat = fy * Xc.y() / Xc.z() + cy;

    // 3) 残差
    residual[0] = T(uv_.x()) - u_hat;
    residual[1] = T(uv_.y()) - v_hat;
    return true;
  }

  const Eigen::Vector3d Xl_;
  const Eigen::Vector2d uv_;
  const Sophus::SE3d   T_Camera_Lidar;
};

class InitialGuessAuto {
public:
  explicit InitialGuessAuto(const std::string& data_path)
    : data_path_(data_path) {
    // 1) 加载 calib.json
    std::ifstream ifs(data_path_ + "/calib.json");
    if (!ifs) {
      std::cerr << console::bold_red
                << "error: failed to open " << data_path_ << "/calib.json"
                << console::reset << std::endl;
      std::exit(1);
    }
    ifs >> config_;

    // 2) 构造相机模型
    const std::string cam_model = config_["camera"]["camera_model"].get<std::string>();
    const auto intr_init        = config_["camera"]["intrinsics"].get<std::vector<double>>();
    const auto dist_coeffs      = config_["camera"]["distortion_coeffs"].get<std::vector<double>>();
    proj_ = camera::create_camera(cam_model, intr_init, dist_coeffs);

    // 3) 读取 JSON 里的外参 init_T_lidar_camera_auto （Camera→LiDAR）
    auto v = config_["results"]["init_T_lidar_camera_auto"];
    Eigen::Quaterniond q(v[6], v[3], v[4], v[5]);
    Eigen::Vector3d t(v[0], v[1], v[2]);
    // 注意这里是 Camera→LiDAR
    T_Lidar_Camera = Sophus::SE3d(q, t);

    // 4) pick_offsets 与原始实现一致
    constexpr int pw = 1;
    for (int i = -pw; i <= pw; ++i) {
      for (int j = -pw; j <= pw; ++j) {
        if (i != 0 || j != 0) pick_offsets_.emplace_back(i, j);
      }
    }
    std::sort(pick_offsets_.begin(), pick_offsets_.end(),
              [](auto const& a, auto const& b){ return a.squaredNorm() < b.squaredNorm(); });

    // 5) 读取所有 bag 的 2D–3D 对，并收集到 corrs_
    auto bags = config_["meta"]["bag_names"].get<std::vector<std::string>>();
    for (auto const& bag : bags) {
      auto data = std::make_shared<VisualLiDARData>(data_path_, bag);
      auto tmp  = read_correspondences(data_path_, bag, data->points);
      for (auto& p : tmp) {
        // p.first: (u,v), p.second: (x,y,z,1)
        corrs_.emplace_back(p.second.head<3>(), p.first);
      }
    }
  }

  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector4d>>
  read_correspondences(const std::string& data_path,
                       const std::string& bag_name,
                       const Frame::ConstPtr& points) {
    cv::Mat idx8 = cv::imread(data_path + "/" + bag_name + "_lidar_indices.png", -1);
    cv::Mat idx32(idx8.rows, idx8.cols, CV_32SC1,
                  reinterpret_cast<int*>(idx8.data));

    std::ifstream mifs(data_path + "/" + bag_name + "_matches.json");
    if (!mifs) {
      std::cerr << console::bold_red
                << "error: failed to open " << bag_name << "_matches.json"
                << console::reset << std::endl;
      std::exit(1);
    }
    nlohmann::json mr; mifs >> mr;

    auto kpts0   = mr["kpts0"].get<std::vector<int>>();
    auto kpts1   = mr["kpts1"].get<std::vector<int>>();
    auto matches = mr["matches"].get<std::vector<int>>();

    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector4d>> corrs;
    for (size_t i = 0; i < matches.size(); ++i) {
      int m = matches[i];
      if (m < 0) continue;
      Eigen::Vector2i p0{kpts0[2*i],   kpts0[2*i+1]};
      Eigen::Vector2i p1{kpts1[2*m],   kpts1[2*m+1]};

      int idx = idx32.at<int>(p1.y(), p1.x());
      if (idx < 0) {
        // 尝试邻域
        for (auto const& off : pick_offsets_) {
          int ny = p1.y() + off.y(), nx = p1.x() + off.x();
          if (ny < 0 || ny >= idx32.rows || nx < 0 || nx >= idx32.cols) continue;
          idx = idx32.at<int>(ny, nx);
          if (idx >= 0) break;
        }
        if (idx < 0) continue;
      }
      corrs.emplace_back(
        Eigen::Vector2d(p0.x(), p0.y()),
        points->points[idx]
      );
    }
    return corrs;
  }

  void estimate_and_save() {
    // 1) 读初值 intrinsics
    auto ini = config_["camera"]["intrinsics"].get<std::vector<double>>();
    double intr[4] = { ini[0], ini[1], ini[2], ini[3] };

    Sophus::SE3d T_Camera_Lidar = T_Lidar_Camera.inverse();

    // 3) 构建 Ceres 问题
    ceres::Problem problem;
    problem.AddParameterBlock(intr, 4);

    for (auto const& [Xl, uv] : corrs_) {
      auto* cost = new ReprojIntrinsicsCost(Xl, uv, T_Camera_Lidar);
      auto* func = new ceres::AutoDiffCostFunction<ReprojIntrinsicsCost,2,4>(cost);
      problem.AddResidualBlock(func, new ceres::CauchyLoss(1.0), intr);
    }

    // 4) 求解
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // 5) 写回 JSON
    config_["camera"]["intrinsics"] = { intr[0], intr[1], intr[2], intr[3] };
    std::ofstream ofs(data_path_ + "/calib.json");
    ofs << config_.dump(2) << std::endl;
  }

private:
  const std::string data_path_;
  nlohmann::json config_;
  camera::GenericCameraBase::ConstPtr proj_;
  Sophus::SE3d   T_Lidar_Camera;
  std::vector<Eigen::Vector2i>                    pick_offsets_;
  std::vector<std::pair<Eigen::Vector3d,Eigen::Vector2d>> corrs_;
};

}  // namespace vlcal

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description desc("initial_guess_auto");
  desc.add_options()
    ("help",      "show help")
    ("data_path", po::value<std::string>()->required(), "dir with preprocessed data");
  po::positional_options_description p;
  p.add("data_path", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
              .options(desc)
              .positional(p)
              .run(),
            vm);
  if (vm.count("help")) { std::cout << desc << std::endl; return 0; }
  po::notify(vm);

  std::string dp = vm["data_path"].as<std::string>();
  vlcal::InitialGuessAuto ig(dp);
  ig.estimate_and_save();

  return 0;
}
