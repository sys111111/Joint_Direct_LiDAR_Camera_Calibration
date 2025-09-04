#include <atomic>
#include <thread>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <boost/program_options.hpp>

#include <nlohmann/json.hpp>

#include <gtsam/geometry/Pose3.h>

#include <dfo/nelder_mead.hpp>
#include <dfo/directional_direct_search.hpp>

#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <camera/generic_camera_base.hpp>
#include <camera/create_camera.hpp>
#include <memory>
#include <camera/pinhole.hpp>

#include <vlcal/costs/nid_cost.hpp>
#include <vlcal/common/console_colors.hpp>
#include <vlcal/common/visual_lidar_data.hpp>
#include <vlcal/common/points_color_updater.hpp>
#include <vlcal/common/visual_lidar_visualizer.hpp>
#include <vlcal/calib/visual_camera_calibration.hpp>

namespace vlcal {

class VisualLiDARCalibration {
public:
  VisualLiDARCalibration(const std::string& data_path, const boost::program_options::variables_map& vm) : data_path(data_path) {
    std::ifstream ifs(data_path + "/calib.json");
    if (!ifs) {
      std::cerr << vlcal::console::bold_red << "error: failed to open " << data_path << "/calib.json" << vlcal::console::reset << std::endl;
      abort();
    }

    ifs >> config;

    std::vector<std::string> bag_names = config["meta"]["bag_names"];
    for (const auto& bag_name : bag_names) {
      dataset.emplace_back(std::make_shared<VisualLiDARData>(data_path, bag_name));
    }

    if (vm.count("first_n_bags")) {
      const int first_n_bags = vm["first_n_bags"].as<int>();
      dataset.erase(dataset.begin() + first_n_bags, dataset.end());
      std::cout << "use only the first " << first_n_bags << " bags" << std::endl;
    }
  }

  void calibrate(const boost::program_options::variables_map& vm) {
    const std::string camera_model = config["camera"]["camera_model"];
    const std::vector<double> intrinsics = config["camera"]["intrinsics"];
    const std::vector<double> distortion_coeffs = config["camera"]["distortion_coeffs"];

    proj = camera::create_camera(camera_model, intrinsics, distortion_coeffs);
    
    std::vector<double> init_values;
    std::cout << "use CAD initial values" << std::endl;
    const std::vector<double> values = config["results"]["init_T_lidar_camera"];
    init_values.assign(values.begin(), values.end());

    if (init_values.empty()) {
      std::cerr << vlcal::console::bold_red 
                << "error: initial guess of T_lidar_camera must be computed before calibration!!" 
                << vlcal::console::reset << std::endl;
      abort();
    }

    Eigen::Isometry3d init_T_lidar_camera = Eigen::Isometry3d::Identity();
    init_T_lidar_camera.translation() << init_values[0], init_values[1], init_values[2];
    init_T_lidar_camera.linear() = Eigen::Quaterniond(
        init_values[6], init_values[3], init_values[4], init_values[5]
    ).normalized().toRotationMatrix();

    const Eigen::Isometry3d init_T_camera_lidar = init_T_lidar_camera.inverse();

    auto viewer = guik::LightViewer::instance(Eigen::Vector2i(-1, -1), vm.count("background"));
    viewer->set_draw_xy_grid(false);
    viewer->use_arcball_camera_control();

    viewer->invoke([] {
      ImGui::SetNextWindowPos({60, 1300}, ImGuiCond_Once);
      ImGui::Begin("texts");
      ImGui::End();
      ImGui::SetNextWindowPos({1200, 60}, ImGuiCond_Once);
      ImGui::Begin("visualizer");
      ImGui::End();
      ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
      ImGui::Begin("images");
      ImGui::End();
    });

    VisualLiDARVisualizer vis(proj, dataset, false);
    vis.set_T_camera_lidar(init_T_camera_lidar);

    VisualCameraCalibrationParams params;
    params.disable_z_buffer_culling = vm.count("disable_culling");
    params.nid_bins = vm["nid_bins"].as<int>();
    params.nelder_mead_init_step = vm["nelder_mead_init_step"].as<double>();
    params.nelder_mead_convergence_criteria = vm["nelder_mead_convergence_criteria"].as<double>();
    const std::string registration_type = vm["registration_type"].as<std::string>();

    params.callback = [&](const Eigen::Isometry3d& T_camera_lidar) {
      vis.set_T_camera_lidar(T_camera_lidar);
    };

    params.intrinsic_callback = [&](const Eigen::Vector4d& intr,
                                    const Eigen::VectorXd& dist) {
      std::vector<double> ivec{intr[0], intr[1], intr[2], intr[3]};
      std::vector<double> dvec{dist[0], dist[1], dist[2], dist[3], dist[4]};
      proj = camera::create_camera(camera_model, ivec, dvec);
      vis.set_camera(proj);
    };
    
    VisualCameraCalibration calib(proj, dataset, params);
    
    std::atomic_bool done{false};
    Eigen::Vector4d  final_intrinsics;
    Eigen::VectorXd  final_distortion(5);
    Eigen::Isometry3d final_T_camera_lidar = init_T_camera_lidar;

    std::thread worker([&](){
      std::tie(final_intrinsics, final_distortion, final_T_camera_lidar)
          = calib.calibrate(init_T_camera_lidar);
      done = true;
    });

    while (!done) {
      vis.spin_once();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    worker.join();

    std::stringstream sst;
    sst << "--- Camera Intrinsics ---" << std::endl;
    sst << "fx: " << final_intrinsics[0] << std::endl;
    sst << "fy: " << final_intrinsics[1] << std::endl;
    sst << "cx: " << final_intrinsics[2] << std::endl;
    sst << "cy: " << final_intrinsics[3] << std::endl;
    
    sst << "k1: " << final_distortion[0] << std::endl;
    sst << "k2: " << final_distortion[1] << std::endl;
    sst << "p1: " << final_distortion[2] << std::endl;
    sst << "p2: " << final_distortion[3] << std::endl;
    sst << "k3: " << final_distortion[4] << std::endl;

    config["camera"]["distortion_coeffs_final_results"] = nlohmann::json::array({
        final_distortion[0],
        final_distortion[1],
        final_distortion[2],
        final_distortion[3],
        final_distortion[4]
    });

    config["camera"]["intrinsics_final_results"] = nlohmann::json::array({
        final_intrinsics[0],
        final_intrinsics[1],
        final_intrinsics[2],
        final_intrinsics[3]
    });

    std::ofstream ofs(data_path + "/calib.json");
    if (ofs) {
        ofs << config.dump(4); 
        std::cout << "Successfully saved calibration results to calib.json" << std::endl;
    } else {
        std::cerr << "Failed to save calibration results" << std::endl;
    }

    viewer->append_text(sst.str());
    viewer->spin_once();

    if (!vm.count("auto_quit")) {
      viewer->spin();
    }
}
  
private:
  const std::string data_path;
  nlohmann::json config;// shen pipeline


  camera::GenericCameraBase::ConstPtr proj;
  std::vector<VisualLiDARData::ConstPtr> dataset;
};

}  // namespace vlcal

int main(int argc, char** argv) {
  using namespace boost::program_options;
  options_description description("calibrate");

  // clang-format off
  description.add_options()
    ("help", "produce help message")
    ("data_path", value<std::string>(), "directory that contains preprocessed data")
    ("first_n_bags", value<int>(), "use only the first N bags (just for evaluation)")
    ("disable_culling", "disable depth buffer-based hidden points removal")
    ("nid_bins", value<int>()->default_value(16), "Number of histogram bins for NID")
    ("registration_type", value<std::string>()->default_value("nid_bfgs"), "nid_bfgs or nid_nelder_mead")
    ("nelder_mead_init_step", value<double>()->default_value(1e-3), "Nelder-mead initial step size")
    ("nelder_mead_convergence_criteria", value<double>()->default_value(1e-8), "Nelder-mead convergence criteria")
    ("auto_quit", "automatically quit after calibration")
    ("background", "hide viewer and run calibration in background")
  ;
  // clang-format on

  positional_options_description p;
  p.add("data_path", 1);

  variables_map vm;
  store(command_line_parser(argc, argv).options(description).positional(p).run(), vm);
  notify(vm);

  if (vm.count("help") || !vm.count("data_path")) {
    std::cout << description << std::endl;
    return 0;
  }

  const std::string data_path = vm["data_path"].as<std::string>();

  vlcal::VisualLiDARCalibration calib(data_path, vm);

  calib.calibrate(vm);

  return 0;
}