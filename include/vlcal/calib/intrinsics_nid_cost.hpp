#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vlcal/common/visual_lidar_data.hpp> // VisualLiDARData 中包含 FrameCPU
// 假设 FrameCPU 定义在 vlcal/common/frame_cpu.hpp 中，与 Frame 类似
#include <vlcal/common/frame_cpu.hpp>

namespace vlcal {

/// 单帧内参 NID 代价函数
struct IntrinsicsNIDCost {
    IntrinsicsNIDCost(const Eigen::Isometry3d& T_lidar_cam,
                      const cv::Mat& normalized_img,
                      const FrameCPU::Ptr& points,
                      int bins)
      : T_lidar_camera(T_lidar_cam),
        image_normalized(normalized_img),
        lidar_frame(points),
        nbins(bins) {}

    template <typename T>
    bool operator()(const T* intrinsics, T* residual) const;

    Eigen::Isometry3d T_lidar_camera;
    cv::Mat image_normalized;
    FrameCPU::Ptr lidar_frame;
    int nbins;
};

} // namespace vlcal
