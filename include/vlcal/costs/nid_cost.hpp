#pragma once

#include <algorithm>
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <vlcal/common/frame.hpp>
#include <ceres/ceres.h>

namespace vlcal {

template <typename T>
inline double get_real(const T& x) {
    return x.a;
}

template <>
inline double get_real(const double& x) {
    return x;
}

template <typename T>
struct CameraIntrinsics {
    T f_x, f_y;  // 焦距
    T c_x, c_y;  // 主点坐标
    T k1, k2, p1, p2, k3;  // 畸变系数

    CameraIntrinsics() 
        : f_x(0), f_y(0), c_x(0), c_y(0), k1(0), k2(0), p1(0), p2(0), k3(0) {}

    CameraIntrinsics(T f_x, T f_y, T c_x, T c_y, T k1, T k2, T p1, T p2, T k3)
        : f_x(f_x), f_y(f_y), c_x(c_x), c_y(c_y), k1(k1), k2(k2), p1(p1), p2(p2), k3(k3) {}
};

class NIDCost {
public:
    NIDCost(const cv::Mat& normalized_image, 
            const Frame::ConstPtr& points, 
            const int bins = 16)
        : normalized_image(normalized_image.clone()),
          points(points),
          bins(bins) {
        spline_coeffs.row(0) << 1.0, -3.0, 3.0, -1.0;
        spline_coeffs.row(1) << 4.0, 0.0, -6.0, 3.0;
        spline_coeffs.row(2) << 1.0, 3.0, 3.0, -3.0;
        spline_coeffs.row(3) << 0.0, 0.0, 0.0, 1.0;
        spline_coeffs /= 6.0;
    }

    template <typename T>
    bool operator()(const CameraIntrinsics<T>& intrinsics, 
                   const Eigen::Isometry3d& T_camera_lidar, 
                   T* residual) const {
        Eigen::Matrix<T, 3, 3> R = T_camera_lidar.rotation().cast<T>();
        Eigen::Matrix<T, 3, 1> t = T_camera_lidar.translation().cast<T>();

        const T fx = intrinsics.f_x;
        const T fy = intrinsics.f_y;
        const T cx = intrinsics.c_x;
        const T cy = intrinsics.c_y;
        const T k1 = intrinsics.k1;
        const T k2 = intrinsics.k2;
        const T p1 = intrinsics.p1;
        const T p2 = intrinsics.p2;
        const T k3 = intrinsics.k3;

        // 初始化直方图
        Eigen::Matrix<T, -1, -1> hist = Eigen::Matrix<T, -1, -1>::Zero(bins, bins);
        Eigen::Matrix<T, -1, 1> hist_image = Eigen::Matrix<T, -1, 1>::Zero(bins);
        Eigen::VectorXd hist_points = Eigen::VectorXd::Zero(bins);

        int num_outliers = 0;
        int num_valid_points = 0;

        for (int i = 0; i < points->size(); i++) {
            Eigen::Matrix<T, 3, 1> pt;
            pt[0] = T(points->points[i].x());
            pt[1] = T(points->points[i].y());
            pt[2] = T(points->points[i].z());
            
            // 转换到相机坐标系
            Eigen::Matrix<T, 3, 1> pt_camera = R * pt + t;
            
            // 检查深度
            if (pt_camera[2] <= T(0)) {
                num_outliers++;
                continue;
            }

            const double intensity = points->intensities[i];
            const int bin_points = std::max<int>(0, std::min<int>(bins - 1, intensity * bins));

            // 投影点计算
            const Eigen::Matrix<T, 2, 1> pt_normalized(
                pt_camera[0] / pt_camera[2],
                pt_camera[1] / pt_camera[2]
            );

            // //shen 这里修改畸变参数
            // // // 畸变计算
            // const T x2 = pt_normalized.x() * pt_normalized.x();
            // const T y2 = pt_normalized.y() * pt_normalized.y();
            // const T xy = pt_normalized.x() * pt_normalized.y();
            // const T r2 = x2 + y2;
            // const T r4 = r2 * r2;
            // const T r6 = r2 * r4;

            // // 径向畸变
            // const T r_coeff = T(1) + k1 * r2 + k2 * r4 + k3 * r6;

            // // 切向畸变
            // const T t_coeff1 = T(2) * xy;
            // const T t_coeff2 = r2 + T(2) * x2;
            // const T t_coeff3 = r2 + T(2) * y2;

            // const T xd = r_coeff * pt_normalized.x() + p1 * t_coeff1 + p2 * t_coeff2;
            // const T yd = r_coeff * pt_normalized.y() + p1 * t_coeff3 + p2 * t_coeff1;
            
            const T xd = pt_camera[0] / pt_camera[2];
            const T yd = pt_camera[1] / pt_camera[2];
            
            // 投影到像素平面
            Eigen::Matrix<T, 2, 1> projected;
            projected[0] = fx * xd + cx;
            projected[1] = fy * yd + cy;

            // 计算整数坐标
            const Eigen::Vector2i knot_i(
                std::floor(get_real(projected[0])), 
                std::floor(get_real(projected[1]))
            );

            // 边界检查
            if ((knot_i.array() < Eigen::Array2i(0, 0)).any() || 
                (knot_i.array() >= Eigen::Array2i(normalized_image.cols, normalized_image.rows)).any()) {
                num_outliers++;
                continue;
            }

            num_valid_points++;
            hist_points[bin_points]++;

            // 计算样条系数
            const Eigen::Matrix<T, 2, 1> s = projected - knot_i.cast<T>();
            
            Eigen::Matrix<T, 4, 2> se;
            se.row(0).setOnes();
            se.row(1) = s.transpose();
            se.row(2) = s.array().square().transpose();
            se.row(3) = (s.array().square() * s.array()).transpose();

            const Eigen::Matrix<T, 4, 2> beta = spline_coeffs * se;

            // 计算样条点
            Eigen::Array4i knots_x(knot_i.x() - 1, knot_i.x(), knot_i.x() + 1, knot_i.x() + 2);
            Eigen::Array4i knots_y(knot_i.y() - 1, knot_i.y(), knot_i.y() + 1, knot_i.y() + 2);
            
            knots_x = knots_x.max(0).min(normalized_image.cols - 1);
            knots_y = knots_y.max(0).min(normalized_image.rows - 1);

            // 更新直方图
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    const T w = beta(i, 0) * beta(j, 1);
                    const double pix = normalized_image.at<double>(knots_y[j], knots_x[i]);
                    const int bin_image = std::min<int>(pix * bins, bins - 1);
                    hist(bin_image, bin_points) += w;
                    hist_image[bin_image] += w;
                }
            }
        }

        if (num_valid_points < 100) {
            std::cout << "Warning: Too few valid points (" << num_valid_points << ")" << std::endl;
            return false;
        }

        // 归一化直方图
        const double sum = hist_points.sum();

        if (sum < 1e-8) {
            std::cout << "Warning: Empty histogram" << std::endl;
            return false;
        }

        hist_image = hist_image / T(sum);
        hist_points = hist_points / sum;
        hist = hist / T(sum);

        // 计算互信息和NID
        const T epsilon = T(1e-10);
        const T H_image = -(hist_image.array().max(epsilon) * 
                           (hist_image.array().max(epsilon)).log()).sum();
        const double H_points = -(hist_points.array() * 
                                (hist_points.array() + 1e-6).log()).sum();
        const T H_image_points = -(hist.array() * (hist.array() + epsilon).log()).sum();
        
        const T MI = H_image + T(H_points) - H_image_points;
        const T NID = (H_image_points - MI) / H_image_points;

        // residual[0] = NID*T(100);
        residual[0] = NID;

        std::cout << "NID: " << get_real(NID) << std::endl;

        // 调试输出if (ceres::isnan(get_real(NID))) {
        // if (ceres::isnan(get_real(NID))) {
        //     std::cout << "Warning: NaN in NID computation" << std::endl;
        //     return false;
        // }

        // static bool fileInitialized = false;
        // static std::string csv_filepath;
        // static int write_index = 0;

        // if (!fileInitialized) {
        //     // 生成时间戳文件名
        //     auto now = std::chrono::system_clock::now();
        //     auto now_c = std::chrono::system_clock::to_time_t(now);
        //     std::tm* t = std::localtime(&now_c);

        //     std::ostringstream oss;
        //     //shen 写入文件的路径 nid
        //     oss << "/home/yishu/data_process/nid_cost_log_"
        //         << std::put_time(t, "%Y%m%d_%H%M%S")
        //         << ".csv";

        //     csv_filepath = oss.str();
        //     fileInitialized = true;
        // }

        // bool fileExists = std::ifstream(csv_filepath).good();
        // std::ofstream ofs;
        // if (fileExists) {
        //     ofs.open(csv_filepath, std::ios::app);
        // } else {
        //     ofs.open(csv_filepath);
        //     // 写表头
        //     ofs << "index,NID_value" << std::endl;
        // }

        // if (ofs.is_open()) {
        //     ofs << write_index << "," << get_real(NID) << std::endl;
        //     ofs.close();
        //     ++write_index;
        //     // std::cout << "NID cost written to file" << std::endl;
        // } else {
        //     std::cerr << "[NIDCost] Unable to open file for writing NID cost!" << std::endl;
        // }

        return true;
    }

private:
    const cv::Mat normalized_image;
    const Frame::ConstPtr points;
    const int bins;
    Eigen::Matrix<double, 4, 4> spline_coeffs;
};

}  // namespace vlcal
