/*
 * @Author: sys111111 sys2048194132@sjtu.edu.cn
 * @Date: 2025-03-06 21:12:17
 * @LastEditors: sys111111 sys2048194132@sjtu.edu.cn
 * @LastEditTime: 2025-07-17 20:52:34
 * @FilePath: /Joint_Direct_LiDAR_Camera_Calibration/include/vlcal/common/vector3i_hash.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <Eigen/Core>
#include <boost/functional/hash/hash.hpp>

namespace vlcal {

/**
 * @brief Spatial hashing function using boost::hash_combine
 */
class Vector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const {
    size_t seed = 0;
    boost::hash_combine(seed, x[0]);
    boost::hash_combine(seed, x[1]);
    boost::hash_combine(seed, x[2]);
    return seed;
  }
};

/**
 * @brief Spatial hashing function
 *        Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV2003
 */
class XORVector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const {
    const size_t p1 = 73856093;
    const size_t p2 = 19349669;  // 19349663 was not a prime number
    const size_t p3 = 83492791;
    return static_cast<size_t>((x[0] * p1) ^ (x[1] * p2) ^ (x[2] * p3));
  }
};

}  // namespace gtsam_ext