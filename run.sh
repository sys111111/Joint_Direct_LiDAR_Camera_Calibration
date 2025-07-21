# cd build/devel/lib/joint_calibration &&
#   ./preprocess \
#   -av \
#   --camera_model=plumb_bob \
#   --camera_intrinsics=0,0,0,0 \
#   --camera_distortion_coeffs=0,0,0,0,0 \
#   /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/bag_0716_data2 \
#   /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/pre_0716_data2
  
## Superglue Match And Initial Guess ##
# cd scripts/ &&
# python3 superglue_matches.py /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/pre_0716_data2 --rotate_camera 0

## Initial Extrinsic From CAD ##
# cd scripts/ && \
# python initial_extrinsic.py \
#   0.05643,0.00039,0.06362,-0.5,0.5,-0.5,0.5 \
#   --data_path /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/pre_0716_data2

## Initial Intrinsic Guess Auto ##
# cd build/devel/lib/joint_calibration &&
#   ./initial_guess_auto \
#   /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/pre_0716_data2

## intrinsics_calibrate ##
# cd build/devel/lib/joint_calibration &&
#   ./intrinsics_calibrate \
#   /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/pre_0716_data2

## intrinsics_calibrate ##
cd build/devel/lib/joint_calibration &&
  ./extrinsics_calibrate \
  /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/pre_0716_data2

# cd build/devel/lib/joint_calibration &&
#   gdb --args ./extrinsics_calibrate \
#   /home/syss/Joint_Direct_LiDAR_Camera_Calibration/livox_data/pre_0716_data2