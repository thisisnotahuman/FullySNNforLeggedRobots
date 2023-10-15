sudo nvpmodel -m 8 & sleep 1;
sudo ./scrips/max_cpu_freq.sh;
sudo ./scrips/max_emc_freq.sh;
sudo ./scrips/max_gpu_freq.sh;

source ../devel/setup.bash
sudo chmod 777 /dev/ttyTHS1

roslaunch realsense2_camera rs_camera.launch & sleep 1;
roslaunch mavros px4.launch & sleep 10;
rosrun vins vins_node /home/nv/demo-ego-swarm-ws/src/realflight_modules/VINS-Fusion-gpu/config/px4/stereo_imu_config.yaml & sleep 1;

# roslaunch px4ctrl run_ctrl.launch & sleep 1;

# roslaunch ego_planner run_in_exp.launch & sleep 1;

wait;
