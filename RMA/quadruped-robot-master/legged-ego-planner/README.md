# legged-ego-planner

## Introduction
This is a modified version of ego-planner for legged robot, which is a lightweight gradient-based local planner without ESDF construction. It significantly reduces computation time compared to some state-of-the-art methods.
<p align="center">
  <img src="src/docs/tracking.gif" width="800" />
</p>

## Software architecture
This repository consists of below directories:
- ego_planner: The modified version of ego-planner for legged robot.
- legged_real: The path following node which publish the velocity commands for real robot.


## Prepare environment 
For environment preparation, please refer to [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner) to install related dependencies. Then compile the workspace.
```
cd legged-ego-planner
catkin_make
```


# Usage

### View the results in rviz
- Download the [rosbag](https://ascend-devkit-tool.obs.cn-south-1.myhuaweicloud.com/CANN/tracking.bag) for test.
    
- Launch the planner and play the rosbag. 
```
source devel/setup.bash
roslaunch ego_planner run_in_exp.launch
roslaunch ego_planner rviz.launch
rosbag play tracking.bag
```

### Run on real legged robot
For real robot deployment, there are some other modules needed.
* Point cloud or depth image from lidar or depth camera need to be published to construct the grid map.
* An odometry or localization module is needed to provide the pose of the robot.

Please check the `odom_topic` `cloud_topic` `depth_topic` in `run_in_exp.launch` for more details.
```
source devel/setup.bash
roslaunch ego_planner run_in_exp.launch
roslaunch ego_planner rviz.launch
roslaunch legged_real trace.launch
```

## Acknowledgements
- This work extends [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner) to legged robot navigation.

## Communication
If you have any question, please join our discussion group by scanning the following wechat QR code.

<img src="src/docs/QR-code.jpg" alt="QR" title="" width="200" align=center />