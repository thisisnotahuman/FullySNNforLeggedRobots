/************************************************************************
Copyright (c) 2023, DEEPROBOTICS
Copyright (c) 2023, HUAWEI TECHNOLOGIES.
All rights reserved.
Use of this source code is governed by the MIT license, see LICENSE.
************************************************************************/

#include "udpsocket.hpp"
#include "udpserver.hpp"
#include "command_list.h"
#include "parse_cmd.h"
#include "send_to_robot.h"
#include "motionexample.h"

#include <iostream>
#include <time.h>
#include <string.h>

using namespace std;

int main()
{
    double now_time, start_time;
    RobotCmd robot_joint_cmd;
    Time_Tool my_set_Timer;
    SendToRobot *send2robot_cmd = new SendToRobot; ///< 创建发送线程
    ParseCMD *robot_data_rec = new ParseCMD;       ///< 创建接收解析线程

    MotionExample robot_set_up_demo; ///< 测试用的demo，可自行删除

    // send2robot_cmd->startWork();
    send2robot_cmd->init();
    robot_data_rec->startWork();

    RobotState *robot_data = &robot_data_rec->get_recv();
    my_set_Timer.time_init(1);          ///< 定时器初始化，输入：周期；单位：ms
    send2robot_cmd->robot_state_init(); ///< 所有关节回归零位，并获取控制权

    start_time = my_set_Timer.get_start_time();                    ///< 获取时间，用于算法使用
    robot_set_up_demo.GetInitData(robot_data->motor_state, 0.000); ///< 获取所有关节状态，每个阶段（动作）前，都需要获取一次

    /********************************************************/
    int time_tick = 0;
    while (1)
    {
        if (my_set_Timer.time_interrupt() == 1)
        { ///< 时间中断标志，返回1，周期未到，返回0，达到一个周期
            continue;
        }
        now_time = my_set_Timer.get_now_time(start_time); ///< 获得当前时间
        printf("t = %f \n", now_time);
        /*******一个站起来的简单demo（测试用，可自行删除）*********/
        time_tick++;
        if (time_tick < 2000)
        {
            // robot_set_up_demo.PreStandUp(robot_data->motor_state, robot_joint_cmd, now_time); ///< 站起准备动作
        
        }
        RobotState *robot_data = &robot_data_rec->get_recv();
        printf("%f,  %f\n",robot_data->imu.angle_roll,
                robot_data->imu.angular_velocity_roll);
        // if (time_tick == 1000)
        // {
        //     robot_set_up_demo.GetInitData(robot_data->motor_state, now_time); ///< 获取所有关节状态，每个阶段（动作）前，都需要获取一次
        // }
        // if (time_tick >= 1000)
        // {
        //     robot_set_up_demo.StandUp(robot_data->motor_state, robot_joint_cmd, now_time); ///< 完全站起动作
        // }
        // if (time_tick >= 3000)
        // {
        //     send2robot_cmd->control_get(UNABLE); ///< 归还控制权，输入：UNABLE:机器人原算法控制  ABLE:SDK控制  PS：超过5ms，未发数据set_send(cmd)，会失去控制权，要重新发送获取控制权
        // }
        /*********一个站起来的简单demo（测试用，可自行删除）*******/
        send2robot_cmd->set_send(robot_joint_cmd); ///< 发送关节控制数据，给机器人，PS：超过5ms，未发数据set_send(cmd)，会失去控制权，要重新发送获取控制权
    }
    return 0;
}