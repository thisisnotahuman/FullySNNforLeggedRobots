/**
 * @file main.cpp
 * @author xqp
 * @brief 
 * @version 0.1
 * @date 2022-09-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "motionexample.h"

Vec3 goal_angle_fl, goal_angle_hl, goal_angle_fr, goal_angle_hr;
Vec3 init_angle_fl, init_angle_fr, init_angle_hl, init_angle_hr;
double init_time;

void MotionExample::PreStandUp(RobotData data, RobotCmd &cmd, double time)
{   
    //简单摆腿，指定高度逆运动学站立可自行尝试
    double standup_time = 2.0;
    double cycle_time = 0.003;
    goal_angle_fl << 0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian;
    goal_angle_fr << 0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian;
    goal_angle_hl << 0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian;
    goal_angle_hr << 0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian;

    if (time <= init_time + standup_time) 
    {
        SwingToAngle(init_angle_fl, goal_angle_fl, standup_time, time - init_time, cycle_time, "FL", cmd);
        SwingToAngle(init_angle_fr, goal_angle_fr, standup_time, time - init_time, cycle_time, "FR", cmd);
        SwingToAngle(init_angle_hl, goal_angle_hl, standup_time, time - init_time, cycle_time, "HL", cmd);
        SwingToAngle(init_angle_hr, goal_angle_hr, standup_time, time - init_time, cycle_time, "HR", cmd);
    }
}

void MotionExample::StandUp(RobotData data, RobotCmd &cmd, double time)
{   
    //简单摆腿，指定高度逆运动学站立可自行尝试
    double standup_time = 2.0;
    double cycle_time = 0.001;
    goal_angle_fl << 0 * kDegree2Radian, -36 * kDegree2Radian, 85 * kDegree2Radian;
    goal_angle_fr << 0 * kDegree2Radian, -36 * kDegree2Radian, 85 * kDegree2Radian;
    goal_angle_hl << 0 * kDegree2Radian, -36 * kDegree2Radian, 85 * kDegree2Radian;
    goal_angle_hr << 0 * kDegree2Radian, -36 * kDegree2Radian, 85 * kDegree2Radian;

    if (time <= init_time + standup_time) 
    {
        SwingToAngle(init_angle_fl, goal_angle_fl, standup_time, time - init_time, cycle_time, "FL", cmd);
        SwingToAngle(init_angle_fr, goal_angle_fr, standup_time, time - init_time, cycle_time, "FR", cmd);
        SwingToAngle(init_angle_hl, goal_angle_hl, standup_time, time - init_time, cycle_time, "HL", cmd);
        SwingToAngle(init_angle_hr, goal_angle_hr, standup_time, time - init_time, cycle_time, "HR", cmd);
        for(int i = 0; i < 12; i++){
            cout << data.joint_data[i].pos << "  ";
        }
        cout << endl;
    } 
}

void MotionExample::GetInitData(RobotData data, double time)
{   
    init_time = time;
    //只记录了当前时刻的角度值
    init_angle_fl[0] = data.fl_leg[0].pos;
    init_angle_fl[1] = data.fl_leg[1].pos;
    init_angle_fl[2] = data.fl_leg[2].pos;

    init_angle_fr[0] = data.fr_leg[0].pos;
    init_angle_fr[1] = data.fr_leg[1].pos;
    init_angle_fr[2] = data.fr_leg[2].pos;

    init_angle_hl[0] = data.hl_leg[0].pos;
    init_angle_hl[1] = data.hl_leg[1].pos;
    init_angle_hl[2] = data.hl_leg[2].pos;

    init_angle_hr[0] = data.hr_leg[0].pos;
    init_angle_hr[1] = data.hr_leg[1].pos;
    init_angle_hr[2] = data.hr_leg[2].pos;
}

void MotionExample::SwingToAngle(Vec3 initial_angle, Vec3 goal_xyz_angle, double total_time, double run_time, double cycle_time, string side, RobotCmd &cmd)
{
    Vec3 goal_angle;
	Vec3 goal_angle_next;
	Vec3 goal_angle_next2;
    Vec3 goal_vel;
	Vec3 final_angle;
    int leg_side;

    if(side == "FL") leg_side = 0;
    else if(side == "FR") leg_side = 1;
    else if(side == "HL") leg_side = 2;
    else if(side == "HR") leg_side = 3;
    else cout<<"Leg Side Error!!!"<<endl;
    
    final_angle = goal_xyz_angle;
	for (int j = 0; j<3; j++) {
		CubicSpline(initial_angle[j], 0, (double)final_angle[j], 0,
			run_time, cycle_time, total_time, goal_angle[j], goal_angle_next[j], goal_angle_next2[j]);
	}

	goal_vel = (goal_angle_next - goal_angle) / cycle_time;
    
    //简单pd控制
    cmd.joint_cmd[3*leg_side].kp = 60;
    cmd.joint_cmd[3*leg_side+1].kp = 80;
    cmd.joint_cmd[3*leg_side+2].kp = 80;
    cmd.joint_cmd[3*leg_side].kd = 0.7;
    cmd.joint_cmd[3*leg_side+1].kd = 0.7;
    cmd.joint_cmd[3*leg_side+2].kd = 1.2;
    cmd.joint_cmd[3*leg_side].pos = goal_angle[0];
    cmd.joint_cmd[3*leg_side+1].pos = goal_angle[1];
    cmd.joint_cmd[3*leg_side+2].pos = goal_angle[2];
    cmd.joint_cmd[3*leg_side].vel = goal_vel[0];
    cmd.joint_cmd[3*leg_side+1].vel = goal_vel[1];
    cmd.joint_cmd[3*leg_side+2].vel = goal_vel[2];
    for(int i = 0; i < 12; i++){
		cmd.joint_cmd[i].tor = 0;
    }

}

void MotionExample::CubicSpline(double init_pos, double init_vel, double goal_pos, double goal_vel, double run_time, double cycle_time, double total_time, double &sub_goal_pos, double &sub_goal_pos_next, double &sub_goal_pos_next2)
{
	double a, b, c, d;
	d = init_pos;
	c = init_vel;
	a = (goal_vel * total_time - 2 * goal_pos + init_vel * total_time + 2 * init_pos) / pow(total_time, 3);
	b = (3 * goal_pos - goal_vel * total_time - 2 * init_vel*total_time - 3 * init_pos) / pow(total_time, 2);

	if (run_time > total_time)
		run_time = total_time;
	sub_goal_pos = a * pow(run_time, 3) + b * pow(run_time, 2) + c * run_time + d;

	if (run_time + cycle_time > total_time)
		run_time = total_time - cycle_time;
	sub_goal_pos_next = a * pow(run_time + cycle_time, 3) + b * pow(run_time + cycle_time, 2) + c * (run_time + cycle_time) + d;

	if (run_time + cycle_time * 2 > total_time)
		run_time = total_time - cycle_time * 2;
	sub_goal_pos_next2 = a * pow(run_time + cycle_time * 2, 3) + b * pow(run_time + cycle_time * 2, 2) + c * (run_time + cycle_time * 2) + d;
}