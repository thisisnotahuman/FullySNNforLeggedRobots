#include <string>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int64.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <tf/transform_datatypes.h>
// // #include <algorithm>
// #include <math.h>


int cmd_wait_time = 1314520;
int odom_wait_time = 7758258;
float cmd_p_x, cmd_p_y,
    cmd_v_x, cmd_v_y, cmd_yaw,
    odom_x, odom_y, odom_yaw,
    vel_x, vel_y, vel_yaw,
    vel_forward, vel_side;

float K_pos_forward, K_pos_side, K_yaw, K_bbox_offset;
float K_vel_forward, K_vel_side, K_vel_yaw;
float v_forward_max, v_side_max, yawrate_max;
std_msgs::Int64 prepare_to_go_;
float bbox_center_offset = 0;


void cmdCb(const quadrotor_msgs::PositionCommandConstPtr &cmd)
{
    std::cout << "receive set point\n";
    cmd_p_x = cmd->position.x;
    cmd_p_y = cmd->position.y;
    cmd_v_x = cmd->velocity.x;
    cmd_v_y = cmd->velocity.y;
    cmd_yaw = cmd->yaw; // -pi ~ pi
    std::cout << "cmd_yaw "  << cmd_yaw<< std::endl;
    cmd_wait_time = 0;
}
bool first_start = true;

void odomCb(const nav_msgs::Odometry::ConstPtr &odom)
{
    static double x_offset = 0;
    static double y_offset = 0;
    if(false)
    {
        x_offset = odom->pose.pose.position.x;
        y_offset = odom->pose.pose.position.y;
        first_start = false;
    }
    tf::Quaternion q;
    tf::quaternionMsgToTF(odom->pose.pose.orientation, q);
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    odom_x = odom->pose.pose.position.x - x_offset;
    odom_y = odom->pose.pose.position.y - y_offset;
    odom_yaw = yaw; // -pi ~ pi
    odom_wait_time = 0;
    vel_x = odom->twist.twist.linear.x;
    vel_y = odom->twist.twist.linear.y;
    vel_yaw = odom->twist.twist.angular.z;  // rad

    vel_forward = vel_x * cos(odom_yaw) + vel_y * sin(odom_yaw);
    vel_side = vel_x * -sin(odom_yaw) + vel_y * cos(odom_yaw);
}

void aimCallback(const nav_msgs::OdometryConstPtr &msg){

ROS_INFO("Received aim: %f, %f, %d", msg->pose.pose.position.x, msg->pose.pose.position.y, 0);
bbox_center_offset = msg->pose.pose.position.y;
}

int mainHelper()
{
    std::cout << "WARNING: Control level is set to HIGH-level." << std::endl
              << "Make sure the robot is standing on the ground." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    ros::NodeHandle nh("~");
    K_pos_forward = nh.param("K_pos_forward", 1.0);
    K_pos_side = nh.param("K_pos_side", 1.0);
    K_vel_forward = nh.param("K_vel_forward", 0.0);
    K_vel_side = nh.param("K_vel_side", 0.0);
    K_yaw = nh.param("K_yaw", 1.0);
    K_bbox_offset = nh.param("K_bbox_offset", 0.0);
    v_forward_max = nh.param("v_forward_max", 23333.0);
    v_side_max = nh.param("v_side_max", 23333.0);
    yawrate_max = nh.param("yawrate_max", 23333.0);

    // ros::Publisher imu_pub = nh.advertise<sensor_msgs::Imu>("/hardware_a1/imu", 100);
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 100);
    ros::Publisher odom_viz_pub = nh.advertise<nav_msgs::Odometry>("/real_odom", 100);
     ros::Publisher dog_state_pub = nh.advertise<std_msgs::Int64>("/dog_state", 100);
    ros::Subscriber cmd_sub = nh.subscribe("/setpoints_cmd", 30, cmdCb, ros::TransportHints().tcpNoDelay());
    // ros::Subscriber odom_sub = nh.subscribe("/vrpn_client_node/Dog/pose", 30, odomCb, ros::TransportHints().tcpNoDelay());
    ros::Subscriber odom_sub = nh.subscribe("/Odometry", 30, odomCb, ros::TransportHints().tcpNoDelay());
    // ros::Subscriber odom_sub = nh.subscribe("/leg_odom", 30, odomCb, ros::TransportHints().tcpNoDelay());
    ros::Subscriber aim_sub = nh.subscribe("/object_tracking/aim/pose", 30, aimCallback, ros::TransportHints().tcpNoDelay());

    ros::Rate loop_rate(500);

    float cmd_v_forward, cmd_v_side;
    float err_p_x, err_p_y;
    float err_p_forward, err_p_side;

    float u_v_forward = 0;
    float u_v_side = 0;
    float u_yawrate = 0;

    double error_yaw = 0;

    enum FSM_STATE
    {
        IDLE,
        WALK,
        SWITCH2WALK,
        SWITCH2IDLE
    } fsm_state = IDLE;
    long switch_time = 0; // ms
    long motion_time = 0;
    bool yaw_control_only = false;
    std::cout << "ready to walk" << std::endl;
    while (ros::ok())
    {
        cmd_wait_time += 2;
        odom_wait_time += 2;
        motion_time += 2;
        
        ros::spinOnce();

        if (fsm_state == IDLE)
        {
            if (cmd_wait_time < 1000 && odom_wait_time < 1000)
            {
                fsm_state = SWITCH2WALK;
                std::cout << "switch to walk" << std::endl;
            }
        }
        else if (fsm_state == WALK)
        {
            if (cmd_wait_time >= 1000 || odom_wait_time >= 1000)
            {
                fsm_state = SWITCH2IDLE;
                std::cout << "switch to idle" << std::endl;
                u_v_forward = 0;
                u_v_side = 0;
                u_yawrate = 0;
            }
            else
            {
                // cmd_v_x = 0;
                // cmd_v_y = 0;
                // forward norvec = [cos sin], side norvec = [-sin cos]
                cmd_v_forward = cmd_v_x * cos(odom_yaw) + cmd_v_y * sin(odom_yaw);
                cmd_v_side = cmd_v_x * -sin(odom_yaw) + cmd_v_y * cos(odom_yaw);
                error_yaw = cmd_yaw - odom_yaw;
                std::cout << "cmd_yaw1 "  << cmd_yaw<< std::endl;
                std::cout << "odom_yaw "  << odom_yaw<< std::endl;
                 std::cout << "error_yaw "  << error_yaw<< std::endl;
                if (fabs(error_yaw) > 3.1415926)
                {
                    if (error_yaw>0)
                    {
                        error_yaw = 1*(error_yaw - 2*3.1415926);
                    }else
                    {
                        error_yaw = 1*(error_yaw + 2 * 3.1415926);
                    }
                }
                std::cout << "error_yaw_modify "  << error_yaw<< std::endl;
                if(fabs(error_yaw)>90*3.1415926/180.0 || yaw_control_only)
                {
                    yaw_control_only = true;
                    odom_wait_time = 0;
                    cmd_wait_time = 0;
                    std::cout << "yaw is larger than 90, only  move yaw" << std::endl;
                    cmd_p_x = odom_x;
                    cmd_p_y = odom_y;
                    if (fabs(error_yaw)<8*3.1415926/180.0)
                    {
                        yaw_control_only = false;
                    }
                }
                if (yaw_control_only)
                {
                    prepare_to_go_.data = 0;
                }
                else
                {
                    prepare_to_go_.data = 1;
                }
                dog_state_pub.publish(prepare_to_go_);

                std::cout << "error_yaw: " << error_yaw << std::endl;
                std::cout << "bbox_center_offset: " << bbox_center_offset << std::endl;

                u_yawrate = K_yaw * error_yaw + K_bbox_offset * bbox_center_offset ;

                double error_v_forward, error_v_side;

                error_v_forward = cmd_v_forward - vel_forward;
                error_v_side = cmd_v_side - vel_side;
                
                err_p_x = cmd_p_x - odom_x;
                err_p_y = cmd_p_y - odom_y;
                err_p_forward = err_p_x * cos(odom_yaw) + err_p_y * sin(odom_yaw);
                err_p_side = err_p_x * -sin(odom_yaw) + err_p_y * cos(odom_yaw);

                u_v_forward = 0 + K_pos_forward * err_p_forward + K_vel_forward * error_v_forward;
                u_v_side = 0 + K_pos_side * err_p_side + K_vel_side * error_v_side;
            }
        }
        else if (fsm_state == SWITCH2WALK)
        {
            switch_time += 2;
            if (switch_time >= 1000)
            {
                fsm_state = WALK;
                switch_time = 0;
            }
        }
        else if (fsm_state == SWITCH2IDLE)
        {
            switch_time += 2;
            if (switch_time >= 1000)
            {
                fsm_state = IDLE;
                switch_time = 0;
            }
        }    

        double therd = 3;
        if (u_v_forward < -therd)
            u_v_forward = -therd;
        else if(u_v_forward > therd)
            u_v_forward = therd;
            
        double therd_side = 3.0;
        if (u_v_side < -therd_side)
            u_v_side = -therd_side;
        else if(u_v_side > therd_side)
            u_v_side = therd_side;

        if (u_yawrate < -1)
            u_yawrate = -1;
        else if(u_yawrate > 1)
            u_yawrate = 1;

        geometry_msgs::Twist vel_cmd;
        vel_cmd.linear.x =  u_v_forward;
        vel_cmd.linear.y = u_v_side;
        vel_cmd.linear.z = 0;

        vel_cmd.angular.x = 0;
        vel_cmd.angular.y = 0;
        vel_cmd.angular.z = u_yawrate;
        
        cmd_vel_pub.publish(vel_cmd);

        nav_msgs::Odometry odom_viz_pub_msg;
        odom_viz_pub_msg.header.frame_id = "world";
        odom_viz_pub_msg.pose.pose.position.x = odom_x;
        odom_viz_pub_msg.pose.pose.position.y = odom_y;
        odom_viz_pub_msg.pose.pose.orientation.z = odom_yaw;
        // odom_yaw = yaw; // -pi ~ pi
        odom_viz_pub.publish(odom_viz_pub_msg);
        loop_rate.sleep();
    }


    return 0;
}


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "walk_ros_mode");

    mainHelper();

    return 0;
}