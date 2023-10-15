/************************************************************************
Copyright (c) 2023, HUAWEI TECHNOLOGIES. All rights reserved.
Use of this source code is governed by the MIT license, see LICENSE.
************************************************************************/

#include "include/deeprobotics_legged_sdk/send_to_robot.h"
#include "include/deeprobotics_legged_sdk/parse_cmd.h"
#include <array>
#include <math.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


void CopyMotorData(MotorData& dest, MotorData& source)
{
    dest.pos = source.pos;
    dest.vel = source.vel;
    dest.tor = source.tor;
    dest.temperature = source.temperature;
}

class Lite3Interface
{
public:
    Lite3Interface()
    {
        lite2Sender.init();
        lite2Receiver.startWork();
        // lite2Sender.control_get(ABLE);
        lite2Sender.robot_state_init();
    }
    RobotStateArray& ReceiveObservation();
    void SendCommand(std::array<float, 60> motorcmd);
    void Initialize()
    {
        lite2Sender.control_get(ABLE);
    }

    ParseCMD lite2Receiver;
    SendToRobot lite2Sender;  
    RobotStateArray lowStateArray;
    RobotCmd cmd;
};

RobotStateArray& Lite3Interface::ReceiveObservation() {
    
    RobotState lowState = lite2Receiver.get_recv();
    
    lowStateArray.tick = lowState.tick;
    lowStateArray.imu = lowState.imu;
    for (int i=0; i<3; ++i) {
        CopyMotorData(lowStateArray.motor_state.fl_leg[i], lowState.motor_state.fl_leg[i]);
        CopyMotorData(lowStateArray.motor_state.fr_leg[i], lowState.motor_state.fr_leg[i]);
        CopyMotorData(lowStateArray.motor_state.hl_leg[i], lowState.motor_state.hl_leg[i]);
        CopyMotorData(lowStateArray.motor_state.hr_leg[i], lowState.motor_state.hr_leg[i]);
    }
    // printf("fl_leg[1] = %f, %f, %f, %f\n", lowState.motor_state.fl_leg[1].pos,lowState.motor_state.fl_leg[1].vel, lowState.motor_state.fl_leg[1].tor, lowState.motor_state.fl_leg[1].temperature);
    // printf("-->fl_leg[1]= %f, %f, %f, %f\n", lowStateArray.motor_state.fl_leg[1].pos,lowStateArray.motor_state.fl_leg[1].vel, lowStateArray.motor_state.fl_leg[1].tor, lowStateArray.motor_state.fl_leg[1].temperature);
    // std::copy(std::begin(lowState.motor_state.fl_leg), std::begin(lowState.motor_state.fl_leg),std::begin(lowStateArray.motor_state.fl_leg));
    // std::copy(std::begin(lowState.motor_state.fr_leg), std::begin(lowState.motor_state.fr_leg),std::begin(lowStateArray.motor_state.fr_leg));
    // std::copy(std::begin(lowState.motor_state.hl_leg), std::begin(lowState.motor_state.hl_leg),std::begin(lowStateArray.motor_state.hl_leg));
    // std::copy(std::begin(lowState.motor_state.hr_leg), std::begin(lowState.motor_state.hr_leg),std::begin(lowStateArray.motor_state.hr_leg));
    
    std::copy(std::begin(lowState.fl_tor), std::end(lowState.fl_tor),std::begin(lowStateArray.fl_tor));
    std::copy(std::begin(lowState.fr_tor), std::end(lowState.fr_tor),std::begin(lowStateArray.fr_tor));
    std::copy(std::begin(lowState.hl_tor), std::end(lowState.hl_tor),std::begin(lowStateArray.hl_tor));
    std::copy(std::begin(lowState.hr_tor), std::end(lowState.hr_tor),std::begin(lowStateArray.hr_tor));

    return lowStateArray;
}

void Lite3Interface::SendCommand(std::array<float, 60> motorcmd) {
    // Eigen::Matrix<float, 5, 12> motorCommandsShaped = motorCommands;
    // Eigen::Matrix<float, 12, 1> angles = motorCommandsShaped.row(POSITION).transpose();
    // motorCommandsShaped.row(POSITION) = (jointDirection.cwiseProduct(angles) - jointOffset).transpose();
    // Eigen::Matrix<float, 12, 1> vels = motorCommandsShaped.row(VELOCITY).transpose();
    // motorCommandsShaped.row(VELOCITY) = jointDirection.cwiseProduct(vels).transpose();
    // Eigen::Matrix<float, 12, 1> tuas = motorCommandsShaped.row(TORQUE).transpose();
    // motorCommandsShaped.row(TORQUE) = jointDirection.cwiseProduct(tuas).transpose();
    
    for (int motorId = 0; motorId < 12; ++motorId) {
        // int motorId_ = (motorId/3)%2 == 0? motorId+3: motorId-3;
        cmd.joint_cmd[motorId].pos = motorcmd[motorId * 5];
        cmd.joint_cmd[motorId].kp = motorcmd[motorId * 5 + 1];
        cmd.joint_cmd[motorId].vel = motorcmd[motorId * 5 + 2];
        cmd.joint_cmd[motorId].kd = motorcmd[motorId * 5 + 3];
        cmd.joint_cmd[motorId].tor = motorcmd[motorId * 5 + 4];
    }
    // std::cout << "cmd = " << cmd.joint_cmd[0] << "\n" << cmd.joint_cmd[1] << "\n" << cmd.joint_cmd[2] << "\n";
    lite2Sender.set_send(cmd);
}

namespace py = pybind11;

// Expose all of comm.h and the Lite3Interface Class.
PYBIND11_MODULE(lite3_interface, m) {
    m.doc() = R"pbdoc(
          Lite3 Robot Interface Python Bindings
          -----------------------
          .. currentmodule:: lite3_robot_interface
          .. autosummary::
             :toctree: _generate
      )pbdoc";
    
    py::class_<ImuData>(m, "IMU")
        .def(py::init<>())
        .def_readwrite("angle_roll", &ImuData::angle_roll)
        .def_readwrite("angle_pitch", &ImuData::angle_pitch)
        .def_readwrite("angle_yaw", &ImuData::angle_yaw)
        .def_readwrite("angular_velocity_roll", &ImuData::angular_velocity_roll)
        .def_readwrite("angular_velocity_pitch", &ImuData::angular_velocity_pitch)
        .def_readwrite("angular_velocity_yaw", &ImuData::angular_velocity_yaw)
        .def_readwrite("acc_x", &ImuData::acc_x)
        .def_readwrite("acc_y", &ImuData::acc_y)
        .def_readwrite("acc_z", &ImuData::acc_z);

    py::class_<MotorData>(m, "MotorState")
        .def(py::init<>())
        .def_readwrite("q", &MotorData::pos)
        .def_readwrite("dq", &MotorData::vel)
        .def_readwrite("tauEst", &MotorData::tor)
        .def_readwrite("temperature", &MotorData::temperature);

    py::class_<RobotDataArray>(m, "RobotDataArray")
        .def(py::init<>())
        .def_readwrite("fl_leg", &RobotDataArray::fl_leg)
        .def_readwrite("fr_leg", &RobotDataArray::fr_leg)
        .def_readwrite("hl_leg", &RobotDataArray::hl_leg)
        .def_readwrite("hr_leg", &RobotDataArray::hr_leg);

    py::class_<JointCmd>(m, "MotorCmd")
        .def(py::init<>())
        .def_readwrite("q", &JointCmd::pos)
        .def_readwrite("dq", &JointCmd::vel)
        .def_readwrite("tau", &JointCmd::tor)
        .def_readwrite("Kp", &JointCmd::kp)
        .def_readwrite("Kd", &JointCmd::kd);
    
    py::class_<RobotStateArray>(m, "LowState")
        .def(py::init<>())
        .def_readwrite("tick", &RobotStateArray::tick)
        .def_readwrite("imu", &RobotStateArray::imu)
        .def_readwrite("motorState", &RobotStateArray::motor_state)
        .def_readwrite("fl_tor", &RobotStateArray::fl_tor)
        .def_readwrite("fr_tor", &RobotStateArray::fr_tor)
        .def_readwrite("hl_tor", &RobotStateArray::hl_tor)
        .def_readwrite("hr_tor", &RobotStateArray::hr_tor);

    // py::class_<RobotCmd>(m, "LowCmd")
    //     .def(py::init<>());
        // .def_readwrite("fl_leg", &RobotCmd::fl_leg)
        // .def_readwrite("fr_leg", &RobotCmd::fr_leg)
        // .def_readwrite("hl_leg", &RobotCmd::hl_leg)
        // .def_readwrite("hr_leg", &RobotCmd::hr_leg);

    py::class_<Lite3Interface>(m, "Lite3Interface")
        .def(py::init<>())
        .def("receive_observation", &Lite3Interface::ReceiveObservation)
        .def("send_command", &Lite3Interface::SendCommand)
        .def("initialize", &Lite3Interface::Initialize);

    #ifdef VERSION_INFO
      m.attr("__version__") = VERSION_INFO;
    #else
      m.attr("__version__") = "dev";
    #endif

      m.attr("TEST") = py::int_(int(42));

}
