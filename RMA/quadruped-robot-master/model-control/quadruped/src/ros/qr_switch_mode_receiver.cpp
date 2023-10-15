// The MIT License

// Copyright (c) 2022
// Robot Motion and Vision Laboratory at East China Normal University
// Contact: tophill.robotics@gmail.com

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "ros/qr_switch_mode_receiver.h"


namespace Quadruped {

qrSwitchModeReceiver::qrSwitchModeReceiver(ros::NodeHandle &nhIn, ros::NodeHandle &privateNhIn):
    nh(nhIn),
    privateNh(privateNhIn)
{
    ROS_INFO("switch mode topic: %s", switchModeTopic.c_str());
    switchModeSub = nh.subscribe(switchModeTopic, 10, &qrSwitchModeReceiver::SwitchModeCallback, this);
}


void qrSwitchModeReceiver::SwitchModeCallback(const std_msgs::Int8::ConstPtr &input)
{
    switchMode = input->data;
}

} // Namespace Quadruped

