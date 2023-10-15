# SPDX-FileCopyrightText: Copyright (c) 2023, HUAWEI TECHNOLOGIES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]

# position mode PD
ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
PI = math.pi

JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
# COM_OFFSET = -np.array([0.005, 0.0028, 0.000515])  # x++ back++     y++ right++
COM_OFFSET = -np.array([0.0, 0.0, 0.0])  # x++ back++     y++ right++

HIP_OFFSETS = np.array([[0.1805, -0.047, 0.], [0.1805, 0.047, 0.],
                        [-0.1805, -0.047, 0.], [-0.1805, 0.047, 0.]
                        ]) + COM_OFFSET

HIP_POSITION = np.array([[0.185, -0.135, 0], [0.185, 0.135, 0],
                         [-0.185, -0.135, 0], [-0.185, 0.135, 0]])

# unitree a1_robot
MPC_BODY_MASS = 10
MPC_BODY_INERTIA = np.array((0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00))  # google
MPC_BODY_HEIGHT = 0.28

STAND_UP_HEIGHT = MPC_BODY_HEIGHT
LEG_LENGTH = 0.20
UPPER_LEG_LENGTH = 0.20
LOWER_LEG_LENGTH = 0.20
HIP_LENGTH = 0.08505
STANDUP_ABDUCTION_ANGLE = 0.0
STANDUP_HIP_ANGLE = np.arccos(STAND_UP_HEIGHT / 2.0 / LEG_LENGTH)
STANDUP_KNEE_ANGLE = -2.0 * STANDUP_HIP_ANGLE
INIT_MOTOR_ANGLES = np.array(
    [STANDUP_ABDUCTION_ANGLE, STANDUP_HIP_ANGLE, STANDUP_KNEE_ANGLE] *
    NUM_LEGS)
SITDOWN_MOTOR_ANGLES = np.array([0.0, 0.93, -2.6] * NUM_LEGS)


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        np.clip((x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
                (2 * l_low * l_up), -1.0, 1.0))
    l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(np.clip(-x / l, -1.0, 1.0)) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])


def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = np.sqrt(l_up**2 + l_low**2 +
                           2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])


def analytical_leg_jacobian(leg_angles, leg_id):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
      l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1)**(leg_id + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2

    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(
        t_eff) / l_eff - l_eff * np.cos(t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J


# For JIT compilation
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), 1)
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), -1)


def foot_positions_in_base_frame(foot_angles):
    foot_angles = foot_angles.reshape((4, 3))
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                       l_hip_sign=(-1)**(i +
                                                                         1))
    return foot_positions + HIP_OFFSETS
