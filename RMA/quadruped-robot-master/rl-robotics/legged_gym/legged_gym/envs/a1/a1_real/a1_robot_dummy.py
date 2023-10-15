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

import torch
from legged_gym.utils.math import *
from .robot_utils import *


class A1Dummy(object):

    def __init__(self, time_step=0.002, **kwargs):
        """Initializes the robot class."""
        print("Dummy Robot MPC_BODY_MASS: ", MPC_BODY_MASS)
        print("Dummy Robot MPC_BODY_INERTIA: ", MPC_BODY_INERTIA)
        print("Dummy Robot MPC_BODY_HEIGHT: ", MPC_BODY_HEIGHT)

        self.device = kwargs.get('device', torch.device('cpu'))
        print("A1Dummy use ", self.device)
        self.time_step = time_step
        self.num_envs = kwargs.get('num_envs', 1)
        # Robot state variables
        self._base_orientation = None
        self.base_rpy = torch.zeros((self.num_envs, 3),
                                    dtype=torch.float,
                                    device=self.device)
        self._motor_angles = torch.zeros(12)
        self._motor_velocities = torch.zeros(12)
        self._joint_states = None
        self.foot_contact = None
        self.estimated_velocity = torch.zeros((self.num_envs, 3),
                                              dtype=torch.float,
                                              device=self.device)
        self.base_rpyrate = torch.zeros((self.num_envs, 3),
                                        dtype=torch.float,
                                        device=self.device)
        self.base_lin_acc = None
        self.base_quat = None
        self._init_complete = True
        self.foot_positions_in_base_frame = torch.zeros((self.num_envs, 4, 3),
                                                        dtype=torch.float,
                                                        device=self.device)
        self.foot_positions_in_base_frame[:, :, 2] -= 0.28
        self.footVelocitiesInBaseFrame = torch.zeros_like(
            self.foot_positions_in_base_frame)
        self.torch_hip_offset = torch.from_numpy(HIP_OFFSETS).to(
            dtype=torch.float, device=self.device).unsqueeze(0)  # (1,4,3)
        self.l_hip = torch.tensor([[0.08505, -0.08505, 0.08505, -0.08505]],
                                  dtype=torch.float,
                                  device=self.device)  # (1,4)
        self.torch_hip_offset[:, 0:2, :] = self.torch_hip_offset[:, [1, 0], :]
        self.torch_hip_offset[:, 2:4, :] = self.torch_hip_offset[:, [3, 2], :]

    def GetBaseRollPitchYaw(self):
        return getEulerFromQuaternion(self._base_orientation)

    def GetTrueBaseRollPitchYaw(self):
        return getEulerFromQuaternion(self._base_orientation)

    def GetBaseRollPitchYawRate(self):
        return self.GetTrueBaseRollPitchYawRate()

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        return self.foot_positions_in_base_frame

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        motor_angles = self._motor_angles[:, leg_id * 3:(leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles.view(-1), leg_id)

    def GetBaseAcc(self):
        return self.base_lin_acc

    def GetBaseOrientation(self):
        return self._base_orientation

    def GetBaseRPY(self):
        return self.base_rpy

    def GetMotorVelocities(self):
        return self._motor_velocities

    def GetFootContacts(self):
        return self.foot_contact

    def GetFootVelocitiesInBaseFrame(self):
        return self.footVelocitiesInBaseFrame

    def set_state(self, original_obs_buf):
        self.base_lin_acc = original_obs_buf[:, :3]
        self.base_rpy = original_obs_buf[:, 3:6]
        self.base_rpyrate = original_obs_buf[:, 6:9]
        self._motor_angles = original_obs_buf[:, 9:21]
        self._motor_velocities = original_obs_buf[:, 21:33]
        self.foot_contact = original_obs_buf[:, 33:37]
        jacobians = self.torch_analytical_leg_jacobian(self._motor_angles)
        self.footVelocitiesInBaseFrame[:] = torch.matmul(
            jacobians, self._motor_velocities.view(-1, 4, 3, 1)).squeeze(-1)

    def torch_analytical_leg_jacobian(self, leg_angles):
        # J.shape = (None, 4, 3, 3)
        # Q.shape = (None, 4, 9, 3)
        # x.shape = (None, 4, 3, 1)
        # Q*x = -- > (None,4 9,1)-->(None, 4 3, 3)
        N = leg_angles.size(0)
        Q = torch.zeros((N, 4, 9, 3),
                        dtype=torch.float,
                        device=leg_angles.device)
        x = torch.zeros((N, 4, 3, 1),
                        dtype=torch.float,
                        device=leg_angles.device)
        motor_angles = self._motor_angles.view(N, 4, 3)
        t1 = torch.select(motor_angles, -1, 0)  # (N, 4)
        t2 = torch.select(motor_angles, -1, 1)  # (N, 4)
        t3 = torch.select(motor_angles, -1, 2)  # (N, 4)
        t_eff = t2 + t3 / 2
        l_eff = torch.sqrt(0.08 + 0.08 * torch.cos(t3))  # (N, 4)
        x[:, :, 0, 0] = self.l_hip
        x[:, :, 1, 0] = l_eff
        x[:, :, 2, 0] = 0.04 / l_eff

        c_t_eff = torch.cos(t_eff)
        s_t_eff = torch.sin(t_eff)
        s_t1 = torch.sin(t1)
        c_t1 = torch.cos(t1)
        s_t3 = torch.sin(t3)
        c_t3 = torch.cos(t3)

        Q[:, :, 1, 1] = -c_t_eff
        Q[:, :, 2, 1] = c_t_eff * (-0.5)
        Q[:, :, 2, 2] = s_t_eff * s_t3

        Q[:, :, 3, 0] = -s_t1
        Q[:, :, 3, 1] = c_t_eff * c_t1
        Q[:, :, 4, 1] = -s_t1 * s_t_eff
        Q[:, :, 5, 1] = s_t_eff * s_t1 * (-0.5)
        Q[:, :, 5, 2] = -c_t_eff * s_t3 * s_t1

        Q[:, :, 6, 0] = c_t1
        Q[:, :, 6, 1] = s_t1 * c_t_eff
        Q[:, :, 7, 1] = s_t_eff * c_t1
        Q[:, :, 8, 1] = s_t_eff * c_t1 * 0.5
        Q[:, :, 8, 2] = c_t_eff * s_t3 * c_t1

        J = torch.matmul(Q, x)
        J = J.view(N, 4, 3, 3)

        L = torch.sqrt(c_t3 * 0.08 + 0.08)
        H = torch.zeros((N, 4, 2, 1),
                        dtype=torch.float,
                        device=leg_angles.device)
        S = torch.zeros((N, 4, 3, 2),
                        dtype=torch.float,
                        device=leg_angles.device)
        S[:, :, 0, 1] = -s_t_eff
        S[:, :, 1, 0] = c_t1
        S[:, :, 1, 1] = s_t1 * c_t_eff
        S[:, :, 2, 0] = s_t1
        S[:, :, 2, 1] = -c_t1 * c_t_eff
        H[:, :, 0, 0] = self.l_hip
        H[:, :, 1, 0] = L
        self.foot_positions_in_base_frame = torch.matmul(
            S, H).squeeze(-1) + self.torch_hip_offset  # (N,4,3)
        return J