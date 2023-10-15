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
"""Estimates base velocity for robot from accelerometer readings."""
import os
import sys
pp = os.path.join(os.path.dirname(__file__), "../../../../")
sys.path.insert(0, pp + 'isaacgym/python')
sys.path.insert(0, pp + 'third_party/torchcast')

import numpy as np
from typing import Callable, Optional, Dict
import collections
import legged_gym.utils.isaacgym_utils as utils
import torch
from torch import Tensor, nn
from torchcast.process.utils import SingleOutput
from torchcast.process.base import Process
from torchcast.process import LocalTrend, QuadrupedDynamics, LinearModel, LocalLevel
from torchcast.kalman_filter import KalmanFilter
from torchcast.internals.utils import get_nan_groups
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter
from .moving_window_filter import MovingWindowFilter
from legged_gym.utils.math import multiplyTransforms, invertTransform, getEulerFromQuaternion, getMatrixFromQuaternion


class TorchMovingWindowFilter(nn.Module):
    """A stable O(1) moving filter for incoming data streams.
    We implement the Neumaier's algorithm to calculate the moving window average,
    which is numerically stable.
    """
    def __init__(self,
                 window_size: int,
                 state_dim: int = 1,
                 device=torch.device('cpu'),
                 num_envs: int = 1):
        """Initializes the class.
        Args:
          window_size: The moving window size.
        """
        super(TorchMovingWindowFilter, self).__init__()
        assert window_size > 0
        self._window_size = window_size
        self.num_envs = num_envs
        self._value_deque = collections.deque(maxlen=window_size)
        # The moving window sum.
        self._sum = torch.zeros((self.num_envs, 1, state_dim),
                                device=device,
                                dtype=torch.float32)
        # The correction term to compensate numerical precision loss during calculation.
        self._correction = torch.zeros((self.num_envs, 1, state_dim),
                                       device=device,
                                       dtype=torch.float32)

    def _neumaier_sum(self, value: Optional[Tensor]):
        """Update the moving window sum using Neumaier's algorithm.
        For more details please refer to:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements
        Args:
          value: The new value to be added to the window. Tensor shape = [num_groups, n_step=1, state_dim]
        """
        new_sum = self._sum + value
        # If self._sum is bigger, low-order digits of value are lost.
        # low-order digits of sum are lost
        temp = torch.where(
            torch.abs(self._sum) >= torch.abs(value),
            (self._sum - new_sum) + value, (value - new_sum) + self._sum)
        self._correction += temp
        self._sum = new_sum

    # @torch.jit.ignore
    def calculate_average(self, new_value: Optional[Tensor]) -> Tensor:
        """Computes the moving window average in O(1) time.
            Args:
            new_value: The new value to enter the moving window.
            Returns:
            The average of the values in the window.
        """
        deque_len = len(self._value_deque)
        if deque_len < self._value_deque.maxlen:
            pass
        else:
            # The left most value to be subtracted from the moving sum.
            self._neumaier_sum(-self._value_deque[0])

        self._neumaier_sum(new_value)
        self._value_deque.append(new_value)

        return (self._sum + self._correction) / self._window_size

    def forward(self, new_value):
        N = new_value.size(0)
        if N != self.num_envs and self.num_envs == 1:
            self._correction = self._correction.repeat(N, 1, 1)
            self._sum = self._sum.repeat(N, 1, 1)
            self.num_envs = N

        value = self.calculate_average(new_value)
        return value


class VelocityEstimator(nn.Module):

    def __init__(
        self,
        robotIn,
        accelerometer_variance=0.1,
        sensor_variance=0.1,
        initial_variance=0.1,
        window_size=120,
        device=torch.device("cpu")):
        super(VelocityEstimator, self).__init__()
        self.robot = robotIn
        self.device = device
        self._initial_variance = initial_variance
        self.filter = filterpy_KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        self.torch_kf = KalmanFilter(processes=[
            QuadrupedDynamics(id='lt',
                              state_elements=["vx", "vy", "vz"],
                              measure=['vx', 'vy', 'vz'],
                              decay_velocity=None,
                              velocity_multi=1.)
        ],
                                     measures=['vx', 'vy', 'vz'],
                                     device=self.device)
        x = torch.tensor([0., 0., 0.], device=self.device).unsqueeze(0)
        P = torch.eye(3, device=self.device) * 0.1
        P = P.unsqueeze(0)

        x_, P_ = self.torch_kf._prepare_initial_state((x, P))
        data = torch.tensor([[[0., 0., 0.]]])
        self.data = data
        kwargs_per_process = self.torch_kf._parse_design_kwargs(
            input=data, out_timesteps=1)

        predict_kwargs, update_kwargs = self.torch_kf._build_design_mats(
            kwargs_per_process, num_groups=1, out_timesteps=1)
        F = predict_kwargs['F'][0][0]
        Q = predict_kwargs['Q'][0][0]
        H = update_kwargs['H'][0][0]
        R = update_kwargs['R'][0][0]
        self.predit_out = self.torch_kf._generate_predictions(
            ([torch.zeros(1, 3, device=self.device)
              ], [torch.eye(3, device=self.device) * 0.1]), [], [],
            ([torch.zeros(1, 3, device=self.device)
              ], [0.1 * torch.eye(3, device=self.device).unsqueeze(0)]))
        self._window_size = window_size
        self.velocityFilter = TorchMovingWindowFilter(window_size, 3,
                                                      self.device)
        self.com_velocity_world_frame = torch.zeros((3, ),
                                                    dtype=torch.float32,
                                                    device=self.device)
        self.com_velocity_body_frame = torch.zeros((3, ),
                                                   dtype=torch.float32,
                                                   device=self.device)
        self._estimated_angular_velocity = torch.zeros((3, ),
                                                       dtype=torch.float32,
                                                       device=self.device)
        self._last_timestamp = 0

    def get_estimated_velocity(self):
        """expressed in base frame"""
        return self._estimated_velocity

    def get_estimated_angular_velocity(self):
        return self._estimated_angular_velocity

    def reset(self, currentTime):
        self._last_timestamp = 0

    def _compute_delta_time(self, robot_state):
        delta_time = 0
        if abs(self._last_timestamp) < 1e-5:
            delta_time = self.robot.time_step  # 0.003
        else:
            delta_time = (robot_state.tick - self._last_timestamp) / 1000.

        self._last_timestamp = robot_state.tick
        return delta_time

    def compute_delta_time(self, currentTime):
        delta_time = 0
        if abs(self._last_timestamp) < 1e-5:
            delta_time = self.robot.time_step  # 0.003
        else:
            delta_time = currentTime - self._last_timestamp

        self._last_timestamp = currentTime
        return delta_time

    def _update(self, currentTime, obs):
        rpyRate = obs[:, 6:9]
        delta_time = self.compute_delta_time(currentTime)
        sensor_acc = self.robot.GetBaseAcc()

        rpy = self.robot.GetBaseRPY()
        R = utils.euler_xyz_to_rotation_matrix(rpy)
        calibratedAcc = torch.matmul(R, sensor_acc.unsqueeze(-1)).squeeze(
            -1)  # base->world
        calibratedAcc[:, 2] -= 9.81
        deltaV = calibratedAcc * delta_time
        # Correct estimation using contact legs
        footContact = self.robot.GetFootContacts().to(torch.bool)  #
        numContact = torch.sum(footContact,
                               axis=1,
                               keepdim=True,
                               dtype=torch.float32)
        footVInBaseFrame = self.robot.GetFootVelocitiesInBaseFrame()  # (N,4,3)
        footVInBaseFrame = footVInBaseFrame.transpose(1, 2)  # (N, 3, 4)
        footPInBaseFrame = self.robot.GetFootPositionsInBaseFrame(
        )  # (n, 4 ,3)
        skewR = utils.vector_2_skewmat(rpyRate)

        vB = torch.where(
            footContact.unsqueeze(1).repeat(1, 3, 1), footVInBaseFrame +
            torch.matmul(skewR, footPInBaseFrame.transpose(1, 2)),
            torch.zeros_like(footVInBaseFrame))  # (N, 3, 4)

        observed_velocities = torch.where(
            numContact > 0.5,  # (N,1)
            torch.sum(-torch.matmul(R, vB), axis=2) / numContact,
            self.com_velocity_world_frame)
        weight = torch.where(torch.any(footContact, dim=1, keepdim=True), 1, 0)
        mean_velocity = self.com_velocity_world_frame * (
            1 - weight) + weight * observed_velocities  # (N, 3)
        self.predit_out = self.torch_kf(
            mean_velocity.unsqueeze(1),
            n_step=1,
            start_offsets=None,
            include_updates_in_output=True,
            out_timesteps=1,
            initial_state=(self.predit_out.update_means[0] + deltaV,
                           self.predit_out.update_covs[0]))
        v = self.velocityFilter(self.predit_out.update_means)  # (N, 1, 3)
        self.com_velocity_world_frame = v.squeeze(1)  # world frame
        self.com_velocity_body_frame = torch.matmul(R, v.transpose(
            1, 2)).squeeze(-1)  # (N,1,1,3) @ (N,3,3)--> (1,1,1,3)
        # estimate robot base height
        Pzs = torch.matmul(R, footPInBaseFrame.transpose(1, 2)).select(dim=1,
                                                                       index=2)
        Pz = torch.where(footContact, Pzs, torch.zeros_like(Pzs))
        Pz = torch.where(numContact > 0.5,
                         Pz.sum(axis=1, keepdim=True) / numContact,
                         torch.zeros_like(numContact))
        h = (-weight) * Pz + (1 - weight) * self.robot.stand_up_height

        return self.com_velocity_body_frame, h

    def forward(self, currentTime, obs):
        """
        Input:
            currentTime: float scalar
            obs : Tensor[float32], shape = (1,320)
        """
        v, h = self._update(currentTime, obs)
        return v, h


class GoogleVelocityEstimator:

    def __init__(self,
                 robot,
                 accelerometer_variance=0.1,
                 sensor_variance=0.1,
                 initial_variance=0.1,
                 moving_window_filter_size=120):
        self.robot = robot

        self.filter = filterpy_KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        self.filter.x = np.zeros(3)
        self._initial_variance = initial_variance
        self.filter.P = np.eye(3) * self._initial_variance  # State covariance
        self.filter.Q = np.eye(3) * accelerometer_variance
        self.filter.R = np.eye(3) * sensor_variance

        self.filter.H = np.eye(3)  # measurement function (y=H*x)
        self.filter.F = np.eye(3)  # state transition matrix
        self.filter.B = np.eye(3)

        self._window_size = moving_window_filter_size
        self.moving_window_filter_x = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_y = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_z = MovingWindowFilter(
            window_size=self._window_size)
        self._estimated_velocity = np.zeros(3)
        self._last_timestamp = 0
        self._com_velocity_world_frame = np.array((0, 0, 0))
        self._com_velocity_body_frame = np.array((0, 0, 0))

    def reset(self):
        self.filter.x = np.zeros(3)
        self.filter.P = np.eye(3) * self._initial_variance
        self.moving_window_filter_x = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_y = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_z = MovingWindowFilter(
            window_size=self._window_size)
        self._last_timestamp = 0
        self._com_velocity_world_frame = np.array((0, 0, 0))
        self._com_velocity_body_frame = np.array((0, 0, 0))

    def _compute_delta_time(self, t):
        if self._last_timestamp == 0.:
            # First timestamp received, return an estimated delta_time.
            delta_time_s = self.robot.time_step
        else:
            delta_time_s = t - self._last_timestamp
        self._last_timestamp = t
        return delta_time_s

    def update(self, obs, current_time):
        """Propagate current state estimate with new accelerometer reading."""
        delta_time_s = self._compute_delta_time(current_time)
        sensor_acc = self.robot.GetBaseAcc()
        sensor_acc = sensor_acc.numpy().flatten()
        base_orientation = self.robot.GetBaseOrientation().flatten()
        rot_mat = getMatrixFromQuaternion(base_orientation)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        calibrated_acc = rot_mat.dot(sensor_acc) + np.array([0., 0., -9.8])
        self.filter.predict(u=calibrated_acc * delta_time_s)

        # Correct estimation using contact legs
        observed_velocities = []
        foot_contact = self.robot.GetFootContacts()
        foot_contact = foot_contact.numpy().flatten()
        motor_velocities = self.robot.GetMotorVelocities()
        motor_velocities = motor_velocities.numpy().flatten()
        for leg_id in range(4):
            if foot_contact[leg_id]:
                jacobian = self.robot.ComputeJacobian(leg_id)
                # Only pick the jacobian related to joint motors
                joint_velocities = motor_velocities[leg_id * 3:(leg_id + 1) *
                                                    3]
                leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
                base_velocity_in_base_frame = -leg_velocity_in_base_frame[:3]
                observed_velocities.append(
                    rot_mat.dot(base_velocity_in_base_frame))

        if observed_velocities:
            observed_velocities = np.mean(observed_velocities, axis=0)
            self.filter.update(observed_velocities)

        vel_x = self.moving_window_filter_x.calculate_average(self.filter.x[0])
        vel_y = self.moving_window_filter_y.calculate_average(self.filter.x[1])
        vel_z = self.moving_window_filter_z.calculate_average(self.filter.x[2])
        self._estimated_velocity = np.array([vel_x, vel_y, vel_z])

        self._com_velocity_world_frame = self._estimated_velocity
        _, inverse_rotation = invertTransform((0, 0, 0), base_orientation)
        self._com_velocity_body_frame, _ = multiplyTransforms(
            (0, 0, 0), inverse_rotation, self._com_velocity_world_frame,
            (0, 0, 0, 1))

    def estimate_robot_height(self):
        contacts = self.robot.GetFootContacts()
        if np.sum(contacts) == 0:
            return self.robot.stand_up_height
        else:
            base_orientation = self.robot.GetBaseOrientation()
            rot_mat = self.robot.getMatrixFromQuaternion(base_orientation)
            rot_mat = np.array(rot_mat).reshape((3, 3))

            foot_positions = self.robot.GetFootPositionsInBaseFrame()
            foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T
            useful_heights = contacts * (-foot_positions_world_frame[:, 2])
            return np.sum(useful_heights) / np.sum(contacts)

    @property
    def estimated_velocity(self):
        return self._estimated_velocity.copy()

    @property
    def com_velocity_body_frame(self):
        return self._com_velocity_body_frame

    @property
    def com_velocity_world_frame(self):
        return self._com_velocity_world_frame
