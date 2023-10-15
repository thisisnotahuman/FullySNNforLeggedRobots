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
"""Estimates base velocity for A1 robot from accelerometer readings."""
import numpy as np
import transformations
from filterpy.kalman import KalmanFilter
from .moving_window_filter import MovingWindowFilter
from legged_gym.utils.math import getMatrixFromQuaternion, invertTransform, multiplyTransforms


class VelocityEstimator:
    """Estimates base velocity of robot.

    The velocity estimator consists of 2 parts:
    1) A state estimator for CoM velocity.

    Two sources of information are used:
    The integrated reading of accelerometer and the velocity estimation from
    contact legs. The readings are fused together using a Kalman Filter.

    2) A moving average filter to smooth out velocity readings
    """

    def __init__(self,
                 robot,
                 accelerometer_variance=0.1,
                 sensor_variance=0.1,
                 initial_variance=0.1,
                 moving_window_filter_size=120):
        """Initiates the velocity estimator.

        See filterpy documentation in the link below for more details.
        https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

        Args:
          robot: the robot class for velocity estimation.
          accelerometer_variance: noise estimation for accelerometer reading.
          sensor_variance: noise estimation for motor velocity reading.
          initial_covariance: covariance estimation of initial state.
        """
        self.robot = robot

        self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
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

    def _compute_delta_time(self, robot_state_tick):
        if self._last_timestamp == 0.:
            # First timestamp received, return an estimated delta_time.
            delta_time_s = self.robot.time_step
        else:
            delta_time_s = (robot_state_tick - self._last_timestamp) / 1000.
        self._last_timestamp = robot_state_tick
        return delta_time_s

    def update(self, robot_state_tick, current_time):
        """Propagate current state estimate with new accelerometer reading."""
        delta_time_s = self._compute_delta_time(robot_state_tick)
        sensor_acc = self.robot.acc
        base_orientation = self.robot.GetBaseOrientation()
        rot_mat = getMatrixFromQuaternion(base_orientation)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        calibrated_acc = rot_mat.dot(sensor_acc) + np.array([0., 0., -9.8])
        self.filter.predict(u=calibrated_acc * delta_time_s)

        # Correct estimation using contact legs
        observed_velocities = []
        foot_contact = self.robot.GetFootContacts()
        for leg_id in range(4):
            if foot_contact[leg_id]:
                jacobian = self.robot.ComputeJacobian(leg_id)
                # Only pick the jacobian related to joint motors
                joint_velocities = self.robot.motor_velocities[leg_id *
                                                               3:(leg_id + 1) *
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
        self.robot.estimated_velocity = self._estimated_velocity

        self._com_velocity_world_frame = self._estimated_velocity
        _, inverse_rotation = invertTransform((0, 0, 0), base_orientation)
        self._com_velocity_body_frame, _ = multiplyTransforms(
            (0, 0, 0), inverse_rotation, self._com_velocity_world_frame,
            (0, 0, 0, 1))

    def estimate_robot_height(self):
        contacts = self.robot.GetFootContacts()
        if np.sum(contacts) == 0:
            # All foot in air, no way to estimate
            return self.robot.stand_up_height
        else:
            base_orientation = self.robot.GetBaseOrientation()
            rot_mat = getMatrixFromQuaternion(base_orientation)
            rot_mat = np.array(rot_mat).reshape((3, 3))

            foot_positions = self.robot.GetFootPositionsInBaseFrame()
            foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T
            # pylint: disable=unsubscriptable-object
            useful_heights = contacts * (-foot_positions_world_frame[:, 2])
            return np.sum(useful_heights) / np.sum(contacts)

    @property
    def estimated_velocity(self):
        return self._estimated_velocity.copy()

    @property
    def com_velocity_body_frame(self):
        """The base velocity projected in the body aligned inertial frame.
        The body aligned frame is a intertia frame that coincides with the body
        frame, but has a zero relative velocity/angular velocity to the world frame.

        Returns:
          The com velocity in body aligned frame.
        """
        return self._com_velocity_body_frame

    @property
    def com_velocity_world_frame(self):
        return self._com_velocity_world_frame
