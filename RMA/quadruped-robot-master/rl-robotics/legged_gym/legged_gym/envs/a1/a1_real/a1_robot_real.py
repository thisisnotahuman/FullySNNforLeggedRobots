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

import os
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import time
from absl import logging
import torch
from legged_gym.envs.base.motor_config import MotorControlMode
from legged_gym.envs.a1.a1_real.robot_interface import RobotInterface
from legged_gym.utils.math import *
from .robot_utils import *


class A1Robot:
    """Interface for real A1 robot."""

    def __init__(self, time_step=0.002, **kwargs):
        """Initializes the robot class."""
        print("Real Robot MPC_BODY_MASS: ", MPC_BODY_MASS)
        print("Real Robot MPC_BODY_INERTIA: ", MPC_BODY_INERTIA)
        print("Real Robot MPC_BODY_HEIGHT: ", MPC_BODY_HEIGHT)
        self.device = kwargs.get('device', torch.device("cpu"))
        # Initialize pd gain vector
        self.motor_kps = np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] *
                                  4)
        self.motor_kds = np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] *
                                  4)
        self._motor_control_mode = kwargs['motor_control_mode']
        self.stand_up_height = STAND_UP_HEIGHT
        self.time_step = time_step

        # Robot state variables
        self._init_complete = False
        self._base_orientation = None
        self._base_rpy = None
        self.raw_state = None
        self._last_raw_state = None
        self._motor_angles = np.zeros(12, dtype=np.float32)
        self._motor_velocities = np.zeros(12, dtype=np.float32)
        self._joint_states = None
        self.drpy = np.zeros(3, dtype=np.float32)
        self.acc = np.zeros(3, dtype=np.float32)
        self.footForce = np.array([50, 50, 50, 50], dtype=np.float32)

        # Initiate UDP for robot state and actions
        self._robot_interface = RobotInterface()
        self._robot_interface.send_command(np.zeros(60, dtype=np.float32))

        self._state_action_counter = 0
        self.estimated_velocity = np.zeros(3)
        self._last_reset_time = time.time()
        self._init_complete = True
        self.clip_angles = np.array(
            [[-0.73, 0.73], [-0.85, 2.8], [-2.6, -1.0]], dtype=np.float32)
        self.unsafty = False
        self.safty_action = np.zeros((12), dtype=np.float32)
        self.tick = 0

    def ReceiveObservation(self):
        """Receives observation from robot.
        """
        state = self._robot_interface.receive_observation()
        self.raw_state = state

        self.tick = state.tick
        for i in range(3):
            self.drpy[i] = state.imu.gyroscope[i]
            self.acc[i] = state.imu.accelerometer[i]
        for i in range(4):
            self.footForce[i] = state.footForce[i]

        q = state.imu.quaternion
        self._base_rpy = np.array([state.imu.rpy], dtype=np.float32)
        # Convert quaternion from wxyz to xyzw
        self._base_orientation = np.array([q[1], q[2], q[3], q[0]])
        self._motor_angles = np.array(
            [motor.q for motor in state.motorState[:12]], dtype=np.float32)
        self._motor_velocities = np.array(
            [motor.dq for motor in state.motorState[:12]], dtype=np.float32)
        self._joint_states = np.array(
            list(zip(self._motor_angles, self._motor_velocities)))

    def GetTrueMotorAngles(self):
        return self._motor_angles.copy()

    def GetMotorAngles(self):
        return self._motor_angles.copy()

    def GetMotorVelocities(self):
        return self._motor_velocities.copy()

    def GetBaseRollPitchYaw(self):
        return getEulerFromQuaternion(self._base_orientation)

    def GetTrueBaseRollPitchYaw(self):
        return getEulerFromQuaternion(self._base_orientation)

    def GetBaseRollPitchYawRate(self):
        return self.GetTrueBaseRollPitchYawRate()

    def GetTrueBaseRollPitchYawRate(self):
        return self.drpy.copy()

    def GetBaseVelocity(self):
        return self.estimated_velocity.copy()

    def GetFootContacts(self):
        return self.footForce > 20

    def GetFootForce(self):
        return self.footForce.copy()

    def GetTimeSinceReset(self):
        return time.time() - self._last_reset_time

    def GetBaseOrientation(self):
        return self._base_orientation.copy()

    def GetTrueBaseOrientation(self):
        return self._base_orientation.copy()

    @property
    def motor_velocities(self):
        return self._motor_velocities.copy()

    def ApplyAction(self, motor_commands, motor_control_mode=None):
        """Clips and then apply the motor commands using the motor model.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).
          motor_control_mode: A MotorControlMode enum.
        """
        if motor_control_mode is None:
            motor_control_mode = self._motor_control_mode

        command = np.zeros(60, dtype=np.float32)
        if motor_control_mode == MotorControlMode.POSITION:
            motor_commands = np.clip(motor_commands.reshape(4, 3),
                                     self.clip_angles[:, 0],
                                     self.clip_angles[:, 1]).reshape(12)
            for motor_id in range(NUM_MOTORS):
                command[motor_id * 5] = motor_commands[motor_id]
                command[motor_id * 5 + 1] = self.motor_kps[motor_id]
                command[motor_id * 5 + 3] = self.motor_kds[motor_id]
        elif motor_control_mode == MotorControlMode.TORQUE:
            for motor_id in range(NUM_MOTORS):
                motor_commands = np.clip(motor_commands, -25, 25)
                command[motor_id * 5 + 4] = motor_commands[motor_id]
        elif motor_control_mode == MotorControlMode.HYBRID:
            command = np.array(motor_commands, dtype=np.float32)
        else:
            raise ValueError(
                'Unknown motor control mode for A1 robot: {}.'.format(
                    motor_control_mode))

        self._robot_interface.send_command(command)

    def Reset(self, default_motor_angles=None, reset_time=3.0):
        """Reset the robot to default motor angles."""
        self._reset()
        self._state_action_counter = 0

    def Step(self, action, motor_control_mode=None):
        if not self.unsafty:
            self.ApplyAction(action, motor_control_mode)
        self.ReceiveObservation()
        self._state_action_counter += 1

    def check_safty(self):
        reshaped_motor_angles = self._motor_angles.reshape(4, 3)
        for leg_id in range(4):
            if reshaped_motor_angles[leg_id, 1] < self.clip_angles[1, 0]:
                self.safty_action[leg_id * 3 + 1] = 1.0 / (
                    (self.clip_angles[1, 0] - reshaped_motor_angles[leg_id, 1])
                    + 0.1)
                self.unsafty = True
            elif reshaped_motor_angles[leg_id, 1] > self.clip_angles[1, 1]:
                self.safty_action[leg_id * 3 + 1] = 1.0 / (
                    (self.clip_angles[1, 1] - reshaped_motor_angles[leg_id, 1])
                    + 0.1)
                self.unsafty = True

        return self.unsafty

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                                foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.

        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        # assert len(self._foot_link_ids) == self.num_legs
        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = list(
            range(leg_id * motors_per_leg,
                  leg_id * motors_per_leg + motors_per_leg))

        joint_angles = foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - HIP_OFFSETS[leg_id],
            l_hip_sign=(-1)**(leg_id + 1))

        # Joint offset is necessary for Laikago.
        joint_angles = np.multiply(
            np.asarray(joint_angles) -
            np.asarray(self._motor_offset)[joint_position_idxs],
            self._motor_direction[joint_position_idxs])

        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        motor_angles = self.GetMotorAngles()
        return foot_positions_in_base_frame(motor_angles)

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles, leg_id)

    def stand_up(self,
                 standup_time=1.5,
                 reset_time=5,
                 default_motor_angles=None):
        logging.warning(
            "About to reset the robot, make sure the robot is hang-up.")

        if default_motor_angles is None:
            default_motor_angles = INIT_MOTOR_ANGLES

        for _ in range(20):
            self.ReceiveObservation()

        current_motor_angles = self.GetMotorAngles()
        tik = time.time()
        count = 0
        for t in np.arange(0, reset_time, self.time_step):
            count += 1
            stand_up_last_t = time.time()
            blend_ratio = min(t / standup_time, 1)
            action = blend_ratio * default_motor_angles + (
                1 - blend_ratio) * current_motor_angles
            self.Step(action, MotorControlMode.POSITION)
            while time.time() - stand_up_last_t < self.time_step:
                pass
        tok = time.time()
        print("stand up cost time =  ", -tik + tok)

    def sit_down(self,
                 sitdown_time=1.5,
                 reset_time=2,
                 default_motor_angles=None):
        if default_motor_angles is None:
            default_motor_angles = SITDOWN_MOTOR_ANGLES

        self.ReceiveObservation()
        current_motor_angles = self.GetMotorAngles()
        sitdown_time = max(sitdown_time, 1.5)
        tik = time.time()
        count = 0
        for t in np.arange(0, reset_time, self.time_step):
            count += 1
            sitdown_last_t = time.time()
            blend_ratio = min(t / sitdown_time, 1)
            action = blend_ratio * default_motor_angles + (
                1 - blend_ratio) * current_motor_angles
            self.Step(action, MotorControlMode.POSITION)
            while time.time() - sitdown_last_t < self.time_step:
                pass
        tok = time.time()
        print("sit down cost time =  ", -tik + tok)

    def _reset(self):
        for _ in range(5):
            self.ReceiveObservation()
        roll = self._base_rpy[0][0]
        # make sure the angle is located in range [0, 2*pi]
        if roll < 0.:
            roll += 2 * PI
        print("roll = ", roll)
        shrinked_motor_angles = np.array(
            [0, 2.0, -2.6, 0, 2.0, -2.6, 0, 2.0, -2.6, 0, 2.0, -2.6],
            dtype=np.float32)

        if roll < 0.5 * PI or roll > 1.5 * PI:  # the robot is not fliped
            t1 = 2.0  # time for shrink legs
            t2 = 2.0  # time for stand up
            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(t1, current_motor_angles,
                                           shrinked_motor_angles)
            self.stand_up(t2)
        else:
            t1 = 2.0  # time for shrink1 legs
            t2 = 1.0  # time for stand up

            shrinked1_motor_angles = np.array(
                [0, 1.57, -0.9, 0, 1.57, -0.9, 0, 1.57, -0.9, 0, 1.57, -0.9],
                dtype=np.float32)
            shrinked2_motor_angles = np.array(
                [0, 1.57, -2.5, 0, 1.57, -2.5, 0, 1.57, -2.5, 0, 1.57, -2.5],
                dtype=np.float32)

            shrinked3_motor_angles = np.array([
                -0.7, 2.7, -2.5, 0, 1.57, -2.5, -0.7, 2.8, -2.5, 0, 1.57, -2.5
            ],
                                              dtype=np.float32)

            shrinked4_motor_angles = np.array([
                -0.7, 2.7, -2.5, 0.6, 2.5, -2.5, -0.8, 2.8, -2.5, 0.6, 2.5,
                -2.5
            ],
                                              dtype=np.float32)

            shrinked5_motor_angles = np.array([
                -0.1, 1.7, -2.5, -0.6, 2.5, -2.5, -0.1, 1.7, -2.5, -0.6, 2.5,
                -2.5
            ],
                                              dtype=np.float32)

            shrinked6_motor_angles = np.array(
                [0, 0.9, -2.5, 0, 0.9, -2.5, 0, 0.9, -2.5, 0, 0.9, -2.5],
                dtype=np.float32)

            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(t1, current_motor_angles,
                                           shrinked1_motor_angles)

            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(t2, current_motor_angles,
                                           shrinked2_motor_angles)

            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(0.5, shrinked2_motor_angles,
                                           shrinked3_motor_angles)

            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(0.3, shrinked3_motor_angles,
                                           shrinked4_motor_angles)

            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(0.5, current_motor_angles,
                                           shrinked5_motor_angles, 0.5)

            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(t1, current_motor_angles,
                                           shrinked_motor_angles, 0.5)
            self.stand_up(t2)

    def linearly_transform_angles(self, t1, souce_angles, dest_angles, t2=0):
        for t in np.arange(0, t1, self.time_step):
            stand_up_last_t = time.time()
            blend_ratio = min(t / t1, 1.0)
            action = blend_ratio * dest_angles + (1 -
                                                  blend_ratio) * souce_angles
            self.Step(action, MotorControlMode.POSITION)

            while time.time() - stand_up_last_t < self.time_step:
                pass

        for t in np.arange(0, t2, self.time_step):
            stand_up_last_t = time.time()
            while time.time() - stand_up_last_t < self.time_step:
                pass

    def GetRobotObs(self):
        base_acc_in_base_frame = torch.from_numpy(self.acc).unsqueeze(0).to(
            device=self.device, dtype=torch.float)
        is_contact = self.GetFootContacts()
        obs = torch.cat(
            (base_acc_in_base_frame, torch.from_numpy(self._base_rpy).to(
                self.device), torch.from_numpy(
                    self.GetBaseRollPitchYawRate()).unsqueeze(0).to(
                        self.device),
             torch.from_numpy(self._serialize_dof(
                 self._motor_angles)).unsqueeze(0).to(self.device),
             torch.from_numpy(self._serialize_dof(
                 self._motor_velocities)).unsqueeze(0).to(self.device),
             torch.tensor(np.array([[is_contact[i] for i in [1, 0, 3, 2]]]),
                          dtype=torch.float,
                          device=self.device)),
            dim=-1)  # 3+3+3+12+12+4=37
        return obs

    def _serialize_dof(self, dof_data):
        serialized_data = np.zeros(12, dtype=np.float32)
        serialized_data[0:3] = dof_data[3:6]
        serialized_data[3:6] = dof_data[0:3]
        serialized_data[6:9] = dof_data[9:12]
        serialized_data[9:12] = dof_data[6:9]
        return serialized_data