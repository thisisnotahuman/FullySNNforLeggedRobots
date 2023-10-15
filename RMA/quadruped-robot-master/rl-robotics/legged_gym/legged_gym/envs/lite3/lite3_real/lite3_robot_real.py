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
import copy
import torch
from legged_gym.envs.base.motor_config import MotorControlMode
from .lite3_interface import Lite3Interface
from legged_gym.utils.math import *
from .robot_utils import *

np.set_printoptions(precision=3, suppress=True)


class Lite3Robot:
    """Interface for real Lite3 robot."""

    def __init__(self, time_step=0.002, **kwargs):
        """Initializes the robot class."""
        print("--------  init robot begin! -------")
        print("Real Robot MPC_BODY_MASS: ", MPC_BODY_MASS)
        print("Real Robot MPC_BODY_INERTIA: ", MPC_BODY_INERTIA)
        print("Real Robot MPC_BODY_HEIGHT: ", MPC_BODY_HEIGHT)
        self.device = kwargs.get('device', torch.device("cpu"))
        # Initialize pd gain vector
        self.motor_kps = np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] *
                                  4)
        self.motor_kds = np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] *
                                  4)
        self._motor_control_mode = kwargs['motor_control_mode']  # HYBRID
        self.stand_up_height = STAND_UP_HEIGHT
        self.time_step = time_step
        print("Real Robot time_step = ", self.time_step)
        # Robot state variables
        self._init_complete = False
        self._base_orientation = None
        self._base_rpy = None
        self.raw_state = None
        self._last_raw_state = None
        self._motor_angles = np.zeros(12, dtype=np.float32)
        self._motor_velocities = np.zeros(12, dtype=np.float32)
        self._motor_torque = np.zeros(12, dtype=np.float32)

        self._joint_states = None
        self.footForce = np.array([50, 50, 50, 50], dtype=np.float32)
        self.drpy = np.zeros(3, dtype=np.float32)
        self.acc = np.zeros(3, dtype=np.float32)

        # Initiate UDP for robot state and actions
        self._robot_interface = Lite3Interface()
        state = self._robot_interface.receive_observation()
        self.yaw_offset = state.imu.angle_yaw  # in degree unit

        self._state_action_counter = 0
        self.estimated_velocity = np.zeros(3, dtype=np.float32)
        self._last_reset_time = time.time()
        self._init_complete = True
        self.clip_angles = np.array(
            [[-0.523, 0.523], [-0.314, 3.6], [-2.792, -0.524]],
            dtype=np.float32)
        self.unsafty = False
        self.safty_action = np.zeros((12), dtype=np.float32)
        self.tick = 0
        print("--------  init robot end! -------")

    def ReceiveObservation(self):
        """Receives observation from robot.
        """
        state = self._robot_interface.receive_observation()
        self.tick = state.tick
        self._base_rpy = np.array([
            state.imu.angle_roll, state.imu.angle_pitch,
            state.imu.angle_yaw - self.yaw_offset
        ],
                                  dtype=np.float32) / 57.3
        # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
        wxyz = getQuaternionFromEuler(self._base_rpy)
        self._base_orientation = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]],
                                          dtype=np.float32)  # -->xyzw

        for i in range(3):
            self._motor_angles[i] = state.motorState.fr_leg[i].q
            self._motor_angles[i + 3] = state.motorState.fl_leg[i].q
            self._motor_angles[i + 6] = state.motorState.hr_leg[i].q
            self._motor_angles[i + 9] = state.motorState.hl_leg[i].q

            self._motor_velocities[i] = state.motorState.fr_leg[i].dq
            self._motor_velocities[i + 3] = state.motorState.fl_leg[i].dq
            self._motor_velocities[i + 6] = state.motorState.hr_leg[i].dq
            self._motor_velocities[i + 9] = state.motorState.hl_leg[i].dq

            self._motor_torque[i] = state.motorState.fr_leg[i].tauEst
            self._motor_torque[i + 3] = state.motorState.fl_leg[i].tauEst
            self._motor_torque[i + 6] = state.motorState.hr_leg[i].tauEst
            self._motor_torque[i + 9] = state.motorState.hl_leg[i].tauEst

        self._motor_angles = -self._motor_angles
        self._motor_velocities = -self._motor_velocities
        self._motor_torque = -self._motor_torque

        self._joint_states = np.array(
            list(zip(self._motor_angles, self._motor_velocities)))
        self.footForce[0] = state.fr_tor[2]
        self.footForce[1] = state.fl_tor[2]
        self.footForce[2] = state.hr_tor[2]
        self.footForce[3] = state.hl_tor[2]
        self.drpy[0] = state.imu.angular_velocity_roll
        self.drpy[1] = state.imu.angular_velocity_pitch
        self.drpy[2] = state.imu.angular_velocity_yaw
        self.acc[0] = state.imu.acc_x
        self.acc[1] = state.imu.acc_y
        self.acc[2] = state.imu.acc_z

    def GetTrueMotorAngles(self):
        return self._motor_angles.copy()

    def GetMotorAngles(self):
        return self._motor_angles.copy()

    def GetMotorVelocities(self):
        return self._motor_velocities.copy()

    def GetBasePosition(self):
        return -1

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
        return self.footForce > 25

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
                motor_id_ = motor_id + 3 if (motor_id //
                                             3) % 2 == 0 else motor_id - 3
                command[motor_id_ * 5] = -motor_commands[motor_id]
                command[motor_id_ * 5 + 1] = self.motor_kps[motor_id]
                command[motor_id_ * 5 + 3] = self.motor_kds[motor_id]
        elif motor_control_mode == MotorControlMode.TORQUE:
            for motor_id in range(NUM_MOTORS):
                motor_id_ = motor_id + 3 if (motor_id //
                                             3) % 2 == 0 else motor_id - 3
                motor_commands = np.clip(motor_commands, -25, 25)
                command[motor_id_ * 5 + 4] = -motor_commands[motor_id]
        elif motor_control_mode == MotorControlMode.HYBRID:
            for motor_id in range(NUM_MOTORS):
                motor_id_ = motor_id + 3 if (motor_id //
                                             3) % 2 == 0 else motor_id - 3
                command[motor_id_ * 5] = -motor_commands[motor_id * 5]
                command[motor_id_ * 5 + 1] = motor_commands[motor_id * 5 + 1]
                command[motor_id_ * 5 + 2] = -motor_commands[motor_id * 5 + 2]
                command[motor_id_ * 5 + 3] = motor_commands[motor_id * 5 + 3]
                command[motor_id_ * 5 + 4] = -motor_commands[motor_id * 5 + 4]
        else:
            raise ValueError(
                'Unknown motor control mode for A1 robot: {}.'.format(
                    motor_control_mode))

        self._robot_interface.send_command(command)

    def Reset(self, default_motor_angles=None, reset_time=3.0):
        """Reset the robot to default motor angles."""
        self._reset()
        self._state_action_counter = 0
        print("robot reset end -------")

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

        # Return the joing index (the same as when calling GetMotorAngles) as well as the angles.
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
        print("count:", count)

    def _reset(self):
        shrinked_motor_angles = np.array(
            [0, 2.0, -2.6, 0, 2.0, -2.6, 0, 2.0, -2.6, 0, 2.0, -2.6],
            dtype=np.float32)

        self._robot_interface.initialize()  # todo
        count = 0
        while True:
            self.ReceiveObservation()
            count += 1
            print("count = {}, rpy = {}".format(count, self._base_rpy))
            if abs(self._base_rpy[1]) > 1e-5:
                break
            time.sleep(0.001)

        roll = self._base_rpy[0]
        # make sure the angle is located in range [0, 2*pi]
        if roll < 0.:
            roll += 2 * PI

        if roll < 0.5 * PI or roll > 1.5 * PI:  # the robot is not fliped
            t1 = 2.0  # time for shrink legs
            t2 = 2.0  # time for stand up
            current_motor_angles = self.GetMotorAngles()
            self.linearly_transform_angles(t1, current_motor_angles,
                                           shrinked_motor_angles)
            self.stand_up(t2)
        else:
            raise Exception("387")
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
        base_acc_in_base_frame = torch.tensor([self.acc],
                                              device=self.device,
                                              dtype=torch.float)
        is_contact = self.GetFootContacts()
        obs = torch.cat(
            (base_acc_in_base_frame, torch.from_numpy(
                self._base_rpy).unsqueeze(0), to(self.device),
             torch.from_numpy(self.GetBaseRollPitchYawRate()).unsqueeze(0).to(
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


if __name__ == '__main__':
    robot = Lite3Robot(0.003)
    robot.Reset()
