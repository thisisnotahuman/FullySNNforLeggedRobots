# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2023, HUAWEI TECHNOLOGIES

import os
import sys
import time
import numpy as np
import torch

from legged_gym.utils.isaacgym_utils import (to_torch, quat_mul, quat_apply,
                                             torch_rand_float, get_axis_params,
                                             quat_rotate_inverse,
                                             get_euler_xyz)
from legged_gym.utils.torch_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict, parse_device_str
from legged_gym.envs.a1.a1_real.robot_utils import STAND_UP_HEIGHT
from legged_gym.envs.gamepad import gamepad_reader
from legged_gym.envs.lite3.lite3_real import lite3_robot_real
from legged_gym.envs.lite3.lite3_real.lite3_robot_dummy import Lite3Dummy
from legged_gym.envs.base.motor_config import MotorControlMode
from legged_gym.envs.estimator import robot_velocity_estimator
from legged_gym.envs.estimator.torch_robot_velocity_estimator import VelocityEstimator as torch_velocity_estimator
from legged_gym.envs.base.PMTrajectoryGenerator import PMTrajectoryGenerator


class Lite3LeggedRobot:
    """Lite3Robot environment class for the real-world Lite3 robot.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 **kwargs):
        """ Parses the provided config file, initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.base_task_init(cfg, sim_params, physics_engine, sim_device,
                            headless)

        self._init_buffers()

        self.hybrid_action = np.zeros(60, dtype=np.float32)
        self.motor_control_mode = "P"
        self.last_ctontrol_time = 0

        self.gamepad = gamepad_reader.Gamepad()
        self.command_function = self.gamepad.get_command

        # Construct real robot
        self.robot = lite3_robot_real.Lite3Robot(
            motor_control_mode=MotorControlMode.HYBRID,
            enable_action_interpolation=False,
            time_step=self.cfg.sim.dt,
            action_repeat=1,
            device=self.device)

        self.robot_dummy = None
        if cfg.control.use_torch_vel_estimator:
            self.robot_dummy = Lite3Dummy(
                motor_control_mode=MotorControlMode.HYBRID,
                enable_action_interpolation=False,
                time_step=self.cfg.sim.dt,
                action_repeat=1,
                device=self.device)

        self._clock = self.robot.GetTimeSinceReset
        self._reset_time = self._clock()
        self.start_time_wall = self.time_since_reset()

        self._state_estimator = {}
        self._state_estimator[
            'robot_state_estimator'] = robot_velocity_estimator.VelocityEstimator(
                self.robot, moving_window_filter_size=20)
        self._state_estimator[
            'torch_robot_state_estimator'] = torch_velocity_estimator(
                robotIn=self.robot_dummy, window_size=20, device=self.device)
        # self._prepare_reward_function()
        self.init_done = True
        self.tik = 0.0

        self.fixed_commands = self.cfg.commands.fixed_commands
        self.curriculum_factor = self.cfg.env.curriculum_factor
        self.height_noise_mean = 0.
        self.pmtg = PMTrajectoryGenerator(robot=self.robot_dummy,
                                          clock=self.robot.GetTimeSinceReset,
                                          num_envs=self.num_envs,
                                          device=self.device,
                                          param=self.cfg.pmtg,
                                          task_name="lite3_real")

        self.count = 0
        self.reset_flag = False

    def time_since_reset(self):
        """Return the time elapsed since the last reset."""
        return self._clock() - self._reset_time

    def base_task_init(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize the base task parameters.

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            sim_device (string): 'cuda' or 'cpu'
            headless (bool): Run without rendering if True
        """
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = parse_device_str(self.sim_device)
        self.headless = headless
        if sim_device_type == 'cuda':
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.num_policy_outputs = cfg.env.num_policy_outputs

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs,
                                                  self.num_privileged_obs,
                                                  device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.num_dof = 12
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        # create some wrapper tensors for different slices
        self.root_states = to_torch([0., 0., 0., 0., 0., 0., 1, 0., 0., 0., 0., 0., 0.], device=self.device).repeat(
            (self.num_envs, 1))
        self.dof_pos = torch.zeros(self.num_envs,
                                   self.num_dof,
                                   dtype=torch.float32,
                                   device=self.device,
                                   requires_grad=False)
        self.dof_vel = torch.zeros(self.num_envs,
                                   self.num_dof,
                                   dtype=torch.float32,
                                   device=self.device,
                                   requires_grad=False)
        self.base_quat = self.root_states[:, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs,
                                   self.num_actions,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs,
                                   self.num_actions,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs,
                                        self.num_actions,
                                        dtype=torch.float,
                                        device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs,
                                             self.num_actions,
                                             dtype=torch.float,
                                             device=self.device,
                                             requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_last_dof_vel = torch.zeros_like(self.dof_vel)
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel, self.obs_scales.lin_vel,
                self.obs_scales.ang_vel
            ],
            device=self.device,
            requires_grad=False,
        )
        self.feet_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self.feet_air_time = torch.zeros(self.num_envs,
                                         len(self.feet_names),
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs,
                                         len(self.feet_names),
                                         dtype=torch.bool,
                                         device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat,
                                                     self.gravity_vec)
        self.body_height = torch.tensor([[STAND_UP_HEIGHT]],
                                        dtype=torch.float,
                                        device=self.device).repeat(
                                            self.num_envs, 1)

        self.measured_heights = 0
        self.last_last_last_pos = torch.zeros((self.num_envs, self.num_dof),
                                              device=self.device,
                                              dtype=torch.float,
                                              requires_grad=False)
        self.last_last_pos = torch.zeros((self.num_envs, self.num_dof),
                                         device=self.device,
                                         dtype=torch.float,
                                         requires_grad=False)
        self.last_pos = torch.zeros((self.num_envs, self.num_dof),
                                    device=self.device,
                                    dtype=torch.float,
                                    requires_grad=False)
        self.cpg_phase_information = torch.ones((self.num_envs, 13),
                                                device=self.device,
                                                dtype=torch.float,
                                                requires_grad=False)
        self.delta_phi = torch.zeros(self.num_envs,
                                     4,
                                     dtype=torch.float,
                                     device=self.device,
                                     requires_grad=False)
        self.residual_angle = torch.zeros(self.num_envs,
                                          12,
                                          dtype=torch.float,
                                          device=self.device,
                                          requires_grad=False)

        # joint positions offsets and PD gains
        self.dof_names = [
            'FL_HipX_joint', 'FL_HipY_joint', 'FL_Knee_joint', 'FR_HipX_joint',
            'FR_HipY_joint', 'FR_Knee_joint', 'HL_HipX_joint', 'HL_HipY_joint',
            'HL_Knee_joint', 'HR_HipX_joint', 'HR_HipY_joint', 'HR_Knee_joint'
        ]
        self.default_dof_pos = torch.zeros(self.num_dof,
                                           dtype=torch.float,
                                           device=self.device,
                                           requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_angles = torch.tensor(self.cfg.normalization.clip_angles,
                                        dtype=torch.float,
                                        device=self.device)
        self.torque_limits = 25

        self.feet_indices = torch.zeros(len(self.feet_names),
                                        dtype=torch.long,
                                        device=self.device,
                                        requires_grad=False)
        for f in range(len(self.feet_names)):
            self.feet_indices[f] = f

        self.dof_pos_limits = [[-0.2, 0.2], [-0.314, 2.67], [-2.75, -1.0]]

    def reset(self):
        """Resets the robot environment.

        This method resets the environment to its initial state and returns the initial observations.
        """
        self.reset_flag = True
        self.reset_buf.zero_()
        zero_action = torch.zeros((self.num_envs, self.num_actions + 4),
                                  device=self.device,
                                  dtype=torch.float)
        self.step(zero_action)
        self.reset_robot()
        zero_action = torch.zeros((self.num_envs, self.num_actions + 4),
                                  device=self.device,
                                  dtype=torch.float)
        self.step(zero_action)
        self.pmtg.reset([0])
        self._reset_time = self._clock()
        self.start_time_wall = 0
        self.reset_flag = False
        return None, None

    def step(self, policy_outputs):
        """Applies actions, simulates the environment, and performs post-physics steps.

        This method applies the given policy outputs as actions, simulates the environment,
        and performs additional steps such as refreshing state tensors and clipping observations.

        Args:
            policy_outputs (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            Tuple: A tuple containing the observation buffer, privileged observation buffer,
                reward buffer, reset buffer, and additional information.
        """
        for _ in range(self.cfg.control.decimation):
            self.delta_phi = policy_outputs[:, :4]
            self.residual_angle = policy_outputs[:, 4:]
            delta_phi = policy_outputs[:, :4] * self.cfg.control.delta_phi_scale
            residual_angle = policy_outputs[:, 4:] * self.cfg.control.residual_angle_scale
            residual_xyz = torch.zeros((self.num_envs, self.num_actions),
                                       device=self.device,
                                       dtype=torch.float)
            pmtg_joints = self.pmtg.get_action(delta_phi,
                                               residual_xyz,
                                               residual_angle,
                                               self.base_quat,
                                               command=self.commands)
            pmtg_joints = torch.clip(pmtg_joints.reshape(4, 3),
                                     self.clip_angles[:, 0],
                                     self.clip_angles[:, 1])
            self.actions = pmtg_joints.view(1, 12)
            if self.commands[0, 0] < -0.1:
                self.actions[0, 8] += 0.13
                self.actions[0, 11] += 0.13
            if self.common_step_counter < 250:
                self.actions[0, 8] += 0.2
                self.actions[0, 11] += 0.2

            self.hybrid_action = self._compute_command(self.actions)
            if not self.reset_flag:
                phi = self.pmtg.phi.cpu().flatten().numpy() / 6.2830
                newphi = [phi[1], phi[0], phi[3], phi[2]]
                send_b = True
                if self.common_step_counter < 5:
                    for p in newphi:
                        if p > 0.6:
                            send_b = False
                            break
                if send_b:
                    self.robot.ApplyAction(self.hybrid_action)

            while self.time_since_reset() - self.start_time_wall < self.cfg.sim.dt:
                pass
            self.start_time_wall = self.time_since_reset()

            self.robot.ReceiveObservation()  # refresh raw state

            self.refresh_dof_state_tensor()
            if self.robot_dummy is None:
                self.refresh_actor_root_state_tensor()
            else:
                self.torch_refresh_actor_root_state_tensor()

        self.count += 1
        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,
                                                 -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def refresh_dof_state_tensor(self):
        dof_pos = self._serialize_dof(self.robot.GetTrueMotorAngles())
        dof_vel = self._serialize_dof(self.robot.GetMotorVelocities())
        self.dof_pos[:] = torch.from_numpy(dof_pos).to(self.device).repeat(
            (self.num_envs, 1))
        self.dof_vel[:] = torch.from_numpy(dof_vel).to(self.device).repeat(
            (self.num_envs, 1))

    def refresh_actor_root_state_tensor(self):
        self._state_estimator['robot_state_estimator'].update(
            robot_state_tick=self.robot.tick,
            current_time=self.time_since_reset())
        self.body_height = self._state_estimator[
            'robot_state_estimator'].estimate_robot_height()
        base_orientation = self.robot.GetBaseOrientation()
        self.base_quat[:] = torch.from_numpy(base_orientation).to(
            self.device).unsqueeze(0).repeat((self.num_envs, 1))

        base_lin_vel_world = np.array([
            self._state_estimator['robot_state_estimator'].
            com_velocity_world_frame
        ],
                                      dtype=np.float32)  # world frame
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat,
            torch.from_numpy(base_lin_vel_world).to(self.device).repeat(
                (self.num_envs, 1)))
        base_ang_vel = np.array([self.robot.GetBaseRollPitchYawRate()],
                                dtype=np.float32)  # base frame
        self.base_ang_vel[:] = torch.from_numpy(base_ang_vel).to(
            self.device).repeat((self.num_envs, 1))

    def torch_refresh_actor_root_state_tensor(self):
        torch_se = self._state_estimator['torch_robot_state_estimator']
        self.base_quat[:] = torch.from_numpy(self.robot._base_orientation).to(
            self.device, dtype=torch.float).unsqueeze(0)
        robot_obs_buf = self.robot.GetRobotObs().to(device=self.device)
        self.base_ang_vel[:] = robot_obs_buf[0, 6:9]
        self.robot_dummy.set_state(robot_obs_buf)
        v, h = torch_se(self.robot.tick / 1000.0, robot_obs_buf)
        self.base_lin_vel[:] = v
        self.body_height[:] = h

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
        self.cpg_phase_information = self.pmtg.update_observation()  # TODO

        self._post_physics_step_callback()

        self.foot_forces = torch.tensor(self.robot.GetFootForce()).to(
            self.device)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_dof_vel[:] = self.last_dof_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_last_last_pos[:] = self.last_last_pos[:]
        self.last_last_pos[:] = self.last_pos[:]
        self.last_pos[:] = self.dof_pos[:]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.robot.ReceiveObservation()
        self.reset_buf = torch.tensor([False]).to(self.device)
        roll, pitch, yaw = self.robot.GetBaseRollPitchYaw()
        feet_not_contact = (~self.robot.GetFootContacts()).all()
        self.reset_buf |= torch.tensor(feet_not_contact).to(self.device)
        self.reset_buf |= torch.tensor([self.body_height < 0.20
                                        ]).to(self.device)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        if self.reset_buf:
            print('feet_not_contact:', feet_not_contact)
            print('body_height:', self.body_height)
            print('time_out_buf:', self.time_out_buf)

    def reset_robot(self):
        """ Reset a1 robot and real world environment.
            Resets some buffers
        """
        # reset buffers
        self.last_actions[0] = 0.
        self.last_dof_vel[0] = 0.
        self.feet_air_time[0] = 0.
        self.episode_length_buf[0] = 0
        self.reset_buf[0] = 0
        # fill extras
        self.extras["episode"] = {}
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.robot.Reset()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            self.episode_length_buf %
            int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
                as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        if self.cfg.commands.gamepad_commands:
            lin_speed, ang_speed, e_stop = self.command_function(
                self.time_since_reset())
            if e_stop:
                sys.exit(0)
            self.commands[:, 0] = lin_speed[0]
            self.commands[:, 1] = lin_speed[1]
            self.commands[:, 2] = ang_speed

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.fixed_commands is not None:
            self.commands[env_ids,
                          0] = torch.tensor([self.fixed_commands[0]]).repeat(len(env_ids)).to(device=self.device)
            self.commands[env_ids,
                          1] = torch.tensor([self.fixed_commands[1]]).repeat(len(env_ids)).to(device=self.device)
            self.commands[env_ids,
                          2] = torch.tensor([self.fixed_commands[2]]).repeat(len(env_ids)).to(device=self.device)
        else:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                         self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                         self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                             self.command_ranges["heading"][1], (len(env_ids), 1),
                                                             device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                             self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                             device=self.device).squeeze(1)

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _get_heights(self, env_ids=None):
        self.num_height_points = 187
        self.measured_heights = torch.zeros(self.num_envs,
                                            self.num_height_points,
                                            device=self.device)
        return self.measured_heights

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def create_sim(self):
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        # load robot asset from URDF file
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                          asset_options)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        _ = self._process_dof_props(dof_props_asset, 0)

        self.penalize_joint_ids = []
        for i in range(len(self.dof_names)):
            if self.dof_names[i] in [
                    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint',
                    'RR_hip_joint'
            ]:
                self.penalize_joint_ids.append(i)

    def compute_observations(self):
        body_orientation = get_euler_xyz(self.base_quat)
        dof_pos_history = torch.cat(
            (self.last_last_last_pos, self.last_last_pos, self.last_pos),
            dim=-1)
        dof_pos_vel_history = torch.cat(
            (self.last_last_dof_vel, self.last_dof_vel), dim=-1)
        dof_pos_target_history = torch.cat(
            (self.last_last_actions, self.last_actions), dim=-1)
        cpg_phase_information = self.cpg_phase_information

        self.obs_buf = torch.cat(
            (self.commands[:, :3] * self.commands_scale,
             body_orientation * self.obs_scales.orientation,
             self.base_lin_vel * self.obs_scales.lin_vel, self.base_ang_vel *
             self.obs_scales.ang_vel, self.dof_pos * self.obs_scales.dof_pos,
             self.dof_vel * self.obs_scales.dof_vel,
             dof_pos_history * self.obs_scales.dof_pos,
             dof_pos_vel_history * self.obs_scales.dof_vel,
             dof_pos_target_history * self.obs_scales.dof_pos,
             cpg_phase_information * self.obs_scales.cpg_phase),
            dim=-1).to(self.device)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.body_height - self.cfg.rewards.base_height_target -
                self.measured_heights, -1,
                1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def _compute_command(self, actions):
        # PD controller
        control_type = self.cfg.control.control_type
        self.torques = self.p_gains * (
            self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        if control_type == "T":
            torques = self.p_gains * (
                actions - self.dof_pos) - self.d_gains * self.dof_vel
            torques = np.clip(
                torques.squeeze(0).detach().numpy(), -self.torque_limits,
                self.torque_limits)
            torques = self._serialize_dof(torques)

            for index in range(len(torques)):
                self.hybrid_action[5 * index + 4] = torques[index]
        elif control_type == "P":  # True
            phi = self.pmtg.phi.cpu().flatten().numpy() / 6.2830
            newphi = [phi[1], phi[0], phi[3], phi[2]]
            positions = actions.detach().cpu().numpy()[0]
            positions = self._serialize_dof(positions)  # only the first robot
            for leg in range(4):
                for motor in range(3):
                    positions[3 * leg + motor] = np.clip(
                        positions[3 * leg + motor],
                        self.dof_pos_limits[motor][0],
                        self.dof_pos_limits[motor][1])
            for index in range(len(positions)):
                self.hybrid_action[5 * index +
                                   0] = positions[index]  # position

                sin_phi = np.sin(newphi[index // 3] * 3.1415 * 2 / 1.2)
                ratio = 1.0
                if sin_phi > 0:
                    ratio *= 3

                if index in [2, 5]:
                    self.hybrid_action[5 * index +
                                       1] = 17 + 2 * ratio * sin_phi
                elif index in [8, 11]:
                    self.hybrid_action[5 * index +
                                       1] = 17 + 2 * ratio * sin_phi
                elif index in [1, 4]:
                    self.hybrid_action[5 * index +
                                       1] = 19 + 2 * ratio * sin_phi
                elif index in [7, 10]:  # self.cfg.control.stiffness['joint'] +
                    self.hybrid_action[5 * index +
                                       1] = 22 + 2 * ratio * sin_phi

                else:
                    self.hybrid_action[5 * index +
                                       1] = self.cfg.control.stiffness[
                                           'joint']  # p
                self.hybrid_action[5 * index + 2] = 0  # velocity
                self.hybrid_action[5 * index +
                                   3] = self.cfg.control.damping['joint']  # d
                self.hybrid_action[5 * index + 4] = 0  # torque

        return self.hybrid_action

    def _serialize_dof(self, dof_data):
        serialized_data = np.zeros(12, dtype=np.float32)
        serialized_data[0:3] = dof_data[3:6]
        serialized_data[3:6] = dof_data[0:3]
        serialized_data[6:9] = dof_data[9:12]
        serialized_data[9:12] = dof_data[6:9]
        return serialized_data

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params['sim']['dt']
        self.obs_scales = self.cfg.normalization.obs_scales
        self.priv_obs_scales = self.cfg.normalization.priv_obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # ------------ reward functions----------------
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return 0.

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = torch.tensor(self.robot.GetFootContacts()).to(self.device)
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact,
            dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2],
                                  dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 0.

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((self.foot_forces -
                          self.cfg.rewards.max_contact_force).clip(min=0.),
                         dim=1)

    def _reward_feet_height(self):
        # Penalize feet height error
        feet_height_error = self.pmtg.foot_target_position_in_base_frame[:, :, 2] - torch.tensor(
            self.robot.GetFootPositionsInBaseFrame()).unsqueeze(0).to(
                self.device)[:, :, 2]
        return torch.sum(torch.abs(feet_height_error), dim=1)
