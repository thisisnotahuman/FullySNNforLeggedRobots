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

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import (to_torch, quat_mul, quat_apply,
                                  torch_rand_float, tensor_clamp, scale,
                                  get_axis_params, quat_rotate_inverse)
from legged_gym.utils.isaacgym_utils import compute_meshes_normals, Point, get_euler_xyz, get_contact_normals

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.torch_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from .PMTrajectoryGenerator import PMTrajectoryGenerator
from legged_gym.envs.gamepad import gamepad_reader
from legged_gym.envs.a1.a1_real.a1_robot_dummy import A1Dummy
from legged_gym.envs.lite3.lite3_real.lite3_robot_dummy import Lite3Dummy
from legged_gym.envs.base.motor_config import MotorControlMode
from legged_gym.envs.estimator.torch_robot_velocity_estimator import VelocityEstimator, GoogleVelocityEstimator
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class LeggedRobot(BaseTask):

    def __init__(self,
                 cfg: LeggedRobotCfg,
                 sim_params,
                 physics_engine,
                 sim_device,
                 headless,
                 task_name='a1'):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

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
        self.debug_viz = cfg.viewer.debug_viz
        self.init_done = False
        self.task_name = task_name
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device,
                         headless)

        self.gamepad = gamepad_reader.Gamepad()
        self.command_function = self.gamepad.get_command

        self.robot = None
        if cfg.control.use_torch_vel_estimator:
            self.robot = A1Dummy(motor_control_mode=MotorControlMode.HYBRID,
                                 enable_action_interpolation=False,
                                 time_step=self.cfg.sim.dt,
                                 action_repeat=1,
                                 device=self.device,
                                 num_envs=self.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self._state_estimator = {}
        self._state_estimator['robot_state_estimator'] = VelocityEstimator(
            robotIn=self.robot, window_size=20, device=self.device)
        self._state_estimator[
            'google_robot_state_estimator'] = GoogleVelocityEstimator(
                robot=self.robot, moving_window_filter_size=20)

        self.history_update_cnt = 0
        self.count = 0
        self.datax = []
        self.datay1 = []
        self.datay2 = []
        self.datay3 = []
        self.datay4 = []
        self.init_done = True

        self.fixed_commands = self.cfg.commands.fixed_commands
        self.curriculum_factor = self.cfg.env.curriculum_factor
        self.height_noise_mean = 0.

        self.pmtg = PMTrajectoryGenerator(robot=self.robot,
                                          clock=self.clock,
                                          num_envs=self.num_envs,
                                          device=self.device,
                                          param=self.cfg.pmtg,
                                          task_name=task_name)

        # load actuator network
        self.actuator_net = None

    def clock(self):
        return self.gym.get_sim_time(self.sim)

    def step(self, policy_outputs):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            policy_outputs (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.delta_phi = policy_outputs[:, :4]
            self.residual_angle = policy_outputs[:, 4:]
            delta_phi = policy_outputs[:, :4] * self.cfg.control.delta_phi_scale
            residual_angle = policy_outputs[:,
                                            4:] * self.cfg.control.residual_angle_scale
            residual_xyz = torch.zeros(self.num_envs,
                                       self.num_actions).to(self.device)
            pmtg_joints = self.pmtg.get_action(delta_phi,
                                               residual_xyz,
                                               residual_angle,
                                               self.base_quat,
                                               command=self.commands)
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(pmtg_joints, -clip_actions,
                                      clip_actions).to(self.device)
            self.torques = self._compute_torques(self.actions).view(
                self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)

            if self.cfg.commands.gamepad_commands:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

        self.post_physics_step()
        self.count += 1

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,
                                                 -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz(self.base_quat)

        # update info just for terrian moveup/movedown
        if self.count % 3 == 0:  # every 3 step, means  3*12ms = 36ms
            self.episode_v_integral += torch.norm(self.root_states[:, :3] -
                                                  self.old_pos,
                                                  dim=-1)
            dyaw = self.rpy[:, 2] - self.old_rpy[:, 2]

            self.episode_w_integral += torch.where(
                torch.abs(dyaw) > torch.pi / 2, dyaw +
                torch.pow(-1.0,
                          torch.less(self.rpy[:, 2], torch.pi / 2).long() + 1)
                * torch.pi * 2, dyaw)
            self.old_pos[:] = self.root_states[:, :3]
            self.old_rpy[:] = self.rpy

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat,
                                                   self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat,
                                                   self.root_states[:, 10:13])
        self.world_lin_acc[:] = (self.root_states[:, 7:10] -
                                 self.world_lin_vel) / self.dt
        self.world_lin_vel[:] = self.root_states[:, 7:10]
        self.base_lin_acc[:] = quat_rotate_inverse(
            self.base_quat, self.world_lin_acc + self.imu_G_offset)

        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)

        self.feet_pos[:] = self.rigid_body_state[:, self.feet_indices, 0:3]
        self.feet_pos[:, :, :2] /= self.cfg.terrain.horizontal_scale
        self.feet_pos[:, :, :
                      2] += self.cfg.terrain.border_size / self.cfg.terrain.horizontal_scale
        if self.cfg.terrain.mesh_type == 'trimesh' and self.cfg.env.num_privileged_obs is not None:
            self.feet_pos[:, :, 0] = torch.clip(
                self.feet_pos[:, :, 0],
                min=0.,
                max=self.height_samples.shape[0] - 2.)
            self.feet_pos[:, :, 1] = torch.clip(
                self.feet_pos[:, :, 1],
                min=0.,
                max=self.height_samples.shape[1] - 2.)

            if self.cfg.terrain.dummy_normal is False:
                self.contact_normal[:] = get_contact_normals(
                    self.feet_pos, self.mesh_normals, self.sensor_forces)

        self.cpg_phase_information = self.pmtg.update_observation()

        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(self.contact, self.last_contacts)
        self.last_contacts = self.contact

        self._post_physics_step_callback()
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations(
        )  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        if self.num_privileged_obs is not None:
            self.compute_privileged_observations()
        if self.history_update_cnt % self.cfg.normalization.dof_history_interval == 0:
            self.last_last_actions[:] = self.last_actions[:]
            self.last_actions[:] = self.actions[:]
            self.last_last_dof_vel[:] = self.last_dof_vel[:]
            self.last_dof_vel[:] = self.dof_vel[:]
            self.last_last_last_pos[:] = self.last_last_pos[:]
            self.last_last_pos[:] = self.last_pos[:]
            self.last_pos[:] = self.dof_pos[:]
            self.last_root_vel[:] = self.root_states[:, 7:13]
            self.history_update_cnt = 0
        self.history_update_cnt += 1

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            # self._draw_contact_normal()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :],
            dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter %
                                             self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.episode_v_integral[env_ids].zero_()
        self.episode_w_integral[env_ids].zero_()
        self.pmtg.reset(env_ids)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.rpy[env_ids] = get_euler_xyz(self.root_states[env_ids, 3:7])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.old_pos[env_ids] = self.root_states[env_ids, :3]
        self.old_rpy[env_ids] = self.rpy[env_ids]

        self._resample_commands(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = self.dof_pos[env_ids]
        self.last_actions[env_ids] = self.dof_pos[env_ids]
        self.last_last_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.last_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.last_last_last_pos[env_ids] = self.dof_pos[env_ids]
        self.last_last_pos[env_ids] = self.dof_pos[env_ids]
        self.last_pos[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]

        # self.last_actions[env_ids] = 0.
        # self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges[
                "lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = self.reward_functions[i]() * self.reward_scales[
                name]  #* self.curriculum_factor
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        self.original_obs_buf = torch.cat(
            (self.base_lin_acc, self.rpy, self.base_ang_vel, self.dof_pos,
             self.dof_vel, self.contact_filt),
            dim=-1)  # 3+3+3+12+12+4=37

        if self.cfg.env.num_observations == 235:
            self.obs_buf = torch.cat(
                (self.base_lin_vel * self.obs_scales.lin_vel,
                 self.base_ang_vel * self.obs_scales.ang_vel,
                 self.projected_gravity, self.commands[:, :3] *
                 self.commands_scale, (self.dof_pos - self.default_dof_pos) *
                 self.obs_scales.dof_pos,
                 self.dof_vel * self.obs_scales.dof_vel, self.actions),
                dim=-1)
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(
                    self.root_states[:, 2].unsqueeze(1) -
                    self.cfg.rewards.base_height_target -
                    self.measured_heights, -1,
                    1.) * self.obs_scales.height_measurements
                self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

            if self.add_noise:
                if self.cfg.noise.heights_uniform_noise:
                    self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1
                                     ) * self.noise_scale_vec  # [:, -187:]=0.5
                    self.noised_height_samples[:] = self.measured_heights + (
                        2 * torch.rand_like(self.measured_heights) - 1) * 0.1
                else:
                    self.noise_scale_vec[-187:] = 0
                    self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                                     1) * self.noise_scale_vec
                    self.obs_buf[:, -187:] += torch.normal(
                        mean=self.height_noise_mean,
                        std=0.05,
                        size=(self.obs_buf[:, -187:].shape)).to(
                            device=self.device
                        ) * self.obs_scales.height_measurements
                    self.noised_height_samples[:] = self.measured_heights + torch.normal(
                        mean=self.height_noise_mean,
                        std=0.05,
                        size=(self.measured_heights.shape)).to(
                            device=self.device)
        elif self.cfg.env.num_observations == 320:  # True
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
                 self.rpy * self.obs_scales.orientation,
                 self.base_lin_vel * self.obs_scales.lin_vel,
                 self.base_ang_vel * self.obs_scales.ang_vel,
                 self.dof_pos * self.obs_scales.dof_pos,
                 self.dof_vel * self.obs_scales.dof_vel,
                 dof_pos_history * self.obs_scales.dof_pos,
                 dof_pos_vel_history * self.obs_scales.dof_vel,
                 dof_pos_target_history * self.obs_scales.dof_pos,
                 cpg_phase_information * self.obs_scales.cpg_phase),
                dim=-1)  # (N, 133)

            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:  # True
                heights = torch.clip(
                    self.root_states[:, 2].unsqueeze(1) -
                    self.cfg.rewards.base_height_target -
                    self.measured_heights, -1,
                    1.) * self.obs_scales.height_measurements
                self.obs_buf = torch.cat((self.obs_buf, heights),
                                         dim=-1)  # (N, 320)
            if self.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                                 1) * self.noise_scale_vec
        # torch.normal(mean=self.height_noise_mean, std=0.1, size=(self.measured_heights.shape)).to(device=self.device)
        self.original_obs_buf[:, :33] += (
            torch.rand_like(self.original_obs_buf[:, :33]) * 2.0 -
            1.0) * self.measure_noise_scale_vec

        if self.robot is not None:
            sim_t = self.gym.get_sim_time(self.sim)
            self.robot._base_orientation = self.base_quat
            self.robot.set_state(self.original_obs_buf)

            state_est = self._state_estimator['robot_state_estimator']
            base_lin_vel, h = state_est(sim_t, self.original_obs_buf)
            self.robot.estimated_velocity = base_lin_vel
            self.obs_buf[:, 6:9] = base_lin_vel * self.obs_scales.lin_vel
            heights = torch.clip(
                h - self.cfg.rewards.base_height_target -
                self.measured_heights, -1., 1.)
            self.obs_buf[:,
                         -187:] = heights * self.obs_scales.height_measurements

    def compute_privileged_observations(self):
        """ Computes privileged observations
        """
        contact_states = torch.norm(self.sensor_forces[:, :, :2], dim=2) > 1.
        contact_forces = self.sensor_forces.flatten(1)
        contact_normals = self.contact_normal
        if self.friction_coeffs is not None:
            friction_coefficients = self.friction_coeffs.squeeze(1).repeat(
                1, 4).to(self.device)
        else:
            friction_coefficients = torch.tensor(
                self.cfg.terrain.static_friction).repeat(self.num_envs,
                                                         4).to(self.device)
        thigh_and_shank_contact = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :],
            dim=-1) > 0.1
        external_forces_and_torques = torch.cat(
            (self.push_forces[:, 0, :], self.push_torques[:, 0, :]), dim=-1)
        airtime = self.feet_air_time
        self.privileged_obs_buf = torch.cat(
            (contact_states * self.priv_obs_scales.contact_state,
             contact_forces * self.priv_obs_scales.contact_force,
             contact_normals * self.priv_obs_scales.contact_normal,
             friction_coefficients * self.priv_obs_scales.friction,
             thigh_and_shank_contact *
             self.priv_obs_scales.thigh_and_shank_contact_state,
             external_forces_and_torques *
             self.priv_obs_scales.external_wrench,
             airtime * self.priv_obs_scales.airtime),
            dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id,
                                       self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.cfg.terrain.static_friction
            random_foot_restitution = self.cfg.asset.restitution_mean + torch_rand_float(
                self.cfg.asset.restitution_offset_range[0],
                self.cfg.asset.restitution_offset_range[1], (1, 1),
                device=self.device)
            if 'a1' in self.task_name:
                feet_list = [4, 8, 12, 16]
            elif 'lite' in self.task_name:
                feet_list = [3, 7, 11, 15]
            else:
                raise Exception("")
            if s in feet_list:
                props[s].restitution = random_foot_restitution
                props[s].compliance = self.cfg.asset.compliance

        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0],
                                                    friction_range[1],
                                                    (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof,
                                              2,
                                              dtype=torch.float,
                                              device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof,
                                              dtype=torch.float,
                                              device=self.device,
                                              requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof,
                                             dtype=torch.float,
                                             device=self.device,
                                             requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = 1.8  # props["velocity"][i].item()
                self.torque_limits[i] = 25  # props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[
                    i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[
                    i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            print(self.dof_pos_limits)
        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.randomize_com_offset:
            com_offset_rng = self.cfg.domain_rand.com_offset_range
            props[0].com += gymapi.Vec3(
                np.random.uniform(com_offset_rng[0][0], com_offset_rng[0][1]),
                np.random.uniform(com_offset_rng[1][0], com_offset_rng[1][1]),
                np.random.uniform(com_offset_rng[2][0], com_offset_rng[2][1]))
        return props

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
            lin_speed, ang_speed, e_stop = self.command_function(0)
            if e_stop:
                sys.exit(0)
            self.commands[:, 0] = torch.tensor([lin_speed[0]
                                                ]).to(device=self.device)
            self.commands[:, 1] = torch.tensor([lin_speed[1]
                                                ]).to(device=self.device)
            self.commands[:,
                          2] = torch.tensor([ang_speed]).to(device=self.device)

        if self.cfg.terrain.measure_heights:  # True
            if self.cfg.noise.heights_downgrade_frequency:
                if self.common_step_counter == 1 or self.common_step_counter % int(
                        1 / self.dt / 10) == 0:
                    self.measured_heights = self._get_heights()
            else:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
                self.common_step_counter % self.cfg.domain_rand.push_interval
                == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.fixed_commands is not None:
            self.commands[env_ids,
                          0] = torch.tensor([self.fixed_commands[0]]).repeat(
                              len(env_ids)).to(device=self.device)
            self.commands[env_ids,
                          1] = torch.tensor([self.fixed_commands[1]]).repeat(
                              len(env_ids)).to(device=self.device)
            self.commands[env_ids,
                          2] = torch.tensor([self.fixed_commands[2]]).repeat(
                              len(env_ids)).to(device=self.device)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                device=self.device).squeeze(1)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(
                    self.command_ranges["heading"][0],
                    self.command_ranges["heading"][1], (len(env_ids), 1),
                    device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(
                    self.command_ranges["ang_vel_yaw"][0],
                    self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                    device=self.device).squeeze(1)

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(
                self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (
                actions - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (
                actions - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # todo
    def _compute_torques_net(self, actions):
        torques = torch.zeros((self.num_envs, 12),
                              dtype=torch.float,
                              device=self.device)
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                pos_error_cur = actions - self.dof_pos
                pos_error_his_past2 = self.pos_error_his_deque.popleft()
                pos_error_his_past1 = self.pos_error_his_deque[0]
                pos_error_his = torch.stack(
                    (pos_error_his_past2, pos_error_his_past1, pos_error_cur),
                    dim=2)
                self.pos_error_his_deque.append(pos_error_cur.clone())

                vel_his_past2 = self.vel_his_deque.popleft()
                vel_his_past1 = self.vel_his_deque[0]
                vel_his = torch.stack(
                    (vel_his_past2, vel_his_past1, self.dof_vel), dim=2)
                self.vel_his_deque.append(self.dof_vel.clone())
                torques = self.actuator_net(pos_error_his, vel_his)
            # exit(0)
        # return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1., 1., (len(env_ids), 2),
                device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6),
            device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.cfg.env.num_privileged_obs is None:
            max_vel = self.cfg.domain_rand.max_push_vel_xy
            self.root_states[:,
                             7:9] = torch_rand_float(
                                 -max_vel,
                                 max_vel, (self.num_envs, 2),
                                 device=self.device)  # lin vel x/y
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_states))
        else:
            max_force = self.cfg.domain_rand.max_push_force
            max_torque = self.cfg.domain_rand.max_push_torque
            self.push_forces[:, 0, :] = torch_rand_float(-max_force,
                                                         max_force,
                                                         (self.num_envs, 3),
                                                         device=self.device)
            self.push_torques[:, 0, :] = torch_rand_float(-max_torque,
                                                          max_torque,
                                                          (self.num_envs, 3),
                                                          device=self.device)
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.push_forces),
                gymtorch.unwrap_tensor(self.push_torques), gymapi.ENV_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] -
                              self.env_origins[env_ids, :2],
                              dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2

        # robots that walked less than half of their required distance go to simpler terrains
        # move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        move_down = torch.logical_or(self.episode_v_integral[env_ids] < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5, \
                        (self.episode_w_integral[env_ids] / self.commands[env_ids, 2]/self.max_episode_length_s) < 0.5
                    ) * ~move_up

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        if self.cfg.terrain.random_reset:
            # Robots that solve the last level are sent to a random one
            self.terrain_levels[env_ids] = torch.where(
                self.terrain_levels[env_ids] >= self.max_terrain_level,
                torch.randint_like(self.terrain_levels[env_ids],
                                   self.max_terrain_level),
                torch.clip(self.terrain_levels[env_ids],
                           0))  # (the minumum level is zero)
        else:
            # Robots that solve the last level are remained in the last row
            self.terrain_levels[env_ids] = torch.where(
                self.terrain_levels[env_ids] >= self.max_terrain_level,
                torch.clip(self.terrain_levels[env_ids],
                           max=self.max_terrain_level - 1),
                torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]
                      ) / self.max_episode_length > 0.8 * self.reward_scales[
                          "tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        if self.obs_buf.shape[1] == 235:
            noise_vec[:
                      3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            noise_vec[
                3:
                6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.gravity * noise_level
            noise_vec[9:12] = 0.  # commands
            noise_vec[
                12:
                24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[
                24:
                36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[36:48] = 0.  # previous actions
            if self.cfg.terrain.measure_heights:
                noise_vec[
                    48:
                    235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        elif self.obs_buf.shape[1] in [320, 133]:
            noise_vec[:3] = 0.  # commands
            noise_vec[
                3:
                6] = noise_scales.orientation * noise_level * self.obs_scales.orientation
            noise_vec[
                6:
                9] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            noise_vec[
                9:
                12] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[
                12:
                24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[
                24:
                36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[36:72] = 0.  # joint position history
            noise_vec[72:96] = 0.  # joint velocity history
            noise_vec[96:120] = 0.  # joint target history
            noise_vec[120:133] = 0.  # CPG phase info
            if self.cfg.terrain.measure_heights:
                noise_vec[
                    133:
                    320] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[...,
                                                                           0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[...,
                                                                           1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz(self.base_quat)  # xyzw
        self.old_rpy = self.rpy.clone()
        self.old_pos = torch.zeros(self.num_envs,
                                   3,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
        self.old_pos[:] = self.root_states[:, :3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13)
        print(self.rigid_body_state.shape)  # 4*5 + 1
        print(self.feet_indices)
        print("num_envs = ", self.num_envs)

        self.feet_pos = self.rigid_body_state[:, self.feet_indices, 0:3]
        # self.hip_pos = self.rigid_body_state[:, self.feet_indices-3, 0:3]

        self.sensor_forces = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, 4, 6)[..., :3]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.measure_noise_scale_vec = torch.tensor(
            [
                0.1,
                0.1,
                1.0,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
            ] + [0.01] * 24,
            dtype=torch.float,
            device=self.device).unsqueeze(0)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx),
                                    device=self.device).repeat(
                                        (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat(
            (self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs,
                                   self.num_actions,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
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
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
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
        )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.desired_feet_air_time = torch.tensor([
            1.0 / (self.cfg.pmtg.base_frequency) *
            (1.0 - self.cfg.pmtg.duty_factor)
        ],
                                                  dtype=torch.float,
                                                  device=self.device,
                                                  requires_grad=False)

        self.last_contacts = torch.zeros(self.num_envs,
                                         len(self.feet_indices),
                                         dtype=torch.bool,
                                         device=self.device,
                                         requires_grad=False)
        self.contact = torch.zeros_like(self.last_contacts)
        self.contact_filt = torch.zeros_like(self.last_contacts)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 7:10])
        self.world_lin_vel = self.root_states[:, 7:10]
        self.world_lin_acc = torch.zeros_like(self.world_lin_vel)
        self.base_lin_acc = torch.zeros_like(self.world_lin_acc)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat,
                                                     self.gravity_vec)
        self.imu_G_offset = to_torch([0., 0., 9.8], device=self.device).repeat(
            (self.num_envs, 1))
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.friction_coeffs = None
        self.push_forces = torch.zeros((self.num_envs, self.num_bodies, 3),
                                       device=self.device,
                                       dtype=torch.float,
                                       requires_grad=False)
        self.push_torques = torch.zeros((self.num_envs, self.num_bodies, 3),
                                        device=self.device,
                                        dtype=torch.float,
                                        requires_grad=False)
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
        self.contact_normal = torch.tensor([0., 0., 1.
                                            ]).repeat(self.num_envs,
                                                      4).to(device=self.device)
        self.noised_height_samples = torch.zeros(self.num_envs,
                                                 187,
                                                 dtype=torch.float,
                                                 device=self.device,
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
        self.default_dof_pos = torch.zeros(self.num_dof,
                                           dtype=torch.float,
                                           device=self.device,
                                           requires_grad=False)
        for i in range(self.num_dofs):
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

        self.original_obs_buf = torch.cat(
            (self.base_lin_acc, self.rpy, self.base_ang_vel, self.dof_pos,
             self.dof_vel, self.contact_filt),
            dim=-1)  # 3+3+3+12+12+4=37
        if self.robot is not None:
            self.robot._base_orientation = self.base_quat
            self.robot.set_state(self.original_obs_buf)
        self.extras["episode"] = {}

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
            name: torch.zeros(self.num_envs,
                              dtype=torch.float,
                              device=self.device,
                              requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples,
                                 hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

        if self.cfg.env.num_privileged_obs is not None and not self.cfg.terrain.dummy_normal:
            print('vertices: ', self.terrain.vertices.shape[0],
                  ' triangeles: ', self.terrain.triangles.shape[0])
            start_time = time.time()
            if torch.cuda.is_available():
                from legged_gym.utils.isaacgym_utils import compute_meshes_normals_gpu
                self.mesh_normals = compute_meshes_normals_gpu(
                    self.terrain.tot_rows, self.terrain.tot_cols,
                    self.terrain.vertices, self.terrain.triangles).to(self.device)
            else:
                self.mesh_normals = compute_meshes_normals(
                    self.terrain.tot_rows, self.terrain.tot_cols,
                    self.terrain.vertices, self.terrain.triangles).to(self.device)
            
            if torch.sum(
                    torch.logical_or(~self.mesh_normals.isreal(),
                                     self.mesh_normals.isnan())) > 0:
                raise ValueError("compute_meshes_normals")
            print(
                f'Time for computing mesh normal:{time.time() - start_time} s')

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
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
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            robot_asset)
        print("robot_asset", robot_asset)
        rigid_shape_props_asset_names = self.gym.get_asset_rigid_body_shape_indices(
            robot_asset)
        for ii in rigid_shape_props_asset_names:
            print("start = {}, count = {}".format(ii.start, ii.count))
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print(rigid_shape_props_asset)
        print(len(rigid_shape_props_asset))
        print(body_names)
        print(self.num_bodies)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        shoulder_names = [
            s for s in body_names if self.cfg.asset.shoulder_name in s
        ]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend(
                [s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in body_names if name in s])
        self.penalize_joint_ids = []
        self.upperAndlow_leg_joint_ids = []
        print(self.dof_names)
        print(penalized_contact_names)
        joint_names = list(self.cfg.init_state.default_joint_angles.keys())
        print("joint_names = ", joint_names)
        for i in range(len(self.dof_names)):
            if self.dof_names[i] in [
                    joint_names[0], joint_names[1], joint_names[2],
                    joint_names[3]
            ]:
                self.penalize_joint_ids.append(i)
            else:
                self.upperAndlow_leg_joint_ids.append(i)
        print("penalize_joint_ids = ", self.penalize_joint_ids)
        print("upperAndlow_leg_joint_ids = ", self.upperAndlow_leg_joint_ids)
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list,
                                        device=self.device,
                                        requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose,
                                               sensor_options)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper,
                                             int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1),
                                        device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset,
                                                      rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle,
                                              dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props,
                recomputeInertia=False)  # True

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names),
                                        dtype=torch.long,
                                        device=self.device,
                                        requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i])

        self.shoulder_indices = torch.zeros(len(shoulder_names),
                                            dtype=torch.long,
                                            device=self.device,
                                            requires_grad=False)
        for i in range(len(shoulder_names)):
            self.shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shoulder_names[i])
        print("self.shoulder_indices = ", self.shoulder_indices)
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[
                i] = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0],
                    penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[
                i] = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0],
                    termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs,
                                           3,
                                           device=self.device,
                                           requires_grad=False)
            # put robots at the origins defined by the terrain
            if self.cfg.terrain.evaluation_mode is True:
                self.terrain_levels = torch.zeros(
                    (self.num_envs, ), device=self.device).to(torch.long)
                self.terrain_types = torch.div(
                    torch.arange(self.num_envs, device=self.device),
                    (self.num_envs / self.cfg.terrain.num_cols),
                    rounding_mode='floor').to(torch.long)
                self.terrain_origins = torch.from_numpy(
                    self.terrain.env_origins).to(self.device).to(torch.float)
                self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                                                           self.terrain_types]
            else:
                max_init_level = self.cfg.terrain.max_init_terrain_level
                if not self.cfg.terrain.curriculum:
                    max_init_level = self.cfg.terrain.num_rows - 1
                self.terrain_levels = torch.randint(0,
                                                    max_init_level + 1,
                                                    (self.num_envs, ),
                                                    device=self.device)
                self.terrain_types = torch.div(
                    torch.arange(self.num_envs, device=self.device),
                    (self.num_envs / self.cfg.terrain.num_cols),
                    rounding_mode='floor').to(torch.long)
                self.max_terrain_level = self.cfg.terrain.num_rows
                self.terrain_origins = torch.from_numpy(
                    self.terrain.env_origins).to(self.device).to(torch.float)
                self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                                                           self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs,
                                           3,
                                           device=self.device,
                                           requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows),
                                    torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.priv_obs_scales = self.cfg.normalization.priv_obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02,
                                                      4,
                                                      4,
                                                      None,
                                                      color=(1, 0, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            if self.add_noise:
                heights = self.noised_height_samples[i].cpu().numpy()
            else:
                heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(
                self.base_quat[i].repeat(heights.shape[0]),
                self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer,
                                   self.envs[i], sphere_pose)

    def _draw_contact_normal(self):
        if self.cfg.env.num_privileged_obs is None or self.cfg.terrain.mesh_type not in [
                'heightfield', 'trimesh'
        ]:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        contact_states = torch.norm(self.sensor_forces[:, :, :2], dim=2) > 1.
        contact_normals = self.contact_normal.view(-1, 4, 3)
        for i in range(self.num_envs):
            pos = self.feet_pos[i]
            contact_state = contact_states[i]
            contact_normal = contact_normals[i]
            for j in range(pos.shape[0]):
                if contact_state[j].cpu().numpy() == True:
                    z = self.height_samples[int(pos[j][0])][int(
                        pos[j][1])] * self.cfg.terrain.vertical_scale
                    x = (pos[j][0] - self.terrain.border
                         ) * self.cfg.terrain.horizontal_scale
                    y = (pos[j][1] - self.terrain.border
                         ) * self.cfg.terrain.horizontal_scale

                    x_ = x + contact_normal[j][0]
                    y_ = y + contact_normal[j][1]
                    z_ = z + contact_normal[j][2]

                    p1 = Point(x, y, z)
                    p2 = Point(x_, y_, z_)
                    color = gymapi.Vec3(0.0, 1.0, 0.0)
                    gymutil.draw_line(p1, p2, color, self.gym, self.viewer,
                                      self.envs[i])

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y,
                         device=self.device,
                         requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x,
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids]) + (
                    self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points),
                self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights.view(self.num_envs,
                            -1) * self.terrain.cfg.vertical_scale

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        if not self.cfg.rewards.loose_pitch:
            return torch.sum(torch.square(self.projected_gravity[:, :2]),
                             dim=1)
        else:
            return torch.sum(torch.square(
                self.projected_gravity[:, 1].unsqueeze(1)),
                             dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        is_stance = ~self.pmtg.is_swing
        stance_leg_num = is_stance.sum(dim=1)
        contact_feet_height = self.rigid_body_state[:, self.feet_indices,
                                                    2] * is_stance
        contact_shoulder_height = self.rigid_body_state[:,
                                                        self.shoulder_indices,
                                                        2] * is_stance
        contact_leg_height_z = contact_shoulder_height - contact_feet_height
        rew = torch.sum(torch.abs(self.cfg.rewards.base_height_target -
                                  contact_leg_height_z) * is_stance,
                        dim=1)
        return rew

    #def _reward_base_height(self):   # cassie
    #    base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #    return torch.square(base_height - self.cfg.rewards.base_height_target)  

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(
            (self.last_dof_vel - self.dof_vel) / self.dt),
                         dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_target_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions) +
                         torch.square(self.actions - 2 * self.last_actions +
                                      self.last_last_actions),
                         dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
                               > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.)  # lower limit
        out_of_limits += (self.dof_pos -
                          self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) -
             self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(
                 min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) -
             self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(
                 min=0.),
            dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] -
                                               self.base_lin_vel[:, :2]),
                                  dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] -
                                     self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - self.desired_feet_air_time) * first_contact,
            dim=1)  # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        if not self.cfg.rewards.still_all:
            return torch.sum(torch.abs(self.dof_pos[:, self.penalize_joint_ids] \
                                    - self.default_dof_pos[:, self.penalize_joint_ids]), dim=1) \
                             * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        else:

            abad_joint_reward = torch.sum(torch.abs(self.dof_pos[:, self.penalize_joint_ids] - \
                                            self.default_dof_pos[:, self.penalize_joint_ids]), \
                                        dim=1)
            leg_joint_reward = 0
            if self.cfg.pmtg.consider_foothold:
                factor = self.pmtg.is_swing * self.pmtg.swing_phi
                leg_joint_reward = torch.sum(torch.reshape(torch.reshape(torch.abs(self.dof_pos[:, self.upperAndlow_leg_joint_ids] - \
                                                self.pmtg.planned_joint_angles[:, self.upperAndlow_leg_joint_ids]),
                                                (self.num_envs, 4, 2)) * (factor**2).unsqueeze(-1),
                                                (self.num_envs, 8)), dim=1)

            return abad_joint_reward + leg_joint_reward

    def _reward_feet_height(self):
        # Penalize feet height error
        is_swing = self.pmtg.is_swing
        # swing_leg_num = is_swing.sum(dim=1)
        feet_height_error = self.pmtg.foot_target_position_in_base_frame[:, :, 2] - (
            self.rigid_body_state[:, self.feet_indices, 2] -
            self.root_states[:, 2].reshape([-1, 1]))
        feet_height_error *= is_swing
        return torch.sum(torch.abs(feet_height_error.clip(min=0.0)), dim=1)

    def _reward_feet_velocity(self):
        # Penalize feet height error
        foot_v = self.rigid_body_state[:, self.feet_indices, 7:10]
        feet_v_slip = self.contact_filt * torch.norm(foot_v[:, :, :3], dim=2)

        return torch.sum(feet_v_slip, dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -
             self.cfg.rewards.max_contact_force).clip(min=0.),
            dim=1)

    def _reward_delta_phi(self):
        # penelize large delta_phi
        return torch.norm(self.delta_phi, dim=-1)

    def _reward_residual_angle(self):
        # penelize large residual_angle
        return torch.norm(self.residual_angle, dim=-1)

    def _reward_episode_length(self):
        # print(self.episode_length_buf)
        return 1.0 - torch.exp(
            -1.0 * self.episode_length_buf /
            1500) + 5.0 * self.terrain_levels / self.max_terrain_level
