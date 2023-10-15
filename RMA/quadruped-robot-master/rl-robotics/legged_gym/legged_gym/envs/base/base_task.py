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

import sys
from isaacgym import gymapi, gymutil
import collections
import torch


class BaseTask():
    """ Base class for RL tasks
    """
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(
            self.sim_device)
        self.headless = headless

        if sim_params.use_gpu_pipeline:
            self.device = 'cuda:' + str(self.sim_device_id)
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        # comment below two lines to enable depth camera utility
        if self.headless == True:
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
        self.obs_buf = torch.zeros(self.num_envs,
                                   self.num_obs,
                                   device=self.device,
                                   dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs,
                                   device=self.device,
                                   dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs,
                                    device=self.device,
                                    dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs,
                                              device=self.device,
                                              dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs,
                                        device=self.device,
                                        dtype=torch.bool)
        self.episode_v_integral = torch.zeros(self.num_envs,
                                              device=self.device,
                                              dtype=torch.float)
        self.episode_w_integral = torch.zeros(self.num_envs,
                                              device=self.device,
                                              dtype=torch.float)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs,
                                                  self.num_privileged_obs,
                                                  device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}
        self.pos_error_his_deque = collections.deque(
            maxlen=cfg.control.history_steps + 1)
        self.vel_his_deque = collections.deque(
            maxlen=cfg.control.history_steps + 1)
        for i in range(cfg.control.history_steps):
            self.pos_error_his_deque.append(
                torch.zeros((self.num_envs, 12),
                            dtype=torch.float,
                            device=self.device))
            self.vel_his_deque.append(
                torch.zeros((self.num_envs, 12),
                            dtype=torch.float,
                            device=self.device))

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim,
                                                 gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer,
                                                     gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V,
                                                     "toggle_viewer_sync")

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs,
                        self.num_policy_outputs,
                        device=self.device,
                        requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)