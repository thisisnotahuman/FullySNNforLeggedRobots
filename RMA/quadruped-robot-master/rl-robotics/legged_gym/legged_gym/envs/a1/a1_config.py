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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1RoughCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.36]  # x,y,z [m]
        default_joint_angles = {
            # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,
            'RL_hip_joint': 0.,
            'FR_hip_joint': 0.,
            'RR_hip_joint': 0.,

            'FL_thigh_joint': 0.9,
            'RL_thigh_joint': 0.9,
            'FR_thigh_joint': 0.9,
            'RR_thigh_joint': 0.9,

            'FL_calf_joint': -1.8,
            'RL_calf_joint': -1.8,
            'FR_calf_joint': -1.8,
            'RR_calf_joint': -1.8,
        }

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_privileged_obs = 54  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 40
        episode_length_s = 20  # episode length in seconds
        curriculum_factor = 0.8

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_torch_vel_estimator = False
        use_actuator_network = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'  # a1_unitree.urdf
        name = "a1"
        foot_name = "foot"
        shoulder_name = "shoulder"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "shoulder"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.28
        loose_pitch = False
        still_all = True
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            lin_vel_z = -2.0
            ang_vel_xy = -0.5
            orientation = -5.0
            base_height = -1.0
            torques = -0.0002
            dof_vel = -0.0
            dof_acc = -1.25e-07
            action_rate = -0.0
            target_smoothness = -0.01
            collision = -1.0
            termination = -0.0
            dof_pos_limits = -10.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            feet_air_time = 1.0
            stumble = -0.3
            stand_still = -0.5
            feet_height = -3.0
            delta_phi = -0.1
            residual_angle = -0.1
            feet_velocity = -0.0
            episode_length = 0.1

    class normalization(LeggedRobotCfg.normalization):

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            height_measurements = 0.0

        dof_history_interval = 1

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        heights_uniform_noise = False
        heights_gaussian_mean_mutable = True
        heights_downgrade_frequency = False  # heights sample rate: 10 Hz

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            height_measurements = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        fixed_commands = None  # None or [lin_vel_x, lin_vel_y, ang_vel_yaw]
        gamepad_commands = True
        resampling_time = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # none, plane, heightfield or trimesh
        dummy_normal = False
        random_reset = True
        curriculum = True
        max_init_terrain_level = 2
        # terrain_length = 8.
        # terrain_width = 8.
        # num_rows = 20 # number of terrain rows (levels)
        # num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, wave]
        terrain_proportions = [0.15, 0.15, 0.15, 0.0, 0.2, 0.2, 0.15]
        # rough terrain only:
        measure_heights = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        randomize_com_offset = True
        com_offset_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]

    class pmtg(LeggedRobotCfg.pmtg):
        gait_type = 'trot'
        duty_factor = 0.6
        base_frequency = 1.5
        max_clearance = 0.15
        body_height = 0.28
        max_horizontal_offset = 0.05


class A1RoughCfgPPO(LeggedRobotCfgPPO):

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_a1'
        max_iterations = 6000  # number of policy updates
        resume = False
        resume_path = 'legged_gym/logs/rough_a1'  # updated from load_run and ckpt
        load_run = ''  # -1 = last run
        checkpoint = -1  # -1 = last saved model
