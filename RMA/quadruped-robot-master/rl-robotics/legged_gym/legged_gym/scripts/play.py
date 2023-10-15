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

currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
isaacgym_dir = os.path.join(os.path.dirname(legged_gym_dir), "isaacgym/python")
rsl_rl_dir = os.path.join(os.path.dirname(legged_gym_dir), "rsl_rl")
os.sys.path.insert(0, legged_gym_dir)
os.sys.path.insert(0, isaacgym_dir)
os.sys.path.insert(0, rsl_rl_dir)

import time
import numpy as np
import csv
from legged_gym.utils import get_args, Logger, register


def play(args):
    if not "real" in args.task:
        import isaacgym
        from legged_gym.utils.isaacgym_utils import export_policy_as_jit
    from legged_gym.utils.task_registry import task_registry
    from legged_gym import LEGGED_GYM_ROOT_DIR
    import torch
    record_policy_output = False
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1 if "real" in args.task else min(env_cfg.env.num_envs, 1)

    env_cfg.viewer.real_time_step = True
    env_cfg.pmtg.train_mode = False
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.terrain.evaluation_mode = True

    # customized terrain mode
    env_cfg.terrain.selected = True
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.commands.gamepad_commands = True
    # env_cfg.commands.fixed_commands = [0.5, 0.0, 0.0]
    # env_cfg.viewer.debug_viz = True
    env_cfg.terrain.terrain_length = 8
    env_cfg.terrain.terrain_width = 8
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 2
    env_cfg.env.episode_length_s = 100
    env_cfg.terrain.slope_treshold = 0.5  # for stair generation
    # env_cfg.terrain.terrain_kwargs = {'type': 'sloped_terrain', 'slope': 0.26}
    # env_cfg.terrain.terrain_kwargs = [{'type': 'slope_platform_stairs_terrain', 'slope': 0.36, 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5}]
    # env_cfg.terrain.terrain_kwargs = [{'type': 'slope_platform_stairs_terrain', 'slope': 0.36, 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5},
    #                                   {'type': 'stairs_platform_slope_terrain', 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5, 'slope': 0.36}]
    env_cfg.terrain.terrain_kwargs = [{
        'type': 'pyramid_stairs_terrain',
        'step_width': 0.3,
        'step_height': -0.1,
        'platform_size': 3.
    }, {
        'type': 'pyramid_stairs_terrain',
        'step_width': 0.3,
        'step_height': 0.1,
        'platform_size': 3.
    }, {
        'type': 'pyramid_sloped_terrain',
        'slope': 0.26
    }, {
        'type': 'discrete_obstacles_terrain',
        'max_height': 0.10,
        'min_size': 0.1,
        'max_size': 0.5,
        'num_rects': 200
    }, {
        'type': 'wave_terrain',
        'num_waves': 4,
        'amplitude': 0.15
    }, {
        'type': 'stepping_stones_terrain',
        'stone_size': 0.1,
        'stone_distance': 0.,
        'max_height': 0.03
    }]

    # evaluation policy mode
    if env_cfg.terrain.evaluation_mode:
        env_cfg.env.episode_length_s = 200  # long enough for traversal
        env_cfg.env.num_envs = 200
        env_cfg.terrain.num_rows = 5  # traverse length
        env_cfg.terrain.num_cols = 20
        env_cfg.terrain.terrain_proportions = [0.1, 0.2, 0.15, 0.15, 0.15, 0.15, 0.1]
        env_cfg.commands.fixed_commands = [0.5, 0.0, 0.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging (0:hip,1:thigh,2:calf)
    logger = Logger(env.dt, joint_index=joint_index)
    stop_state_log = int(4 / env.dt)  # number of steps before plotting states (default: 100)
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    obs_dict = env.get_observations()
    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]

    if env_cfg.terrain.evaluation_mode:
        cur_episode_length = np.zeros(env.num_envs)
        done_envs = np.zeros(env.num_envs, bool)
        finished_envs = np.zeros(env.num_envs, bool)
        while not done_envs.all():
            with torch.no_grad():
                if args.use_npu:
                    actions = policy([obs.numpy(), obs_history.numpy()])
                    actions = torch.tensor(actions[0])
                else:
                    actions = policy(obs, obs_history)
            obs, privileged_obs, rews, dones, infos = env.step(actions.detach())
            if env.num_privileged_obs is not None:
                obs = torch.cat((obs, privileged_obs), dim=-1)

            plus_ids = [idx for idx in range(env.num_envs) if done_envs[idx] == False]
            cur_episode_length[plus_ids] += 1
            done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(0).cpu().numpy()
            done_envs[done_ids] = True
            distances = torch.norm(env.root_states[:, :2] - env.env_origins[:, :2], dim=1)
            finished = distances >= env_cfg.terrain.terrain_length * env_cfg.terrain.num_rows
            finished = ~torch.from_numpy(done_envs).to(device=env.device) * finished
            finished_ids = finished.nonzero(as_tuple=False).squeeze(0).cpu().numpy()
            finished_envs[finished_ids] = True
            done_envs[finished_ids] = True

        print("Average episode length: ", cur_episode_length.mean())
        print("Average travesal rate: ", sum(finished_envs) / len(finished_envs))

    else:
        if record_policy_output:
            csv_header = [str(i) for i in range(env.num_policy_outputs)]
            with open(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run,
                                   'policy_outputs.csv'),
                      'w',
                      newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
        for i in range(10 * int(env.max_episode_length)):
            with torch.no_grad():
                if args.use_npu:
                    actions = policy([obs.numpy(), obs_history.numpy()])
                    actions = torch.tensor(actions[0])
                else:
                    actions = policy(obs, obs_history)
            obs_dict, rews, dones, infos = env.step(actions)
            obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                            'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()

    play(args)
