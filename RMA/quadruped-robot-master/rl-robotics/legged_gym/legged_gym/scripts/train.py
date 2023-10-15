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
import numpy as np
import json
from datetime import datetime
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, Logger, register
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import class_to_dict


def train(args):
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for training
    env_cfg.commands.fixed_commands = None
    env_cfg.commands.gamepad_commands = False
    if args.task == "a1_real":
        env_cfg.env.num_envs = 1
        env_cfg.commands.gamepad_commands = True

    # prepare environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load model
    if args.load_run:
        train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
        enable_summary_writer=True)

    # record configs as log files
    if not os.path.exists(ppo_runner.log_dir):
        os.mkdir(ppo_runner.log_dir)
    with open(os.path.join(ppo_runner.log_dir, 'env_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(env_cfg), fp)
    with open(os.path.join(ppo_runner.log_dir, 'train_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(train_cfg), fp)

    # train ppo policy
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    args.save_rewards = True
    train(args)
