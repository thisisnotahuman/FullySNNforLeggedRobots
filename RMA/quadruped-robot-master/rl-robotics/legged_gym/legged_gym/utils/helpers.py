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
import numpy as np
import random

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def get_load_path(root, load_run='-1', checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs = sorted(runs,
                      key=lambda x: os.path.getmtime(os.path.join(root, x))
                      )
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if not load_run:
        print('no load_run specified, automatically loading from the last run...')
        load_run = '-1'
    if load_run == '-1':
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = filter(
            lambda x: os.path.isfile(os.path.join(load_run, x)) and os.path.splitext(x)[1] == ('.pt' or '.om'),
            os.listdir(load_run))
        models = sorted(models,
                        key=lambda x: os.path.getmtime(os.path.join(load_run, x))
                        )
        try:
            model = models[-1]
        except:
            raise ValueError("No model found in current load_run!")
    else:
        model = str(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def parse_device_str(device_str):
    device = 'cpu'
    device_id = 0

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(':')
        assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(
                f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        help="Resume training or start testing from a checkpoint. Overrides config file if provided.",
                        default="a1")
    parser.add_argument("--resume", help="Resume training from a checkpoint",
                        default=False, action="store_true")
    parser.add_argument("--experiment_name", type=str,
                        help="Name of the experiment to run or load. Overrides config file if provided.")
    parser.add_argument("--run_name", type=str,
                        help="Name of the run. Overrides config file if provided.")
    parser.add_argument("--load_run", type=str,
                        help="Name of the run to load when resume=True. "
                             "If -1: will load the last run. Overrides config file if provided.")
    parser.add_argument("--checkpoint", type=str,
                        help="Saved model checkpoint number. If -1: will load the last checkpoint. "
                             "Overrides config file if provided.")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Force display off at all times")
    parser.add_argument("--horovod", action="store_true", default=False,
                        help="Use horovod for multi-gpu training")
    parser.add_argument("--rl_device", type=str, default="cuda:0",
                        help="Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)")
    parser.add_argument("--num_envs", type=int,
                        help="Number of environments to create. Overrides config file if provided.")
    parser.add_argument("--seed", type=int,
                        help="Random seed. Overrides config file if provided.")
    parser.add_argument("--max_iterations", type=int,
                        help="Maximum number of training iterations. Overrides config file if provided.")
    parser.add_argument("--save_rewards", action="store_true", default=False,
                        help="Save every reward term to a csv file.")
    parser.add_argument("--physics_engine", type=str, default="physicsX")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--use_npu", action="store_true", default=False, help="Whether use npu for inference.")
    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    args = parser.parse_args()
    return args


def register(task_name, task_registry):
    if task_name == 'a1_real':
        from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
        from legged_gym.envs.a1.a1_legged_robot import A1LeggedRobot
        task_registry.register("a1_real", A1LeggedRobot, A1RoughCfg(),
                               A1RoughCfgPPO())
    elif task_name == 'lite3_real':
        from legged_gym.envs.lite3.lite3_config import Lite3RoughCfg, Lite3RoughCfgPPO
        from legged_gym.envs.lite3.lite3_legged_robot import Lite3LeggedRobot
        task_registry.register("lite3_real", Lite3LeggedRobot, Lite3RoughCfg(),
                               Lite3RoughCfgPPO())
    elif task_name == 'a1':
        from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
        from legged_gym.envs.base.legged_robot import LeggedRobot
        task_registry.register("a1", LeggedRobot, A1RoughCfg(),
                               A1RoughCfgPPO())
    elif task_name == 'lite3':
        from legged_gym.envs.base.legged_robot import LeggedRobot
        from legged_gym.envs.lite3.lite3_config import Lite3RoughCfg, Lite3RoughCfgPPO
        task_registry.register("lite3", LeggedRobot, Lite3RoughCfg(),
                               Lite3RoughCfgPPO())
    elif task_name == 'cassie':
        from legged_gym.envs.cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
        from legged_gym.envs.cassie.cassie import Cassie
        task_registry.register("cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO())
    else:
        raise Exception("no such task_name")
