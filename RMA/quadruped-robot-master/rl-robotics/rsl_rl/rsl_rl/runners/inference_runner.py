# SPDX-FileCopyrightText: Copyright (c) 2023 HUAWEI TECHNOLOGIES. All rights reserved.
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
import sys

currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(os.path.join(os.path.dirname(legged_gym_dir), "acllite"))

from legged_gym.utils.helpers import get_load_path
from rsl_rl.env import HistoryWrapper


class InferenceRunner:
    """
    Inference runner for running model inference on an NPU device.
    """

    def __init__(self, env: HistoryWrapper, train_cfg, log_dir=None):
        """
        Initializes an instance of the InferenceRunner class.

        Args:
            env (HistoryWrapper): The environment wrapper for model inference.
            train_cfg (dict): The configuration dictionary for training.
            log_dir (str, optional): The directory path for logging. Defaults to None.
        """
        self.model = None
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.env = env
        self.num_steps_per_env = self.cfg["num_steps_per_env"]

        # Import ACL libs
        import acl
        from acllite_model import AclLiteModel
        from acllite_resource import AclLiteResource

        # ACL resource initialization
        self.acl_resource = AclLiteResource()
        self.acl_resource.init()

        if self.cfg['resume']:
            # load previously trained model
            resume_path = get_load_path(os.path.dirname(log_dir),
                                        load_run=self.cfg['load_run'],
                                        checkpoint=self.cfg['checkpoint'])  # last one
            print(f"Loading model from: {resume_path}")
            self.model = AclLiteModel(resume_path)

        self.env.reset()

    def get_inference_policy(self, device=None):
        """
        Return inference policy.

        Args:
            device (str, optional): Unused device placeholder.
        """
        if self.model is not None:
            return self.model.execute
