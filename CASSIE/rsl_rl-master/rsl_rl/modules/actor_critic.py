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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import math

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

# NOTE popsan仅为actor网络！！！
# NOTE 自定义了两种 snn 的 forward 和 backward 方法： PseudoEncoderSpikeRegular 和 PseudoSpikeRect

# NOTE 3级
# @torch.jit.export
class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        # print("DEBUG", input.shape, " ", input)
        return input.gt(ENCODER_REGULAR_VTH).float() # NOTE 前向传播：input比ENCODER_REGULAR_VTH大的部分为1，否则为0
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() # NOTE 反向传播：复制不变
        return grad_input

# NOTE 2级
# @torch.jit.export
class PopSpikeEncoderRegularSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply

        print("check device", self.device)
        # Compute evenly distributed mean and variance
        # NOTE mean均值 variance方差 range先前指定的范围
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)

        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num # NOTE 最小维度单位中的值都会加上依次翻倍的delta_mean
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean) # NOTE 递增mean (tensor)
        self.std = nn.Parameter(tmp_std) # NOTE std (tensor)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1) # NOTE [env_num, obs_num, 1]
        # print("DEBUG1", torch.max(obs), " ", torch.min(obs))
        # Receptive Field of encoder population has Gaussian Shape
        # NOTE 每组（每个population）都是独立的分布，std一样但是mean不一样，这里pop_act代表的是每个值在给定分布下可能的概率
        # NOTE [env_num, obs_num*pop_num] 这obs_num个obsevation的值，都加减不同的值分为5个population，然后再将全部obs_num*pop_num个数据作为此次观测的值
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)  # NOTE if sim cant work, change rl to cpu
        # print("self.mean ", self.mean.shape, " ", self.mean)
        # print("pop_act ", pop_act.shape)
        # print("result", (obs - self.mean).shape, " ", (obs - self.mean)[0])
        # print("result2", pop_act[0])
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        # print("pop_volt ", pop_volt.shape)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # print("pop_spikes ", pop_spikes.shape)
        # Generate Regular Spike Trains
        # NOTE 构造时长为5的脉冲信号
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt) # NOTE 达峰值（伪求导过程）
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH # NOTE 电信号衰减(原来达到ENCODER_REGULAR_VTH的位置的值衰减为ENCODER_REGULAR_VTH以下)
            # print("pop_spikes ", pop_spikes.shape, " ", pop_spikes[:, :, step]) # NOTE (10, 170, 5)
        # print("DEBUG1", torch.sum(pop_spikes))
        return pop_spikes # NOTE 一定时间轴内（时间步为5）各事件发生的轨迹（越接近各自种群的mean发生概率就越高（表现为这一事件的时间轴上1越多，0越少））

# NOTE 2级
class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    # TODO 一定要仔细查看decoder的输出，也就是mean的范围
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):  # Tanh  # nn.Identity
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        # print("pop_act", pop_act.shape)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        # raw_act = self.decoder(pop_act).view(-1, self.act_dim)
        # print("DEBUG1", torch.max(raw_act), " ", torch.min(raw_act))
        return raw_act

# NOTE 3级
@torch.jit.export
class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # NOTE 存储在forward()期间生成的值，在backward()期间从ctx.saved_tensors属性访问保存的值
        return input.gt(NEURON_VTH).float() # 大于0.5的部分为1否则为0
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # NOTE forward前的input与NEURON_VTH足够相似，对应元素值置1，否则置0
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PreMLP(nn.Module):
    def __init__(self, obs_dim, device):
        super().__init__()
        self.obs_dim = obs_dim
        network_shape = [96, 192, 96]
        layer_num = len(network_shape)
        self.model = [nn.Linear(obs_dim, network_shape[0]),
                      nn.ELU()]
        if layer_num > 1:
            for layer in range(layer_num - 1):
                self.model.extend(
                    [nn.Linear(network_shape[layer], network_shape[layer + 1]),
                     nn.ELU()])
        self.model.extend([nn.Linear(network_shape[-1], obs_dim)])
        self.model = nn.Sequential(*self.model)

    def forward(self, state):
        out = self.model(state)
        return out

# NOTE 2级
class SpikeMLP(nn.Module):
    """ Spike MLP with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        # NOTE 240(obs_num*pop_num)->256->256->60(act_num*pop_num)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        # NOTE syn_func为一个nn.Linear
        # NOTE pre_layer_output为[env_num, obs_num*pop_num]
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)  # NOTE if sim cant work, change rl to cpu
        # TODO 弄懂LIF模型
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = [] # NOTE 一个二维的list，其中hidden_states[][0～2]分别代表current, volt, spike
        print("check device", self.device)
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
        # print("DEBUGHERE", len(hidden_states[0]), " ", hidden_states[0])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )
            out_pop_act += out_pop_states[2]
            # print("DEBUGmu", out_pop_states[2])
        out_pop_act = out_pop_act / self.spike_ts
        # print("here", out_pop_act[2])
        return out_pop_act

# NOTE 1级（构建整个popsan）
class PopSpikeActor(nn.Module):
    """ Squashed Gaussian Stochastic Population Coding Spike Actor with Fix Encoder """
    # NOTE 只有前两个参数和环境有关，后面全部是人为设定的
    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, device):
        """
        :param obs_dim: observation dimension  NOTE 输入多少个观测
        :param act_dim: action dimension  NOTE 输出多少个动作
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        # self.act_dim = act_dim
        # self.premlp = PreMLP(obs_dim, device)
        self.encoder = PopSpikeEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)
        log_std = -0.001 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))  # WHY std为啥会一直增大
        # self.log_std = nn.Parameter(-0.5 * torch.zeros(act_dim))

    def forward(self, obs):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        batch_size = obs.shape[0]  # NOTE 第一个维度为batchsize
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        mu = self.decoder(out_pop_activity)
        std = torch.exp(self.log_std)
        # out = self.decoder(out_pop_activity)
        # out = F.softplus(out)
        # alpha, beta = out[:, :self.act_dim], out[:, self.act_dim:]

        print("DEBUGobs", torch.max(obs), " ", torch.min(obs))
        print("DEBUGobs", mu)
        print("DEBUGstd", std)
        #print("DEBUGmu", mu.shape, " ", obs.shape, " ", out.shape)

        return mu, std

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        # actor_layers = []
        # actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # actor_layers.append(activation)
        # for l in range(len(actor_hidden_dims)):
        #     if l == len(actor_hidden_dims) - 1:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
        #     else:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
        #         actor_layers.append(activation)
        # self.actor = nn.Sequential(*actor_layers)

        self.actor = PopSpikeActor(mlp_input_dim_a, num_actions, 10, 10, actor_hidden_dims, (-5, 5), math.sqrt(0.15),
                                   spike_ts=5, device="cuda")  # NOTE if sim cant work, change rl to cpu

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # mean = self.actor(observations) # ANN
        # self.distribution = Normal(mean, mean*0. + self.std) # ANN
        mean, std = self.actor(observations) # SNN
        self.distribution = Normal(mean, std) # SNN

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # actions_mean = self.actor(observations) # ANN
        actions_mean, _ = self.actor(observations)  # SNN
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
