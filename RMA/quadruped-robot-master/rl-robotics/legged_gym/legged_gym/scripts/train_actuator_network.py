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
import yaml
import time
import argparse
from datetime import datetime
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
import legged_gym.utils.torch_schedule as schedulers
import legged_gym.utils.torch_datasets as datasets
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader


def get_time_stamp():
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')
    return '{}-{}-{}-{}-{}-{}'.format(month, day, year, hour, minute, second)

def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    
    args = parser.parse_args()
    
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args

def get_args(): # TODO: delve into the arguments
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--cfg_file_path", "type": str, "default": "legged_gym/legged_gym/scripts/actuator_net.yaml",
            "help": "Configuration file for training/playing"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights"},
        {"name": "--logdir", "type": str, "default": "legged_gym/logs/nets/"},
        {"name": "--save-interval", "type": int, "default": 0},
        {"name": "--no-time-stamp", "action": "store_true", "default": False,
            "help": "whether not add time stamp at the log path"},
        {"name": "--device", "type": str, "default": "cuda:0"},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {"name": "--render", "action": "store_true", "default": False,
            "help": "whether generate rendering file."}]
    
    # parse arguments
    args = parse_arguments(
        description="ActuatorNet",
        custom_parameters=custom_parameters)
    
    return args


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_activation_func(activation_name):
    if activation_name.lower() == 'softsign':
        return nn.Softsign() # x/(1+ |x|)
    elif activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError('Actication func {} not defined'.format(activation_name))

class ActuatorNet(nn.Module):
    def __init__(self, input_dim, out_dim, cfg, device='cuda:0'):
        super(ActuatorNet, self).__init__()
        self.cfg = cfg
        self.alg_cfg = cfg["algorithm"]
        self.device = device
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.layer_dims = [input_dim] + cfg['network']['mlp']['units'] + [out_dim]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
                       
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(cfg['network']['mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.net = nn.Sequential(*modules).to(device)
        
        print(self.net)

        # initialize optimizer
        self.lr = float(cfg["config"]['learning_rate'])
        self.epoch_num = int(cfg["config"]["max_epochs"])
        self.batch_size = int(cfg["config"]["batch_size"])
        self.truncate_grads = bool(cfg["config"]["truncate_grads"])
        if cfg["config"]["lr_schedule"] == "adaptive":
            self.scheduler = schedulers.AdaptiveScheduler(cfg["config"]["kl_threshold"])
        elif cfg["config"]["lr_schedule"] == "linear":
            self.scheduler = schedulers.LinearScheduler(self.lr, 
                max_steps=self.epoch_num, 
                apply_to_entropy=cfg["config"].get('schedule_entropy', False),
                start_entropy_coef=cfg["config"].get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()
        self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()
        self.dataloader = torch.utils.data.DataLoader(datasets.MotorDataset(self.batch_size, 
                                                            cfg["data"]["dataset_path"]),
                                                        batch_size=self.batch_size,
                                                        shuffle=False)
        
        self.log_dir = cfg["general"]["logdir"]
        self.writer = None
        self.tot_steps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def forward(self, x):
        """
        Input:
            pos_error_his: (N, 12, 3), Tensor[float]
            vel_his: (N, 12, 3), Tensor[float]
        """
        # input = torch.cat((x), dim=1)
        x = x.view((-1, 72))
        y = self.net(x)
        return y


    def train(self, num_learning_iterations=0):
        self.net.train()
        iter_ = 0
        if self.log_dir is not None and self.writer is None:
            print(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            print(len(self.dataloader.dataset))
        num_learning_iterations = len(self.dataloader.dataset)/self.batch_size * self.epoch_num
        # self.current_learning_iteration += num_learning_iterations
        # self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        start = time.time()
        for epcho in range(self.epoch_num):
            self.lr, _ = self.scheduler.update(self.lr, 0, self.epoch_num, 0, 0)
            for param_group in self.net_optimizer.param_groups:
                param_group['lr'] = self.lr

            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                print("x.size = ", inputs.shape)
                print("y.size = ", targets.shape)
                self.net_optimizer.zero_grad()
                out = self.forward(inputs)
                mean_value_loss = self.calc_gradients(out, targets)
                end = time.time()
                learn_time = end - start
                start = end
        
            # if self.save_interval != -1 and iter_ % self.save_interval == 0:
            #     self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            # if self.play()
                if self.log_dir is not None:
                    self.log(locals())

        return 1

    def calc_gradients(self, batch_out, batch_pred_out):
        loss = self.loss(batch_pred_out, batch_out)
        # for param in self.net.parameters():
        #     param.grad = None
        loss.backward()
        if self.truncate_grads:
            #self.scaler.unscale_(self.net_optimizer)
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg['config']['grad_norm'])
            self.net_optimizer.step()
        else:
            self.net_optimizer.step()
        
        return loss

    def test(self):
        # self.eval()
        test_loss = 0
        # correct = 0
        
        # criterion = nn.BCELoss()

        # with torch.no_grad():
        #     for (inputes, targets) in self.test_loader:
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         outputs = model(inputs).squeeze()
        #         test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
        #         pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
        #         correct += pred.eq(targets.view_as(pred)).sum().item()
        # test_loss /= len(test_loader.dataset)

        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))
        return test_loss

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.net_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
    
    def log(self, locs, width=80, pad=35):
        iteration_time = locs['learn_time']
        self.tot_time += iteration_time
        
        mean_std = torch.tensor(0) #self.alg.actor_critic.std.mean()
        fps = self.batch_size / iteration_time

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['iter_'])
        self.writer.add_scalar('Loss/learning_rate', self.lr, locs['iter_'])
        # self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['iter_'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['iter_'])
        # self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['iter_'])
        self.writer.add_scalar('Perf/learning_time', iteration_time, locs['iter_'])
        
        str = f" \033[1m Learning iteration {locs['iter_']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                    
        
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['iter_'] - self.current_learning_iteration + 1) * (
                               self.current_learning_iteration + locs['num_learning_iterations'] - locs['iter_']):.1f}s\n""")
        print(log_string)
        return

if __name__ == '__main__':
    args = get_args()
    with open(args.cfg_file_path, 'r') as f:
        print(args.cfg_file_path)
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)
    # if args.play or args.test:
    #     cfg_train["params"]["config"]["num_actors"] = cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)
    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())   
    args.device = torch.device(args.device)
    vargs = vars(args)
    print(vargs)
    for key in vargs.keys():
        cfg_train["general"][key] = vargs[key]
    
    print(cfg_train)
    traj_optimizer = ActuatorNet(12*3+12*3, 12, cfg_train)
    

    if args.train:
        traj_optimizer.train()
    # else:
    #     traj_optimizer.play(cfg_train)


