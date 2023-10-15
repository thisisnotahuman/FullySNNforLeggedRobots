import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

reward_file = 'legged_gym/logs/rough_a1/Mar26_10-40-05_/rewards.csv'

DT = 0.012 # sim_params.dt * control.decimation

reward_scales = {'rew_action_rate': -0.01, 
                 'rew_ang_vel_xy': -0.05,
                 'rew_collision': -1.0,
                 'rew_dof_acc': -2.5e-7,
                 'rew_dof_pos_limits': -10.0,
                 'rew_feet_air_time': 1.0,
                 'rew_lin_vel_z': -2.0,
                 'rew_torques': -0.0002,
                 'rew_tracking_ang_vel': 0.5,
                 'rew_tracking_lin_vel': 1.0}

header = list(reward_scales.keys())

def plot_reward_csv(csv_file, dt):
    origin_data = pd.read_csv(csv_file)
    plt.subplot(3, 1, 1)
    plt.title('origin_reward')
    for rew in header:
        plt.plot(origin_data[rew])
    plt.legend(header)

    scaled_data = origin_data.copy(deep=True)
    rew_tol = pd.DataFrame(data=[0]*scaled_data.shape[0])
    for rew in header:
        scaled_data[rew] *= reward_scales[rew]*dt
        rew_tol[0] += scaled_data[rew]

    plt.subplot(3, 1, 2)
    plt.title('scaled_reward')
    for rew in header:
        plt.plot(scaled_data[rew])
    plt.legend(header)

    plt.subplot(3, 1, 3)
    plt.title('total_reward')
    plt.plot(origin_data['total_reward'])

    plt.show()

def save_reward_csv(csv_file, dt):
    origin_data = pd.read_csv(csv_file)
    plt.subplot(3, 1, 1)
    plt.title('origin_reward')
    for rew in header:
        plt.plot(origin_data.get(rew, 0))
    plt.legend(header)

    scaled_data = origin_data.copy(deep=True)
    rew_tol = pd.DataFrame(data=[0]*scaled_data.shape[0])
    for rew in header:
        scaled_data[rew] = scaled_data.get(rew, 0) * reward_scales.get(rew, 0)*dt
        rew_tol[0] += scaled_data.get(rew, 0)

    plt.subplot(3, 1, 2)
    plt.title('scaled_reward')
    for rew in header:
        plt.plot(scaled_data[rew])
    plt.legend(header)

    plt.subplot(3, 1, 3)
    plt.title('total_reward')
    plt.plot(origin_data['total_reward'])

    plt.savefig(os.path.join(os.path.dirname(csv_file), 'rewards.png'))

def main():
    plot_reward_csv(reward_file, dt=DT)

if __name__ == '__main__':
    main()