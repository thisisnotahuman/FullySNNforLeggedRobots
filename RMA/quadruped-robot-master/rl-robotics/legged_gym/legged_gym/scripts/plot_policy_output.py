import os
from tkinter import font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

run_dir = 'legged_gym/logs/rough_a1/'
policy_output_file = os.path.join(run_dir, 'policy_outputs.csv')

delta_phi_idx_to_name = {'0': 'FL', '1': 'FR', '2': 'RL', '3': 'RR'}
residual_angle_idx_to_name = {'4': 'FL_hip', '5': 'FL_thigh', '6': 'FL_calf',
                              '7': 'FR_hip', '8': 'FR_thigh', '9': 'FR_calf',
                              '10': 'RL_hip', '11': 'RL_thigh', '12': 'RL_calf',
                              '13': 'RR_hip', '14': 'RR_thigh', '15': 'RR_calf'}

def plot_policy_output_csv(csv_file: str, col_idx_list: list, figure_title: str):
    legend_list = []
    for c in col_idx_list:
        if figure_title == 'delta_phi':
            legend_list.append(delta_phi_idx_to_name[c])
        elif figure_title == 'residual_angle':
            legend_list.append(residual_angle_idx_to_name[c])
        else:
            return
    origin_data = pd.read_csv(csv_file)
    plt.figure()
    plt.title('policy output - '+figure_title, fontsize=20)
    for col_idx in col_idx_list:
        plt.plot(origin_data[col_idx])
    plt.legend(legend_list, fontsize=15)
    plt.savefig(os.path.join(run_dir, figure_title+legend_list[0][2:]+'.png'))
    plt.show()

def main():
    figure_title = 'delta_phi'
    col_idx = ['0', '1', '2', '3']

    # figure_title = 'residual_angle'
    # col_idx = ['5', '8', '11', '14'] # thigh
    # col_idx = ['6', '9', '12', '15'] # calf

    plot_policy_output_csv(policy_output_file, col_idx, figure_title)


def test():
    k1 = np.array(np.arange(0.0, 1.0, 0.01))
    print(k1)
    f1 = -2 *np.power(k1, 3)+3*np.power(k1, 2)
    k2 = k1 + 1.0
    f2 = 2*np.power(k2, 3) - 9 * np.power(k2, 2) + 12*k2 - 4
    # plt.plot(k1, f1)
    plt.plot(k2, f2)
    plt.show()

if __name__ == '__main__':
    # main()

    test()