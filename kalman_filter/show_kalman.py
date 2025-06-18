import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import argparse
import pdb

if __name__ == "__main__":
    df_kalman = pd.read_csv("kalman.csv")
    # Columns x_hat_0, x_hat_1, x_hat_2, x_0, x_1, x_2, trace_Kf, trace_P

    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    # Plot x_hat_0 - x_0, x_hat_1 - x_1, x_hat_2 - x_2, trace_Kf, trace_P over iterations 
    axs[0].plot(df_kalman['x_hat_0'] - df_kalman['x_0'], label='x_hat_0 - x_0', linestyle='-')
    axs[1].plot(df_kalman['x_hat_1'] - df_kalman['x_1'], label='x_hat_1 - x_1', linestyle='-')
    axs[2].plot(df_kalman['x_hat_2'] - df_kalman['x_2'], label='x_hat_2 - x_2', linestyle='-')
    axs[3].plot(df_kalman['trace_Kf'], label='trace_Kf', linestyle='-')
    axs[4].plot(df_kalman['trace_P'], label='trace_P', linestyle='-')

    # Set labels and legends
    axs[0].set_ylabel('x_hat_0 - x_0')
    axs[0].legend()
    axs[1].set_ylabel('x_hat_1 - x_1')
    axs[1].legend()
    axs[2].set_ylabel('x_hat_2 - x_2')
    axs[2].legend()
    axs[3].set_ylabel('trace_Kf')
    axs[3].legend()
    axs[4].set_ylabel('trace_P')
    axs[4].set_xlabel('Iterations')
    axs[4].legend()

    plt.tight_layout()
    plt.show()