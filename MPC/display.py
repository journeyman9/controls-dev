import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import argparse
import pdb

if __name__ == "__main__":
    df_ref = pd.read_csv("ref_trajectory.csv")
    df_motion = pd.read_csv("motion.csv")

    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

    # Plot psi_1, psi_2, y2 from ref_trajectory.csv
    axs[0].plot(df_ref['psi_1'], label='psi_1_r', linestyle='--')
    axs[0].plot(df_motion['x0'], label='x0')
    axs[0].legend()

    axs[1].plot(df_ref['psi_2'], label='psi_2_r', linestyle='--')
    axs[1].plot(df_motion['x1'], label='x1')
    axs[1].legend()

    axs[2].plot(df_ref['y2'], label='y2_r', linestyle='--')
    axs[2].plot(df_motion['x2'], label='x2')
    axs[2].legend()

    # Plot u from motion.csv
    axs[3].plot(df_motion['u'], label='u')
    axs[3].set_xlabel('Time')
    axs[3].legend()

    plt.tight_layout()
    plt.show()