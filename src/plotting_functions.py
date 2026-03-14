import matplotlib.pyplot as plt
import numpy as np

def plot_value_evolution(compilation_runs, label, ax, window=20, color='blue'):
    """
    Plots the evolution of portfolio values across episodes for multiple runs.
    
    Parameters:
    - compilation_runs: List of lists, where each inner list contains the final portfolio values for each episode in a run.
    - label: Label for the plot legend.
    - ax: Matplotlib axis to plot on.
    - window: Window size for moving average smoothing.
    - color: Color for the plot.
    """
    # shape of compilation_runs is (n_runs, n_episodes)
    compilation_runs_array = np.array(compilation_runs)  # Convert to numpy array for easier calculations

    # moving average per run, then average across runs
    mov_avg = np.array([np.convolve(run, np.ones(window)/window, mode='valid') for run in compilation_runs_array])
    mean_mov_avg = mov_avg.mean(axis=0)
    std_mov_avg = mov_avg.std(axis=0)
    x = range(window - 1, window - 1 + len(mean_mov_avg))  # Adjust x-axis for smoothing

    ax.plot(x, mean_mov_avg, label=label, color=color)
    ax.fill_between(x, mean_mov_avg - std_mov_avg, mean_mov_avg + std_mov_avg, alpha=0.2, color=color)
