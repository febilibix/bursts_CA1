import numpy as np 
from neuron import PyramidalCells
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d



def boxcar_input(t, t_ons, t_offs, I_max):
    for t_on, t_off in zip(t_ons, t_offs):
        if t_on <= t < t_off:
            return np.array([I_max])
    else: return np.array([0])



def create_plot_one_neuron(ca1):
    v_b, v_a = np.array(ca1.all_values['v_b']), np.array(ca1.all_values['v_a'])
    t_values = ca1.t_values
    bursts, spikes = ca1.burst_count, ca1.spike_count

    fig, ax = plt.subplots(5, 1, figsize=(6, 6), sharex=True)

    # Minimalistic plot style settings
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_linewidth(1.5)
        a.spines['bottom'].set_linewidth(1.5)
        a.grid(False)  # Disable grid
        a.tick_params(direction='out', length=4, width=1.5, colors='black')

    # Voltage plots
    ax[2].plot(t_values, v_b, label='v_b', color='black', linewidth=1.5)
    ax[2].plot(t_values, np.ones(len(t_values)) * ca1.v_th_b, label='v_th', ls='--', color='gray', linewidth=1)
    ax[2].set_ylim(ca1.v_reset_b - 2, ca1.v_th_b + 2)
    ax[2].set_ylabel(r'$V_b$', fontsize=10, color='black')
    ax[2].yaxis.label.set_rotation(0)  # Set horizontal label

    ax[3].plot(t_values, v_a, label='v_a', color='black', linewidth=1.5)
    ax[3].plot(t_values, np.ones(len(t_values)) * ca1.v_th_a, label='v_th', ls='--', color='gray', linewidth=1)
    ax[3].set_ylabel(r'$V_a$', fontsize=10, color='black')
    ax[3].yaxis.label.set_rotation(0)  # Set horizontal label

    # Current plots
    I_b = [ca1.I_b(t) for t in t_values]
    ax[0].plot(t_values, I_b, label='I_b', color='blue', linewidth=1.5)
    ax[0].set_ylabel(r'$I_b$', fontsize=10, color='black')
    ax[0].yaxis.label.set_rotation(0)  # Set horizontal label

    I_a = [ca1.I_a(t) for t in t_values]
    ax[1].plot(t_values, I_a, label='I_a', color='blue', linewidth=1.5)
    ax[1].set_ylabel(r'$I_a$', fontsize=10, color='black')
    ax[1].yaxis.label.set_rotation(0)  # Set horizontal label

    # Spike and Burst plots with neighboring indices handling
    burst_indices = np.where(bursts > 0)[0]
    window = 2  # Define the size of neighboring window (e.g., 2 indices before and after each burst)
    
    burst_with_neighbors = np.zeros_like(bursts, dtype=np.float64)  # Initialize array for burst plotting
    burst_with_neighbors[:] = np.nan  # Set all values to NaN initially

    for idx in burst_indices:
        start = max(0, idx - window)  # Ensure indices don't go below zero
        end = min(len(bursts), idx + window + 1)  # Ensure indices don't go out of bounds
        burst_with_neighbors[start:end] = bursts[start:end]  # Copy burst and neighbors

    # Plot spikes and bursts with neighbors
    ax[4].plot(t_values, spikes[:len(t_values)], label='Spikes', color='black', linewidth=1.5)
    ax[4].plot(t_values, burst_with_neighbors[:len(t_values)], label='Bursts', color='red', linewidth=1.5)
    ax[4].legend(frameon=False, loc='upper right', fontsize=8)
    ax[4].set_yticks([0, 1])
    ax[4].set_ylim(-0, 1.1)

    plt.xlabel(r'$t$', fontsize=10, color='black')
    plt.tight_layout()
    plt.savefig('plots/one_neuron.png', dpi=300)


def create_plot_cos_dist(N_patterns, mean_cos_mean, mean_cos_std, interp_N_patterns, interp_cos_mean, popt1, popt2):

    plt.figure()
    plt.plot(N_patterns, mean_cos_mean, label='Mean Cosine Distance')
    # plt.plot(interp_N_patterns, interp_cos_mean)
    plt.plot(interp_N_patterns, func1(interp_N_patterns, *popt1), 'r--')
    plt.plot(interp_N_patterns, func2(interp_N_patterns, *popt2), 'b--')
    # plt.plot(interp_N_patterns, 5/interp_N_patterns + 0.5, 'g--')
    plt.fill_between(N_patterns, mean_cos_mean - mean_cos_std, mean_cos_mean + mean_cos_std, alpha=0.3, label='Std Dev')
    plt.title("Cosine distances as function of number of patterns")
    plt.xlabel("Number of patterns")
    plt.ylabel("Cosine distance")
    plt.legend()
    plt.savefig('plots/cosine_distances.png', dpi = 300)



def run_one_neuron_and_plot():
    n_cells = {'pyramidal' : 1, 'inter_a' : 0, 'inter_b' : 0, 'CA3' : 0}
    weights = {'pi_a': np.zeros(1), 'pi_b': np.zeros(1), 'ip_a': np.zeros(1), 
               'ip_b': np.zeros(1), 'pp_a': np.zeros(1), 'pp_b' : np.zeros(1), "CA3": np.ones(1)}

    # Setting the simulation time parameters 
    tn = 20
    dt = 0.01
    learning_rate = 0

    ca1 = PyramidalCells(n_cells, weights, learning_rate, dt)

    ca1.spike_count = np.zeros((int(round(tn / dt)), n_cells['pyramidal']))
    ca1.burst_count = np.zeros((int(round(tn / dt)), n_cells['pyramidal']))

    # This i will need to play around with:    
    ca1.I_a = lambda t: boxcar_input(t, [1, 13], [6,  18], 5)
    ca1.I_b = lambda t: boxcar_input(t, [7, 13], [12, 18], 5)

    ca1.run_one_epoch(tn, dt, plasticity = False)

    v_b, v_a = np.array(ca1.all_values['v_b']), np.array(ca1.all_values['v_a'])
    t_values = ca1.t_values

    create_plot_one_neuron(ca1)

    properties = [attr for attr in dir(ca1) if not callable(getattr(ca1, attr)) and not attr.startswith("__")]
    print("Properties of ca1:")
    print(properties)


def func1(x, a, b, c): 
    return a*np.exp(-b*x) + c


def func2(x, a, b, c, d):
    return a/x + b/(x**(1/2)) + c/(x**(1/3)) + d


def get_cos_dist_and_plot():

    df = pd.read_csv('cos_distances.csv')
    N_patterns = df['N_patterns'].values
    mean_cos_mean = df['Mean_cos_mean'].values
    mean_cos_std = df['Mean_cos_std'].values

    # Interpolate between each point for smoother distribution
    interp_func = interp1d(N_patterns, mean_cos_mean, kind='linear')
    interp_N_patterns = np.linspace(min(N_patterns), max(N_patterns), 1000)  # Adjust for desired density
    interp_mean_cos_mean = interp_func(interp_N_patterns)

    popt1, pcov = curve_fit(func1, interp_N_patterns, interp_mean_cos_mean)
    popt2, pcov = curve_fit(func2, interp_N_patterns, interp_mean_cos_mean)

    print(popt1)

    create_plot_cos_dist(N_patterns, mean_cos_mean, mean_cos_std, interp_N_patterns, interp_mean_cos_mean, popt1, popt2)


def main():
    # run_one_neuron_and_plot()
    get_cos_dist_and_plot()


if __name__ == "__main__":
    main()