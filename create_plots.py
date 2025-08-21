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


def create_plot_one_neuron(ca1, t_values):
    v_b, v_a = np.array(ca1.all_values['v_b']), np.array(ca1.all_values['v_a'])
    bursts, spikes = ca1.burst_count, ca1.spike_count

    # Create figure with 3 subplots (inputs, V_b with spikes, V_a)
    fig, ax = plt.subplots(2, 1, figsize=(6, 2), sharex=True, 
                         gridspec_kw={'hspace': 0.1, 'height_ratios': [2, 2]})

    I_b_scaled = ca1.I_b.T * 1 + ca1.pb['E_L']   # Adjust scale & offset
    I_a_scaled = ca1.I_a.T * 1 + ca1.pa['E_L']

    # Common style settings - minimalistic
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.tick_params(axis='both', which='both', length=3, width=0.5)
        a.set_facecolor('none')
        a.set_ylabel('')  # Remove y labels
    
    # Combined input plot
    ax[0].plot(t_values, I_b_scaled, color='blue', linewidth=1, label=r'$I_b$')
    ax[1].plot(t_values, I_a_scaled, color='blue', linewidth=1, label=r'$I_a$')
    ax[0].set_yticks([])
    ax[0].legend(frameon=False, fontsize=7, loc='upper right', 
               handlelength=1, handletextpad=0.4)

    # V_b plot with spikes and bursts above
    ax[0].plot(t_values, v_b, color='black', linewidth=1)
    ax[0].plot(t_values, np.ones(len(v_b)) * ca1.pb['v_th'], 
             ls=':', color='gray', linewidth=0.8)
    # ax[0].set_ylim(ca1.pb['E_L'] - 2, ca1.pb['v_th'] + 2)
    ax[0].set_yticks([])

    # Add spikes and bursts as markers above V_b
    spike_y = ca1.pb['v_th'] +2   # Just above threshold
    burst_y = ca1.pb['v_th'] +2 # Slightly higher than spikes
    
    # Convert spike times to marker positions
    spike_times = t_values[np.where(spikes > 0)[0]]
    burst_times = t_values[np.where(bursts > 0)[0]]
    
    ax[0].plot(spike_times, np.ones_like(spike_times) * spike_y, 
             '|', color='black', markersize=20)
    ax[0].plot(burst_times, np.ones_like(burst_times) * burst_y, 
             '|', color='red', markersize=20)

    # V_a plot
    ax[1].plot(t_values, v_a, color='black', linewidth=1)
    ax[1].plot(t_values, np.ones(len(v_a)) * ca1.pa['v_th'], 
             ls=':', color='gray', linewidth=0.8)
    ax[1].set_yticks([])
    ax[1].legend(frameon=False, fontsize=7, loc='upper right', 
               handlelength=1, handletextpad=0.4)
    # ax[2].set_ylim(ca1.pa['E_L'] - 2, ca1.pa['v_th'] + 2)

    # Only show x-axis on bottom plot
    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].spines['bottom'].set_linewidth(0.5)
    ax[-1].set_xticks([])
    # ax[-1].set_xlabel(r'$t$ (ms)', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plots/one_neuron_clean.png', dpi=300, bbox_inches='tight')

def create_plot_cos_dist(N_patterns, mean_cos_mean, mean_cos_std, interp_N_patterns, interp_cos_mean, popt1, popt2):

    plt.figure()
    plt.plot(N_patterns, mean_cos_mean, label='Mean Cosine Distance')
    # plt.plot(interp_N_patterns, interp_cos_mean)
    # plt.plot(interp_N_patterns, func1(interp_N_patterns, *popt1), 'r--')
    # plt.plot(interp_N_patterns, func2(interp_N_patterns, *popt2), 'b--')
    # plt.plot(interp_N_patterns, 5/interp_N_patterns + 0.5, 'g--')
    plt.fill_between(N_patterns, mean_cos_mean - mean_cos_std, mean_cos_mean + mean_cos_std, alpha=0.3, label='Std Dev')
    plt.title("Cosine distances as function of number of patterns")
    plt.xlabel("Number of patterns")
    plt.ylabel("Cosine distance")
    plt.legend()
    plt.savefig('plots/cosine_distances.png', dpi = 300)



def run_one_neuron_and_plot():
    n_cells = {'pyramidal' : 1, 'inter_a' : 0, 'inter_b' : 0, 'CA3' : 1}
    weights = {'pi_a': np.zeros(1), 'pi_b': np.zeros(1), 'ip_a': np.zeros(1), 
               'ip_b': np.zeros(1), 'pp_a': np.zeros(1), 'pp_b' : np.zeros(1), "CA3": np.ones(1)}

    # Setting the simulation time parameters 
    tn = 10
    dt = 0.01
    learning_rate = 0

    ca1 = PyramidalCells(n_cells, len_track=100)
    ca1.W_CA3 = np.ones(1)

    ca1.spike_count = np.zeros((int(round(tn / dt)), n_cells['pyramidal']))
    ca1.burst_count = np.zeros((int(round(tn / dt)), n_cells['pyramidal']))

    # This i will need to play around with:    
    I_a = lambda t: boxcar_input(t, [1, 6], [2, 9], 5)
    I_b = lambda t: boxcar_input(t, [3, 6], [5, 9], 2.2)

    t = np.arange(0, tn+dt, dt)

    ca1.I_a = np.array([I_a(ti) for ti in t]).T
    ca1.I_b = np.array([I_b(ti) for ti in t]).T

    print(ca1.I_a.shape, ca1.I_b.shape)

    ca1.run_one_epoch(tn, plasticity = False)

    # v_b, v_a = np.array(ca1.all_values['v_b']), np.array(ca1.all_values['v_a'])
    # t_values = ca1.t_values

    create_plot_one_neuron(ca1, t)

    properties = [attr for attr in dir(ca1) if not callable(getattr(ca1, attr)) and not attr.startswith("__")]
    print("Properties of ca1:")
    print(properties)


def func1(x, a, b, c): 
    return a/x**.5 + b


# def func2(x, a, b, c, d):
#     return  c/(x**(1/3)) + d


def get_cos_dist_and_plot():

    df = pd.read_csv('cos_distances_50_cells.csv')
    N_patterns = df['N_patterns'].values
    mean_cos_mean = df['Mean_cos_mean'].values
    mean_cos_std = df['Mean_cos_std'].values

    # Interpolate between each point for smoother distribution
    interp_func = interp1d(N_patterns, mean_cos_mean, kind='linear')
    interp_N_patterns = np.linspace(min(N_patterns), max(N_patterns), 1000)  # Adjust for desired density
    interp_mean_cos_mean = interp_func(interp_N_patterns)

    popt1, pcov = curve_fit(func1, interp_N_patterns, interp_mean_cos_mean)
    # popt2, pcov = curve_fit(func2, interp_N_patterns, interp_mean_cos_mean)
    popt2 = None
    print(popt1)

    create_plot_cos_dist(N_patterns, mean_cos_mean, mean_cos_std, interp_N_patterns, interp_mean_cos_mean, popt1, popt2)


def plot_multiple_cos_dist():
    plt.figure()

    for n_cells in [50, 100, 200]:
        df = pd.read_csv(f'cos_distances_{n_cells}_cells.csv')
        N_patterns = df['N_patterns'].values
        mean_cos_mean = df['Mean_cos_mean'].values
        mean_cos_std = df['Mean_cos_std'].values

        plt.plot(N_patterns, mean_cos_mean, label=f'{n_cells} cells')
        plt.fill_between(N_patterns, mean_cos_mean - mean_cos_std, mean_cos_mean + mean_cos_std, alpha=0.3)

    plt.legend()
    plt.savefig('plots/cosine_distances_multiple.png', dpi = 300)


def main():
    run_one_neuron_and_plot()
    # get_cos_dist_and_plot()
    # plot_multiple_cos_dist()


if __name__ == "__main__":
    main()