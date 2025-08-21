import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d, gaussian
import sys 
sys.path.append('../')
from neuron import PyramidalCells
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd

SEED = 1903
np.random.seed(SEED)


ENVIRONMENTS_RUNS = {
    "F1": {'new_env': False, 'top_down': True},
    "F2": {'new_env': False, 'top_down': False},
    "N1": {'new_env': True,  'top_down': False},
    "F3": {'new_env': False, 'top_down': True}, 
    "N2": {'new_env': True,  'top_down': True},
    }


N_CELLS = {
    '1D': {'pyramidal' : 200 , 'inter_a' : 20 , 'inter_b' : 20 , 'CA3' : 120 },
    # '2D': {'pyramidal': 196, 'inter_a': 20, 'inter_b': 20, 'CA3': 121}
    '2D': {'pyramidal' : 289 , 'inter_a' : 30 , 'inter_b' : 30 , 'CA3' : 169 },
    # '2D': {'pyramidal' : 2500, 'inter_a' : 250, 'inter_b' : 250, 'CA3' : 1444}
}

T_EPOCH = 1
SPEED = 20
DT = 0.001
A = 0.3

LEN_TRACK_1D = 100. 
TN_1D = LEN_TRACK_1D/SPEED*32

LEN_EDGE_2D = 50
TN_2D = 250  




def boxcar_input(t, t_ons, t_offs, I_max):
    for t_on, t_off in zip(t_ons, t_offs):
        if t_on <= t < t_off:
            return np.array([I_max])
    else: return np.array([0])


def run_one_neuron():
    n_cells = {'pyramidal' : 1, 'inter_a' : 0, 'inter_b' : 0, 'CA3' : 1}

    # Setting the simulation time parameters 
    tn = 10
    dt = 0.01

    ca1 = PyramidalCells(n_cells, len_edge=100, inh_plasticity=False)
    ca1.W_CA3 = np.ones(1)

    ca1.spike_count = np.zeros((int(round(tn / dt)), n_cells['pyramidal']))
    ca1.burst_count = np.zeros((int(round(tn / dt)), n_cells['pyramidal']))

    # This i will need to play around with:    
    I_a = lambda t: boxcar_input(t, [1, 6], [2, 9], 5)
    I_b = lambda t: boxcar_input(t, [3, 6], [5, 9], 2.2)

    t = np.arange(0, tn+dt, dt)

    ca1.I_a = np.array([I_a(ti) for ti in t]).T
    ca1.I_b = np.array([I_b(ti) for ti in t]).T

    ca1.run_one_epoch(tn, plasticity = False)

    return ca1, t


def cor_act_maps(act_map1, act_map2, which='pv'):

    # TODO: I think there is another implementation for the 2D case, I will need to see how to merge them

    axis = 1 if which == 'pv' else 0
    cor = np.zeros(act_map1.shape[axis])

    for i in range(act_map1.shape[axis]):
        if which == 'pv':
            cor[i] = pearsonr(act_map1[:, i], act_map2[:, i])[0]
        elif which == 'spatial':
            cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]
        else:
            raise ValueError("Parameter 'which' must be either 'pv' or 'spatial'")

    return cor




def cor_act_maps_2d(act_map1, act_map2, which='pv'):
    # act_map1, act_map2 = act_map[out1], act_map[out2]

    if which == 'pv':
        act_map1, act_map2 = act_map1.T, act_map2.T
        # act_map1 = zscore(act_map1, axis=0)
        # act_map2 = zscore(act_map2, axis=0)

        # act_map1 = act_map1/np.sum(act_map1, axis=0)[np.newaxis, :]
        # act_map2 = act_map2/np.sum(act_map2, axis=0)[np.newaxis, :]
        # if np.any(np.isnan(act_map1_new)) or np.any(np.isnan(act_map1_new)):
        #     print("NaN values found in act_map1 or act_map2")
        #     print(out1, out2)
        #     print(act_map1, act_map2)
        #     quit()
        #     print(act_map1, act_map2)
        #     print(np.any(np.isnan(act_map1)), np.any(np.isnan(act_map2)))
        #     print(np.all(np.isnan(act_map1)), np.all(np.isnan(act_map2)))
        # 
        #     cor = np.zeros(act_map1.shape[0])
        # 
        #     for i in range(act_map1.shape[0]):
        #         cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]
        #     
        #     print(cor, np.mean(cor), np.std(cor))
        #     quit()
        
        

    cor = np.zeros(act_map1.shape[0])

    for i in range(act_map1.shape[0]):
        valid_idx = ~np.isnan(act_map1[i, :]) & ~np.isnan(act_map2[i, :])
        if np.any(valid_idx):
            cor[i] = pearsonr(act_map1[i, valid_idx], act_map2[i, valid_idx])[0]
        else:
            cor[i] = np.nan  # Handle case where all values are NaN
        # cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]
       
    return cor


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000, seed=42):
    ## np.random.seed(seed)

    ### TODO: SEE IF THIS IS SAFE, OTHERWISE USE THE FOLLOWING THINGS WHICH ARE COMMENTED OUT
    ### TODO: This i am testing, maybe go back to commented out version
    rng = np.random.default_rng(seed)
    bins = np.arange(0., len_track)
    fps = 1/dt

    x = np.array([])
    i = 0
    while True:
        stopping_time = rng.uniform(0, 1, 2)
        stop1 = np.ones((int(stopping_time[0]*fps),)) * 0.
        # speed = av_running_speed + np.random.randn() * 5
        speed = av_running_speed + rng.normal(0, 5)  # ensure speed is positive
        speed = speed if speed > 0 else av_running_speed # ensure speed is positive
        run_length = len(bins) * fps / speed
        run1 = np.linspace(0., float(len(bins)-1), int(run_length))
        stop2 = np.ones((int(stopping_time[1]*fps),)) * (len(bins)-1.)
        # speed = av_running_speed + np.random.randn() * 5
        speed = av_running_speed + rng.normal(0, 5)
        speed = speed if speed > 0 else av_running_speed # ensure speed is positive
        run_length = len(bins) * fps / speed
        run2 = np.linspace(len(bins)-1., 0., int(run_length))
        x = np.concatenate((x, stop1, run1, stop2, run2))
        if len(x) >= tn*fps:
            break
        i += 1

    x = x[:int(tn*fps)]
    t = np.arange(len(x))/fps

    return t, x



def simulate_2d_run(len_edge=20, av_running_speed=20, dt=0.01, tn=1000, a=np.pi/40, seed=42):
    """
    Simulates a mouse moving in a 2D square environment with smooth, random changes in direction.
    The mouse updates its movement direction gradually and bounces off walls naturally.
    """

    rng = np.random.default_rng(seed)
    fps = 1 / dt  # Frames per second
    total_time_steps = int(tn * fps)
    x_positions, y_positions = [], []
    
    # Initial position and direction
    x, y = len_edge/2, len_edge/2  # Start in the middle
    phi = rng.uniform(0, 2 * np.pi)  # Initial movement direction

    for _ in range(total_time_steps):
        # Update direction with small random change
        phi += rng.normal(0, a)  # Random change in direction

        # Compute new position based on speed and direction
        dx = np.cos(phi) * (av_running_speed + rng.normal(0, 3)) * dt
        dy = np.sin(phi) * (av_running_speed + rng.normal(0, 3)) * dt

        # Check for boundary conditions
        if x + dx >= len_edge:
            x = x - len_edge + dx
        elif x + dx <= 0:
            x = x + len_edge + dx 
        if y + dy >= len_edge:
            y = y - len_edge + dy
        elif y + dy <= 0:
            y = y + len_edge + dy
        
        # Update position
        x = np.clip(x + dx, 0, len_edge)
        y = np.clip(y + dy, 0, len_edge)
        
        x_positions.append(x)
        y_positions.append(y)
    
    t = np.arange(len(x_positions)) / fps
    return t, np.vstack((x_positions, y_positions))



def get_firing_rates_gaussian(dt, event_count, x_run, sigma_s=0.1, n_bins=None, n_dim=1):
    """
    Compute firing rates by convolving spike counts with a Gaussian kernel.

    Parameters
    ----------
    dt : float
        Time step of simulation (s).
    event_count : array, shape (T, N)
        Spike counts per time step (0 or 1 for Poisson-like spiking; can be >1 if multiple spikes).
    x_run : array, shape (n_dim*T,)
        Position trace over time (flattened).
    sigma_s : float
        Standard deviation of Gaussian kernel in seconds.
    n_bins : int or None
        Number of output bins for downsampling (if None, keep full resolution).
    n_dim : int
        Dimensionality of position.

    Returns
    -------
    firing_rates : array, shape (N, n_bins)
    x_run_reshaped : array, shape (n_dim, n_bins)
    """
    T, N = event_count.shape
    sigma_t = sigma_s / dt  # std in time steps
    # kernel length ~ 6 sigma, make odd so convolution is centered
    win_len = int(np.ceil(sigma_t * 6)) | 1  
    kernel = gaussian(win_len, std=sigma_t)
    kernel /= np.sum(kernel) * dt  # normalize so output is in Hz

    # Convolve spikes with Gaussian kernel
    firing_rates_full = np.array([
        np.convolve(event_count[:, i], kernel, mode='same') for i in range(N)
    ])

    # Position: reshape into (n_dim, T)
    x_run = x_run.reshape((n_dim, -1))

    if n_bins is None or n_bins == T:
        return firing_rates_full, x_run

    # Downsample to n_bins by averaging
    step_size = T // n_bins
    firing_rates = np.zeros((N, n_bins))
    x_run_reshaped = np.zeros((n_dim, n_bins))
    for i in range(n_bins):
        print(i)
        firing_rates[:, i] = np.mean(firing_rates_full[:, i*step_size:(i+1)*step_size], axis=1)
        x_run_reshaped[:, i] = np.mean(x_run[:, i*step_size:(i+1)*step_size], axis=1)

    return firing_rates, x_run_reshaped


def get_firing_rates(dt, event_count, x_run, n_bins=1024, n_dim=1):
    ## TODO: Instead of having 1024 hardcoded here i might want to make it dependend on length of simulation

    firing_rates = np.zeros((event_count.shape[1], n_bins))
    x_run_reshaped = np.zeros((n_dim, n_bins))
    step_size = len(event_count)//n_bins
    x_run = x_run.reshape((n_dim, -1)) 
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*dt)
        x_run_reshaped[:, i] = np.mean(x_run[:, i * step_size:(i + 1) * step_size], axis=1)

    return firing_rates, x_run_reshaped


def get_firing_rates_old(pyramidal, event_count, x_run):
    #TODO: Delete this once the one above is working for both 1D and 2D cases   

    firing_rates = np.zeros((event_count.shape[1], 1024))
    x_run_reshaped = np.zeros(1024)
    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)
        x_run_reshaped[i] = np.mean(x_run[i * step_size:(i + 1) * step_size])

    return firing_rates, x_run_reshaped


def get_activation_map_2d(firing_rates, len_edge, x_run_reshaped, n_bins = 225):

    bins = np.arange(n_bins)
    n_cell = np.arange(firing_rates.shape[0])
    out_collector = {k : [] for k in product(n_cell, bins)}
    out = np.zeros((firing_rates.shape[0], n_bins))
    n_edge = int(np.sqrt(n_bins))
    position_bins = np.mgrid[.5:(len_edge-.5):n_edge*1j, .5:(len_edge-.5):n_edge*1j] 
    position_bins = np.vstack((position_bins[0].flatten(), position_bins[1].flatten())).T

    for idx, pos in enumerate(x_run_reshaped.T):
        bin_idx = np.argmin(np.linalg.norm(position_bins - pos, axis = 1)) 

        for i in range(firing_rates.shape[0]):
            out_collector[(i, bin_idx)].append(firing_rates[i, idx])

    for k, v in out_collector.items():
        out[k] = np.mean(v)

    return out, position_bins


def get_activation_map(firing_rates, m_EC, x_run_reshaped, n_bins = 64):

    ## TODO: I think i can do the sorting at the end depending on whether i use EC sorting or peak

    if m_EC is not None:
        sort_TD = np.argsort(m_EC)
        sorted_fr = firing_rates[np.ix_(sort_TD, np.arange(firing_rates.shape[1]))]
    else:
        sorted_fr = firing_rates

    bins = np.arange(n_bins)
    n_cell = np.arange(sorted_fr.shape[0])
    out_collector = {k : [] for k in product(n_cell, bins)}
    out = np.zeros((sorted_fr.shape[0], n_bins))
    position_bins = np.linspace(0, x_run_reshaped.max(), n_bins)

    for idx, pos in enumerate(x_run_reshaped[0, :]):
        bin_idx = np.argmin(np.abs(position_bins - pos))

        for i in range(sorted_fr.shape[0]):
            out_collector[(i, bin_idx)].append(sorted_fr[i, idx])

    for k, v in out_collector.items():
        out[k] = np.mean(v)

    if m_EC is None:
        weighted_vals = out * np.arange(out.shape[1])[np.newaxis, :]
        m_EC = weighted_vals.sum(axis=1) / out.sum(axis=1)
        sort_TD = np.argsort(m_EC) 
        out = out[np.ix_(sort_TD, np.arange(out.shape[1]))]
        return out, m_EC

    return out


def compute_burst_props(brs, frs, window=500):
    """
    Compute burst proportions from burst rates and firing rates.
    """
    brs_out = {}
    for condition in ['exp', 'control']:
        brs_out[condition] = []
        for trial in ['F1', 'F2', 'N1', 'F3', 'N2']:
            m_br = np.mean(brs[condition][trial] / (frs[condition][trial] + 1e-10), axis=0)
            # window = 500
            running_avg = np.convolve(m_br, np.ones(window) / window, mode='valid')
            if condition == 'exp' and trial in ['F2', 'N1']:
                running_avg = np.empty(running_avg.shape)
                running_avg[:] = np.nan  # NaN because no bursts in F2 and N1 in exp condition
            brs_out[condition].extend(running_avg)
    return brs_out


def compute_cors_time_series(all_act_maps, all_act_maps_split):
    all_cors = {}
    for anchor in ['F1', 'N1']:
        all_cors[anchor] = {}

        for condition in ['exp', 'control']:
            all_cors[anchor][condition] = []
            for trial in ['F1', 'F2', 'N1', 'F3', 'N2']:
                
                for i in range(8):
                    am = all_act_maps_split[condition][trial][i]
                    act_map_use = np.where(~np.isnan(am), am, 0)  # Replace NaNs with 0
                    sp_cor = cor_act_maps(all_act_maps[condition][anchor], act_map_use, which='spatial')
                    all_cors[anchor][condition].append(sp_cor)

    return all_cors


def compute_cross_correlogram(act_maps, out1, out2):

    reshape_val = int(np.sqrt(act_maps[out1].shape[1]))
    act_map1 = act_maps[out1].reshape(act_maps[out1].shape[0], reshape_val, reshape_val)
    act_map2 = act_maps[out2].reshape(act_maps[out2].shape[0], reshape_val, reshape_val)
    
    n_cells = act_map1.shape[0]
    cross_corrs = []

    for i in range(n_cells):
        map1 = act_map1[i] - np.mean(act_map1[i])
        map2 = act_map2[i] - np.mean(act_map2[i])
        
        cross_corr = correlate2d(map1, map2, mode='full', boundary='fill')
        cross_corrs.append(cross_corr)
    
    avg_cross_corr = np.mean(cross_corrs, axis=0)
    return avg_cross_corr


def smooth_act_maps(act_maps_exp, act_maps_cont, sigma):
    
    for act_maps in [act_maps_exp, act_maps_cont]:
        for key in act_maps.keys():
            # print(act_maps[key].shape)
            reshape_val = int(np.sqrt(act_maps[key].shape[1]))
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], reshape_val, reshape_val)
            act_maps[key] = gaussian_filter(act_maps[key], sigma=sigma, mode='wrap')
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], -1)
    
    return act_maps_exp, act_maps_cont


def extract_active_cells(act_maps_exp, act_maps_cont):

    act_idxs, avg_acts = [], []

    for act_maps in [act_maps_exp, act_maps_cont]:
        all_trials = np.concatenate(list(act_maps.values()), axis=1)
        avg_act = np.mean(all_trials, axis=1)
        avg_acts.append(avg_act)
        # active_cells = np.where(avg_act > np.percentile(avg_act, 90))[0]
        act_idxs.append(np.argsort(-avg_act))

    rank_a = {idx: rank for rank, idx in enumerate(act_idxs[0])}
    rank_b = {idx: rank for rank, idx in enumerate(act_idxs[1])}

    # Step 3: Combine ranks (sum or average)
    all_indices = np.union1d(act_idxs[0], act_idxs[1])  # Unique indices
    combined_ranks = [
        (idx, rank_a.get(idx, len(avg_acts[0])) + rank_b.get(idx, len(avg_acts[1])))  # Default penalizes missing
        for idx in all_indices
    ]

    # Step 4: Sort by combined rank (lower = better) and pick top 40
    combined_ranks_sorted = sorted(combined_ranks, key=lambda x: x[1])
    active_cells = [idx for idx, rank in combined_ranks_sorted[:40]]

    for act_maps in [act_maps_exp, act_maps_cont]:
        for key in act_maps.keys():
            act_maps[key] = act_maps[key][active_cells, :]

    return act_maps_exp, act_maps_cont



######################## PLOTTING FUNCTIONS ########################


COLOR_SETTINGS = {
    'exp_normal': "#1dde50",  # Vivid green, not too bright
    'exp_no_inh': "#1887d7",         # Original blue
    'exp_no_heb': "#7b1ed2",         # Original purple

    'control_normal': "#448A52",  # Greenish grey
    'control_no_inh': "#486b8d",    # Greyed-down blue
    'control_no_heb': "#5F4585",    # Greyed-down purple
}

def plot_firing_rates(ax, mean_firing_rates, out, vmin=None, vmax=None, fontsize=12, con=''):
    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower', vmin=vmin, vmax=vmax, cmap = 'jet')
    ax.set_title(f"{out}", fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(con, fontsize=fontsize)
    # ax.set_ylabel("Unit", fontsize=fontsize)
    return im


def plot_run(t, x, activity):
    fig, axs = plt.subplots(2,1, figsize = (10,8), dpi = 600, sharex=True)

    extent = [t.min(), t.max(), 0, activity.shape[0],]   

    fig.suptitle(f"")

    axs[0].set_title("Mouse trajectory")
    axs[0].plot(t, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (cm)')

    # axs[1].set_title("CA3 activity")
    # fig.colorbar(axs[1].imshow(activity, aspect='auto', extent=extent), ax=axs[1])
    # axs[1].imshow(activity, aspect='auto', extent=extent)
    # axs[1].set_ylabel("Neuron")
    # axs[1].set_xlabel("Time (s)")

    plt.show()
    plt.close()


def plot_condition(activation_maps, condition):
    fs = 20

    # for condition in all_act_maps.keys():
    fig, axs = plt.subplots(1, 4, figsize=(10, 5), sharey=True, dpi=400)
    axs = axs.flatten()
    # activation_maps = all_act_maps[condition]
    # Find global vmin/vmax for colorbar scaling
    all_maps = [activation_maps[out] for out in ['F2', 'N1', 'F3', 'N2']]
    vmin = min(np.min(m) for m in all_maps)
    vmax = max(np.max(m) for m in all_maps)
    ims = []
    for i, out in enumerate(['F2', 'N1', 'F3', 'N2']):
        im = plot_firing_rates(axs[i], activation_maps[out], out, vmin=vmin, vmax=vmax, fontsize=fs, con=condition)
        ims.append(im)
        if i > 0:
            axs[i].set_ylabel(None)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    # fig.suptitle(condition)
    # Add a single colorbar for all subplots
    cbar = fig.colorbar(ims[0], ax=axs, orientation='vertical', fraction=0.025, pad=0.04)
    cbar.set_label('Firing rate (Hz)', fontsize=fs)
    cbar.ax.tick_params(labelsize=int(fs/1.5))
    # plt.savefig(f"plots/full_experiment/activation_maps/{condition}_act_map.png", dpi=300)
    plt.show()
    plt.close()



def create_plot_one_neuron(ca1, t_values):
    v_b, v_a = np.array(ca1.all_values['v_b']), np.array(ca1.all_values['v_a'])
    bursts, spikes = ca1.burst_count, ca1.spike_count

    # Create figure with 3 subplots (inputs, V_b with spikes, V_a)
    fig, ax = plt.subplots(2, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={'hspace': 0.1, 'height_ratios': [2, 2]}, dpi=600)

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
    ax[0].plot(t_values, I_b_scaled, color='blue', linewidth=1, label=r'$I_B$')
    ax[1].plot(t_values, I_a_scaled, color='blue', linewidth=1, label=r'$I_A$')
    ax[0].set_yticks([])
    ax[0].legend(frameon=False, fontsize=12, loc='upper right', 
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
    ax[1].legend(frameon=False, fontsize=12, loc='upper right', 
               handlelength=1, handletextpad=0.4)
    # ax[2].set_ylim(ca1.pa['E_L'] - 2, ca1.pa['v_th'] + 2)

    # Only show x-axis on bottom plot
    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].spines['bottom'].set_linewidth(0.5)
    ax[-1].set_xticks([])
    # ax[-1].set_xlabel(r'$t$ (ms)', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/one_neuron_clean.png', dpi=300, bbox_inches='tight')


def plot_2d_activity_map(act, len_edge=50):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    act_map = act[np.random.randint(0, act.shape[0]), :].reshape(int(np.sqrt(act.shape[1])), int(np.sqrt(act.shape[1])))

    im = ax.imshow(act_map, cmap='jet', origin='lower', 
                   extent=(0, len_edge, 0, len_edge))

    ax.axis('off')
    plt.show()


def format_val(val, digits=3):
    if abs(val) < 10**(-digits):
        return f"{val:.{1}e}".replace('e', r'\times 10^{') + '}'
    else:
        return f"{val:.{digits}f}" 
    

def plot_pv_corr_distributions(groups, out1, out2, ax=None, p=(1,), colors=('green', '#808080')):
    fs = 20

    # color_exp, color_cont = 'green', '#808080'

    # Plot KDEs (smoothed histograms)
    for i, group in enumerate(groups):
        kde = sns.kdeplot(group, fill=True, color=colors[i], alpha=0.6, linewidth=1.5, ax=ax)
        # kde_cont = sns.kdeplot(pv_corr_cont, fill=True, color=color_cont, alpha=0.6, linewidth=1.5, ax=ax)

        # Plot medians as dashed lines
        if p is not None:
            ax.axvline(np.median(group), color=colors[i], linestyle='--', linewidth=1.5)
        # ax.axvline(np.median(pv_corr_cont), color=color_cont, linestyle='--', linewidth=1.5)

    # Labeling
    ax.set_xlabel('PV corr. coeff.', fontsize=fs)
    ax.set_ylabel('Frequency', fontsize=fs)
    ax.set_title(f'{out1} vs {out2}', fontsize =fs)
    ax.tick_params(labelsize=fs/1.5)
    # ax.legend([f'exp', f'control'], loc='upper right')

    sns.despine()

    if p is None:
        # legend_elements = [
        #     Patch(facecolor=colors[i], label=param_vals[list(param_vals.keys())[0]][i]) for i in range(len(colors))
        # ]
        # ax.legend(handles=legend_elements, title=list(param_vals.keys())[0], loc='upper left', fontsize=fs/1.5)
        return 

    elif len(p) == 1:

        y = ax.get_ylim()[1] * 0.9  # 98% up the y-axis
        xmin, xmax = ax.get_xlim()
        x_width = xmax - xmin
        x = np.mean([np.mean(groups[0]), np.mean(groups[1])]) - 0.1 * x_width   
        col = 'black'
        text = '***' if p[0] < 0.001 else '**' if p[0] < .01  else '*' if p[0] < .05 else 'n.s.'
        ax.text(x, y, text, ha='center', va='bottom', color=col)
        

    elif len(p) == 2:
        text1 = '***' if p[0] < 0.001 else '**' if p[0] < .01  else '*' if p[0] < .05 else 'n.s.'
        text2 = '***' if p[1] < 0.001 else '**' if p[1] < .01  else '*' if p[1] < .05 else 'n.s.'
        legend_elements = [
            Patch(facecolor=colors[1], label=text1),
            Patch(facecolor=colors[2], label=text2),
            ]

        ax.legend(handles=legend_elements, title="", loc='upper left')
        # ax.legend([f'{colors[0]} vs {colors[1]}', f'{colors[0]} vs {colors[2]}'], loc='upper right', fontsize=fs/1.5)

    


def create_raincloud_plot(groups, out1, out2, ax, p=(1,), colors=('green', '#808080')):
    fs = 20

    conditions = []

    for i, group in enumerate(groups):
        for j in range(len(group)):
            conditions.append(f'group{i}')

    data = {
        'Condition': conditions,
        'Spatial correlation': np.concatenate(groups)
    }
    
    df = pd.DataFrame(data)
    palette = colors

    # Raincloud plot with violin, box, and strip
    for i, condition in enumerate(df['Condition'].unique()):
        df_use = df[df['Condition'] == condition]
        sns.violinplot(data=df_use, y='Spatial correlation', inner=None, linewidth=2, color=palette[i], split = True, alpha = 0.3, linecolor=palette[i], ax=ax)
    
    df['x'] = ' '
    sns.boxplot(data=df, x = 'x', hue='Condition', y='Spatial correlation', whis=[0, 100], width=0.8, gap = 0.5, palette=palette, fill = False, ax=ax)
    sns.stripplot(data=df, x= 'x', hue='Condition', y='Spatial correlation', jitter=0.2, palette=palette, alpha=0.1, dodge=True, size=3, ax=ax)

    # Add significance bar
    y, h, col = 1.1, 0.05, 'black'
    if len(groups) == 2:
        x1, x2 = 0.8, 1.2
    else:
        x1, x2, x3 = 0.733, 1.0, 1.267

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    text1 = '***' if p[0] < 0.001 else '**' if p[0] < .01  else '*' if p[0] < .05 else 'n.s.'
    ax.text((x1 + x2) * 0.5, y + h + 0.02, text1, ha='center', va='bottom', color=col)

    if len(groups) == 3:
        text2 = '***' if p[1] < 0.001 else '**' if p[1] < .01  else '*' if p[1] < .05 else 'n.s.'
        ax.plot([x1, x1, x3, x3], [y+.2, y+h+.2, y+h+.2, y+.2], lw=1.5, c=col)
        ax.text((x1 + x3) * 0.5, y + h + 0.22, text2, ha='center', va='bottom', color=col)

    # Customize axes
    ax.hlines(0, color='grey', lw=0.5, xmin=-1, xmax=1.5)

    ax.set_title(f'{out1} vs {out2}', fontsize=fs)
    ax.set_ylabel("Spatial correlation", fontsize=fs)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.tick_params(labelsize=fs/1.5)
    if len(groups) == 2:
        ax.set_ylim(-1.1, 1.3)
    else:
        ax.set_ylim(-1.1, 1.5)
    ax.set_xlim(-0.5, 1.5)
    sns.despine(ax=ax)
    ax.legend_.remove()
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/act_maps_exp_cont_{out1}_{out2}.png', dpi=300)
    # plt.close()

    
def plot_pv_corr_distributions_old(pv_corr_exp, pv_corr_cont, out1, out2, ax=None, p=1):
        ## TODO: DELETE THIS FUNCTION

    fs = 20

    color_exp, color_cont = 'green', '#808080'

    # Plot KDEs (smoothed histograms)
    kde_exp = sns.kdeplot(pv_corr_exp, fill=True, color=color_exp, alpha=0.6, linewidth=1.5, ax=ax)
    kde_cont = sns.kdeplot(pv_corr_cont, fill=True, color=color_cont, alpha=0.6, linewidth=1.5, ax=ax)

    # Plot medians as dashed lines
    ax.axvline(np.median(pv_corr_exp), color=color_exp, linestyle='--', linewidth=1.5)
    ax.axvline(np.median(pv_corr_cont), color=color_cont, linestyle='--', linewidth=1.5)

    # Labeling
    ax.set_xlabel('PV corr. coeff.', fontsize=fs)
    ax.set_ylabel('Frequency', fontsize=fs)
    ax.set_title(f'{out1} vs {out2}', fontsize =fs)
    ax.tick_params(labelsize=fs/1.5)
    # ax.legend([f'exp', f'control'], loc='upper right')

    # Use axis coordinates to place the significance star just above the top of the plot
    y = ax.get_ylim()[1] * 0.9  # 98% up the y-axis
    xmin, xmax = ax.get_xlim()
    x_width = xmax - xmin
    x = np.mean([np.mean(pv_corr_exp), np.mean(pv_corr_cont)]) - 0.1 * x_width 

    col = 'black'
    text = '***' if p < 0.001 else '**' if p < .01  else '*' if p < .05 else 'n.s.'
    ax.text(x, y, text, ha='center', va='bottom', color=col)
    sns.despine()


def create_raincloud_plot_old(cor_exp, cor_cont, out1, out2, ax, p=1):
    ## TODO: DELETE THIS FUNCTION
    fs = 20


    data = {
        'Condition': ['exp'] * len(cor_exp) + ['cont'] * len(cor_cont),
        'Spatial correlation': np.concatenate((cor_exp, cor_cont))
    }
    
    df = pd.DataFrame(data)
    palette = ['green', '#808080']

    # Raincloud plot with violin, box, and strip
    for i, condition in enumerate(['exp', 'cont']):
        df_use = df[df['Condition'] == condition]
        sns.violinplot(data=df_use, y='Spatial correlation', inner=None, linewidth=2, color=palette[i], split = True, alpha = 0.3, linecolor=palette[i], ax=ax)
    
    df['x'] = ' '
    sns.boxplot(data=df, x = 'x', hue='Condition', y='Spatial correlation', whis=[0, 100], width=0.8, gap = 0.5, palette=palette, fill = False, ax=ax)
    sns.stripplot(data=df, x= 'x', hue='Condition', y='Spatial correlation', jitter=0.2, palette=palette, alpha=0.1, dodge=True, size=3, ax=ax)

    # Add significance bar
    x1, x2 = 0.8, 1.2
    y, h, col = 1.1, 0.05, 'black'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    text = '***' if p < 0.001 else '**' if p < .01  else '*' if p < .05 else 'n.s.'
    ax.text((x1 + x2) * 0.5, y + h + 0.02, text, ha='center', va='bottom', color=col)

    # Customize axes
    ax.hlines(0, color='grey', lw=0.5, xmin=-1, xmax=1.5)



    ax.set_title(f'{out1} vs {out2}', fontsize=fs)
    ax.set_ylabel("Spatial correlation", fontsize=fs)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.tick_params(labelsize=fs/1.5)
    ax.set_ylim(-1.1, 1.3)
    ax.set_xlim(-0.5, 1.5)
    sns.despine(ax=ax)
    ax.legend_.remove()
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/act_maps_exp_cont_{out1}_{out2}.png', dpi=300)
    # plt.close()


def plot_w_ca3(w_ca3, m_EC_orig, m_CA3_orig):
    plt.figure(figsize=(10, 6))
    sort_EC = np.argsort(m_EC_orig)
    sort_CA3 = np.argsort(m_CA3_orig)
    w_ca3 = w_ca3[np.ix_(sort_EC, sort_CA3)]
    im = plt.imshow(w_ca3, aspect='auto', origin='lower')
    plt.colorbar(im)
    plt.title('W_CA3')
    plt.xlabel('Pyramidal Cells')
    plt.ylabel('CA3 Cells')

    plt.tight_layout()
    plt.savefig(f"../../plots/full_experiment/weights/w_ca3.png", dpi=300)


def plot_burst_props(burst_props, tn, ax):

    for condition in ['exp', 'control']:
        burst_props[condition] = np.array(burst_props[condition])
        t = np.linspace(0, tn, burst_props[condition].shape[0])
        ax.plot(t, burst_props[condition], color=COLOR_SETTINGS[f'{condition}_normal']) 

    ax.set_ylabel('Burst Proportion', fontsize = 20 ) 
    ax.set_yticks([0.1, 0.14, 0.18, 0.22])
    ax.set_yticklabels(['.1', '.14', '.18', '.22'], fontsize=15)


def plot_cor_time_series(all_cors, tn, axs):
    """
    Plot the spatial correlation time series for each anchor and condition.
    """
    for idx, anchor in enumerate(['F1', 'N1']):

        for condition in ['exp', 'control']:
            cors = all_cors[anchor][condition]
            
            if len(cors) == 0:
                continue  # Skip if no correlations are found for this anchor
            
            t = np.linspace(0, tn, len(np.mean(all_cors[anchor][condition], axis=1)))

            axs[idx].plot(t, np.mean(cors, axis=1), color=COLOR_SETTINGS[f'{condition}_normal'])
            axs[idx].fill_between(t, 
                            np.mean(cors, axis=1) - np.std(cors, axis=1), 
                            np.mean(cors, axis=1) + np.std(cors, axis=1), 
                            alpha=0.2, color=COLOR_SETTINGS[f'{condition}_normal'])

            axs[idx].set_yticks([-.5, 0, .5, 1])
            axs[idx].set_yticklabels(['-0.5', '0', '0.5', '1'], fontsize=15)
            axs[idx].set_ylabel(f'Spatial Correlation \n {anchor} anchor', fontsize=20)



def plot_single_maps(act_maps_exp, act_maps_cont, m_ECs, idx_sorted= None):
    if idx_sorted is None:
        idx_sorted = np.arange(act_maps_exp['F1'].shape[0])
    for i in idx_sorted:

    
        fig = plt.figure(figsize=(12, 6))

        # Left 2x2 block (Experimental)
        left_gs = plt.GridSpec(2, 2, left=0.05, right=0.45, wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(left_gs[0, 0])
        ax2 = fig.add_subplot(left_gs[0, 1])
        ax3 = fig.add_subplot(left_gs[1, 0])
        ax4 = fig.add_subplot(left_gs[1, 1])

        # Right 2x2 block (Control)
        right_gs = plt.GridSpec(2, 2, left=0.55, right=0.95, wspace=0.3, hspace=0.3)
        ax5 = fig.add_subplot(right_gs[0, 0])
        ax6 = fig.add_subplot(right_gs[0, 1])
        ax7 = fig.add_subplot(right_gs[1, 0])
        ax8 = fig.add_subplot(right_gs[1, 1])

        axes1 = [ax1, ax2, ax3, ax4]  # Left block axes
        axes2 = [ax5, ax6, ax7, ax8]  # Right block axes
        fig.suptitle('cell no. {}'.format(i))

        # Add left/right titles
        fig.text(0.25, 0.95, "Experimental Group", ha='center', fontsize=14, weight='bold')
        fig.text(0.75, 0.95, "Control Group", ha='center', fontsize=14, weight='bold')

        for j, act_maps in enumerate([act_maps_exp, act_maps_cont]):
            axes = axes1 if j == 0 else axes2
            for k, out in enumerate(['F2', 'N1', 'F3', 'N2']):
                plot_one_2d_map(axes[k], act_maps[out][i, :], out)


        # Manually adjust layout to avoid colorbar overlap
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)
        plt.savefig(f'simulations/data/single_cells/cell_{i}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()


def plot_one_2d_map(ax, act_map, out=None):
    act_map = act_map.reshape(int(np.sqrt(act_map.shape[0])), 
                                int(np.sqrt(act_map.shape[0])))

    im = ax.imshow(
        gaussian_filter(act_map, sigma=1.5),  # Smooth with Gaussian filter
        cmap='jet',
        origin='lower',
        extent=(0, LEN_EDGE_2D, 0, LEN_EDGE_2D),
        interpolation='bilinear'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if out is not None:
        ax.set_title(out, fontsize=20)

    # Add colorbar (with adjusted padding)
       
    # ax.set_title(f'{out}: x = {round(m_ECs[idx][0, i],1)}, y = {round(m_ECs[idx][1, i], 1)}', fontsize=12)
    # ax.axis('off')

    return im 


def plot_cross_correlogram(act_maps, out1, out2, ax, title=None, cmap='jet', sigma=1.5):

    correlogram = compute_cross_correlogram(act_maps, out1, out2)

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(correlogram, sigma=sigma)

    # Plot
    # fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(smoothed, cmap=cmap, origin='lower', interpolation='bilinear')

    # Crosshairs at center
    center_y, center_x = np.array(smoothed.shape) // 2
    ax.axhline(center_y, color='black', linewidth=2)
    ax.axvline(center_x, color='black', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=20)

    # ax.axis('off')
    # plt.tight_layout()
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/cross_cor_exp_test.png', dpi=300)

