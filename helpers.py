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
    '2D': {'pyramidal' : 289 , 'inter_a' : 30 , 'inter_b' : 30 , 'CA3' : 169 },
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
    """Return a one-element array with I_max during any [t_on, t_off) window, else 0.

    Args:
        t: Scalar time (s).
        t_ons: Iterable of start times for boxcar windows (s).
        t_offs: Iterable of end times for boxcar windows (s); must align with t_ons.
        I_max: Amplitude of the boxcar input (arbitrary units).

    Returns:
        np.ndarray: Shape (1,) with the instantaneous input.
    """
   
    for t_on, t_off in zip(t_ons, t_offs):
        if t_on <= t < t_off:
            return np.array([I_max])
    else: return np.array([0])


def run_one_neuron():
    """Run a toy two-compartment simulation for a single CA1 neuron.

    Sets up a single pyramidal cell with fixed CA3 input and two boxcar current
    drives (to basal/apical compartments), runs one epoch without plasticity,
    and returns the simulator object and time vector.

    Returns:
        ca1: PyramidalCells instance after simulation.
        t: 1D array of time points (s).
    """

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

    """Compute per-row/column Pearson correlation across two 1D activation maps.

    Args:
        act_map1: Array (n_cells, n_bins) or (n_bins, n_cells) depending on `which`.
        act_map2: Same shape as act_map1.
        which: 'pv' correlates population vectors across position (axis=1);
               'spatial' correlates spatial bins across cells (axis=0).

    Returns:
        np.ndarray: 1D array of correlations with length equal to the iterated axis.
    """

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

    ### TODO: Merge this with function above

    if which == 'pv':
        act_map1, act_map2 = act_map1.T, act_map2.T
        
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
    """Simulate 1D back-and-forth runs with random speeds and stop epochs.

    Motion cycles: stop at 0 → run to end → stop at end → run back to 0, repeat.

    Args:
        len_track: Track length (cm).
        av_running_speed: Mean speed (cm/s).
        dt: Time step (s).
        tn: Total duration (s).
        seed: RNG seed.

    Returns:
        t: 1D array of time (s).
        x: 1D array of position samples in [0, len_track) with length ≈ tn/dt.
    """

    rng = np.random.default_rng(seed)
    bins = np.arange(0., len_track)
    fps = 1/dt

    x = np.array([])
    i = 0
    while True:
        stopping_time = rng.uniform(0, 1, 2)
        stop1 = np.ones((int(stopping_time[0]*fps),)) * 0.
        speed = av_running_speed + rng.normal(0, 5)  
        speed = speed if speed > 0 else av_running_speed # ensure speed is positive
        run_length = len(bins) * fps / speed
        run1 = np.linspace(0., float(len(bins)-1), int(run_length))
        stop2 = np.ones((int(stopping_time[1]*fps),)) * (len(bins)-1.)
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
    """Simulate smooth 2D motion on a torus-like square with gradual heading changes.

    Heading increments follow N(0, a). Positions wrap at boundaries, then clip.

    Args:
        len_edge: Edge length of square environment (cm).
        av_running_speed: Mean speed (cm/s).
        dt: Time step (s).
        tn: Total duration (s).
        a: Std of per-step heading change (rad).
        seed: RNG seed.

    Returns:
        t: 1D array of time (s).
        pos: (2,T) array with rows [x; y] in cm.
    """

    rng = np.random.default_rng(seed)
    fps = 1 / dt  
    total_time_steps = int(tn * fps)
    x_positions, y_positions = [], []
    
    x, y = len_edge/2, len_edge/2  # Start in the middle
    phi = rng.uniform(0, 2 * np.pi)  

    for _ in range(total_time_steps):
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


def get_firing_rates(dt, event_count, x_run, n_bins=1024, n_dim=1):
    ## TODO: Instead of having 1024 hardcoded here i might want to make it dependend on length of simulation
    ### TODO: I need to check what i am doing now but i will just use the implementation that worked
    """Bin spikes to firing rates and downsample trajectory to the same bins.

    Args:
        dt: Simulation time step used for event_count (s).
        event_count: Array (T, n_cells) of spike counts per time step.
        x_run: Array (T,) for 1D or (n_dim, T) for 2D trajectory samples.
        n_bins: Number of temporal bins.
        n_dim: Dimensionality of x_run (1 or 2).

    Returns:
        firing_rates: (n_cells, n_bins) mean spikes/s per bin.
        x_run_reshaped: (n_dim, n_bins) averaged positions per bin.
    """

    firing_rates = np.zeros((event_count.shape[1], n_bins))
    x_run_reshaped = np.zeros((n_dim, n_bins))
    step_size = len(event_count)//n_bins
    x_run = x_run.reshape((n_dim, -1)) 
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*dt)
        x_run_reshaped[:, i] = np.mean(x_run[:, i * step_size:(i + 1) * step_size], axis=1)

    return firing_rates, x_run_reshaped


# def get_firing_rates_old(pyramidal, event_count, x_run):
#     #TODO: Delete this once the one above is working for both 1D and 2D cases   
# 
#     firing_rates = np.zeros((event_count.shape[1], 1024))
#     x_run_reshaped = np.zeros(1024)
#     step_size = len(event_count)//firing_rates.shape[1]
#     
#     for i in range(firing_rates.shape[1]):
#         firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)
#         x_run_reshaped[i] = np.mean(x_run[i * step_size:(i + 1) * step_size])
# 
#     return firing_rates, x_run_reshaped


def get_activation_map_2d(firing_rates, len_edge, x_run_reshaped, n_bins = 225):
    """Spatially bin 2D firing rates into an activation map over a square grid.

    Assigns each time bin to the nearest position bin center and averages FRs.

    Args:
        firing_rates: (n_cells, T_bins) firing rates.
        len_edge: Environment size (cm).
        x_run_reshaped: (2, T_bins) positions (cm).
        n_bins: Total spatial bins; must be a perfect square.

    Returns:
        out: (n_cells, n_bins) activation map (Hz).
        position_bins: (n_bins, 2) bin centers in cm.
    """

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

    """Spatially bin 1D firing rates and optionally sort cells.

    If m_EC is provided, cells are sorted by EC index; otherwise cells are
    sorted by COM inferred from the map.

    Args:
        firing_rates: (n_cells, T_bins) firing rates (Hz).
        m_EC: 1D array of EC indices or None.
        x_run_reshaped: (1, T_bins) positions (cm).
        n_bins: Number of spatial bins.

    Returns:
        If m_EC is None:
            out: (n_cells, n_bins) sorted activation map (Hz).
            m_EC: Inferred position-of-maximum index per cell.
        Else:
            out: (n_cells, n_bins) activation map (Hz) sorted by m_EC.
    """

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
    # TODO: I don't think i use this function; double check
    """Compute smoothed burst proportion time series per condition.

    Proportion = burst_rate / (firing_rate + 1e-10), then boxcar-smoothed.

    Args:
        brs: Dict {'exp'|'control' -> {'F1'...'N2' -> (T, n_cells) burst rates}}.
        frs: Same structure as brs but for firing rates (Hz).
        window: Smoothing window length (samples).

    Returns:
        Dict with keys 'exp' and 'control', each a 1D array concatenating trials.
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
    """Compute spatial-correlation time series vs. F1/N1 anchors across trials.

    Args:
        all_act_maps: dict[condition][anchor] -> (n_cells, n_bins) anchor maps.
        all_act_maps_split: dict[condition][trial] -> list of 8 sub-maps
                           each shaped (n_cells, n_bins) with possible NaNs.

    Returns:
        all_cors: dict[anchor][condition] -> list of 1D arrays (per time chunk).
    """

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

    """Average 2D cross-correlogram between maps at out1 and out2 over cells.

    Args:
        act_maps: dict[out_label] -> (n_cells, n_bins_sq) flattened square maps.
        out1: First output label.
        out2: Second output label.

    Returns:
        np.ndarray: 2D correlogram averaged across cells, shape (2n-1, 2n-1).
    """

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

    """Gaussian-smooth each cell's 2D activation map.

    Args:
        act_maps_exp: dict[out] -> (n_cells, n_bins_sq).
        act_maps_cont: dict[out] -> (n_cells, n_bins_sq).
        sigma: Gaussian std (pixels) for smoothing.

    Returns:
        Tuple of (act_maps_exp, act_maps_cont) with smoothed, flattened maps.
    """
    
    for act_maps in [act_maps_exp, act_maps_cont]:
        for key in act_maps.keys():
            # print(act_maps[key].shape)
            reshape_val = int(np.sqrt(act_maps[key].shape[1]))
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], reshape_val, reshape_val)
            act_maps[key] = gaussian_filter(act_maps[key], sigma=sigma, mode='wrap') ## wrap because of toroidal B.C.
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], -1)
    
    return act_maps_exp, act_maps_cont


def format_val(val, digits=3):
    """Format a number in fixed precision or scientific notation for LaTeX.

    Args:
        val: Numeric value.
        digits: Decimal places for non-scientific formatting.

    Returns:
        str: Formatted value; scientific form uses '\\times 10^{...}'.
    """
    if abs(val) < 10**(-digits):
        return f"{val:.{1}e}".replace('e', r'\times 10^{') + '}'
    else:
        return f"{val:.{digits}f}" 


####################################################################
####################################################################

######################## PLOTTING FUNCTIONS ########################

####################################################################
####################################################################


COLOR_SETTINGS = {
    'exp_normal': "#1dde50",  # Vivid green, not too bright
    'exp_no_inh': "#1887d7",         # Original blue
    'exp_no_heb': "#7b1ed2",         # Original purple

    'control_normal': "#448A52",  # Greenish grey
    'control_no_inh': "#486b8d",    # Greyed-down blue
    'control_no_heb': "#5F4585",    # Greyed-down purple
}


def plot_firing_rates(ax, mean_firing_rates, out, vmin=None, vmax=None, fontsize=12, con=''):

    """Heatmap helper for firing rates with fixed extent 0–100 on x.

    Args:
        ax: Matplotlib Axes.
        mean_firing_rates: (n_cells, n_bins) array (Hz).
        out: Title suffix (e.g., 'F2').
        vmin, vmax: Color scaling.
        fontsize: Title/label font size.
        con: Y-label text (e.g., condition).

    Returns:
        The AxesImage from imshow (for colorbar sharing).
    """

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower', vmin=vmin, vmax=vmax, cmap = 'jet')
    ax.set_title(f"{out}", fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(con, fontsize=fontsize)
    return im


def plot_run(t, x, ax=None, color='black', novel=False, trial=''):    
    """Plot a 1D trajectory over time with environment-specific styling.

    Args:
        t: 1D array of time points (s).
        x: 1D array of positions along the track (cm).
        ax: Matplotlib Axes to draw on.
        color: Line color for the trajectory.
        novel: If True, highlight as a "novel" environment by thickening left/bottom spines.
               If False, highlight as a "familiar" environment by thickening top/right spines.
        trial: String label placed on the y-axis (e.g., trial name).

    Returns:
        None. Modifies the given Axes.
    """

    ax.plot(t, x, color=color, linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(trial, fontsize=18)

    ### Visualizing different environments:
    if novel:
        # Thicker left and bottom spines
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(0)
        ax.spines['right'].set_linewidth(0)

    else:
        # Thinner top and right spines
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(0)
        ax.spines['bottom'].set_linewidth(0)


def plot_condition(activation_maps, condition):

    """Plot a (1,4) panel of activation maps for a condition across trials.

    Args:
        activation_maps: dict[out] -> (n_cells, n_bins) arrays.
        condition: String label for y-axis (e.g., 'exp' or 'control').
    """

    fs = 20

    fig, axs = plt.subplots(1, 4, figsize=(10, 5), sharey=True, dpi=400)
    axs = axs.flatten()
   
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

    # Add a single colorbar for all subplots
    cbar = fig.colorbar(ims[0], ax=axs, orientation='vertical', fraction=0.025, pad=0.04)
    cbar.set_label('Firing rate (Hz)', fontsize=fs)
    cbar.ax.tick_params(labelsize=int(fs/1.5))

    plt.show()
    plt.close()



def create_plot_one_neuron(ca1, t_values):

    """Plot two-compartment voltages, thresholds, and spike/burst markers.

    Args:
        ca1: PyramidalCells instance containing recorded values.
        t_values: 1D time vector (s).

    Notes:
        Uses ca1.all_values['v_a'], ['v_b'], and spike/burst indicators.
    """

    v_b, v_a = np.array(ca1.all_values['v_b']), np.array(ca1.all_values['v_a'])
    bursts, spikes = ca1.burst_count, ca1.spike_count

    fig, ax = plt.subplots(2, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={'hspace': 0.1, 'height_ratios': [2, 2]}, dpi=600)

    I_b_scaled = ca1.I_b.T * 1 + ca1.pb['E_L']   
    I_a_scaled = ca1.I_a.T * 1 + ca1.pa['E_L']

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.tick_params(axis='both', which='both', length=3, width=0.5)
        a.set_facecolor('none')
        a.set_ylabel('')  
    
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
    ax[0].set_yticks([])

    # Add spikes and bursts as markers above V_b
    spike_y = ca1.pb['v_th'] + 2 
    burst_y = ca1.pb['v_th'] + 2 
    
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

    # Only show x-axis on bottom plot
    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].spines['bottom'].set_linewidth(0.5)
    ax[-1].set_xticks([])
    
    plt.tight_layout()
    plt.show()
    

def plot_pv_corr_distributions(groups, out1, out2, ax=None, p=(1,), colors=('green', '#808080')):

    """Plot KDEs for population-vector correlations with optional significance marks.

    Args:
        groups: Iterable of 1D arrays (e.g., [exp, control] correlations).
        out1, out2: Labels for the comparison title.
        ax: Matplotlib Axes (created externally).
        p: Tuple of p-values for significance stars; None to suppress.
        colors: Sequence of colors for groups.

    Returns:
        None. Draws into ax.
    """
    if ax is None:
        fig, ax = plt.subplots()

    fs = 20 # set font size


    # Plot KDEs (smoothed histograms)
    for i, group in enumerate(groups):
        kde = sns.kdeplot(group, fill=True, color=colors[i], alpha=0.6, linewidth=1.5, ax=ax)

        if p is not None:
            ax.axvline(np.median(group), color=colors[i], linestyle='--', linewidth=1.5)

    ax.set_xlabel('PV corr. coeff.', fontsize=fs)
    ax.set_ylabel('Frequency', fontsize=fs)
    ax.set_title(f'{out1} vs {out2}', fontsize =fs)
    ax.tick_params(labelsize=fs/1.5)

    sns.despine()

    if p is None:
        return 
    
    ### add significances to plot:

    elif len(p) == 1:

        y = ax.get_ylim()[1] * 0.9  
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

    
def create_raincloud_plot(groups, out1, out2, ax, p=(1,), colors=('green', '#808080')):

    """Raincloud-style summary (violin + box + strip) with significance bars.

    Args:
        groups: Iterable of 1D arrays to compare (2 or 3 groups).
        out1, out2: Labels for the panel title.
        ax: Matplotlib Axes to draw on.
        p: Tuple of p-values (length 1 or 2) for significance brackets.
        colors: Colors per group.

    Returns:
        None.
    """

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

def plot_burst_props(burst_props, tn, ax):
    """Plot burst proportion over time for exp/control on the provided axes.

    Args:
        burst_props: Dict with 'exp' and 'control' arrays of equal length.
        tn: Total duration corresponding to the series (s).
        ax: Matplotlib Axes.
    """

    for condition in ['exp', 'control']:
        burst_props[condition] = np.array(burst_props[condition])
        t = np.linspace(0, tn, burst_props[condition].shape[0])
        ax.plot(t, burst_props[condition], color=COLOR_SETTINGS[f'{condition}_normal']) 

    ax.set_ylabel('Burst Proportion', fontsize = 20 ) 
    ax.set_yticks([0.1, 0.14, 0.18, 0.22])
    ax.set_yticklabels(['.1', '.14', '.18', '.22'], fontsize=15)


def plot_cor_time_series(all_cors, tn, axs):
    """Plot mean ± std spatial correlations over time for F1/N1 anchors.

    Args:
        all_cors: dict[anchor]['exp'|'control'] -> list of 1D arrays over time.
        tn: Total duration (s) for x-axis scaling.
        axs: Sequence of two Matplotlib Axes for anchors ['F1','N1'].
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


def plot_one_2d_map(ax, act_map, out=None):

    """Plot one smoothed 2D activation map on the given Axes.

    Args:
        ax: Matplotlib Axes.
        act_map: 1D flattened square map (length n^2).
        out: Optional title (e.g., 'F2').

    Returns:
        The AxesImage handle from imshow.
    """

    act_map = act_map.reshape(int(np.sqrt(act_map.shape[0])), 
                                int(np.sqrt(act_map.shape[0])))

    im = ax.imshow(
        gaussian_filter(act_map, sigma=1.5),  # Smooth with Gaussian filter for illustration
        cmap='jet',
        origin='lower',
        extent=(0, LEN_EDGE_2D, 0, LEN_EDGE_2D),
        interpolation='bilinear'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if out is not None:
        ax.set_title(out, fontsize=20)

    return im 


def plot_cross_correlogram(act_maps, out1, out2, ax, title=None, cmap='jet', sigma=1.5):

    """Plot a smoothed cross-correlogram with center crosshairs.

    Args:
        act_maps: dict[out] -> (n_cells, n_bins_sq) square maps (flattened).
        out1, out2: Keys to compare.
        ax: Matplotlib Axes for drawing.
        title: Optional panel title.
        cmap: Matplotlib colormap name.
        sigma: Gaussian sigma for smoothing.

    Returns:
        None. Draws into ax.
    """

    correlogram = compute_cross_correlogram(act_maps, out1, out2)

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(correlogram, sigma=sigma)

    im = ax.imshow(smoothed, cmap=cmap, origin='lower', interpolation='bilinear')

    # Crosshairs at center
    center_y, center_x = np.array(smoothed.shape) // 2
    ax.axhline(center_y, color='black', linewidth=2)
    ax.axvline(center_x, color='black', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=20)
