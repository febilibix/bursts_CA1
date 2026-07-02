import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import sys 
sys.path.append('../')
from itertools import product
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
    '2D': {'pyramidal' : 289 , 'inter_a' : 30 , 'inter_b' : 30 , 'CA3' : 169 },
}

T_EPOCH = 1
SPEED = 20
DT = 0.001
A = 0.3

LEN_EDGE_2D = 50
TN_2D = 250  



def cor_act_maps_2d(act_map1, act_map2, which='pv'):
    # act_map1, act_map2 = act_map[out1], act_map[out2]

    if which == 'pv':
        act_map1, act_map2 = act_map1.T, act_map2.T

    cor = np.zeros(act_map1.shape[0])

    for i in range(act_map1.shape[0]):
        valid_idx = ~np.isnan(act_map1[i, :]) & ~np.isnan(act_map2[i, :])
        if np.any(valid_idx):
            cor[i] = pearsonr(act_map1[i, valid_idx], act_map2[i, valid_idx])[0]
        else:
            cor[i] = np.nan  # Handle case where all values are NaN
       
    return cor


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


def get_firing_rates(dt, event_count, x_run, n_bins=1024, n_dim=1):

    firing_rates = np.zeros((event_count.shape[1], n_bins))
    x_run_reshaped = np.zeros((n_dim, n_bins))
    step_size = len(event_count)//n_bins
    x_run = x_run.reshape((n_dim, -1)) 
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*dt)
        x_run_reshaped[:, i] = np.mean(x_run[:, i * step_size:(i + 1) * step_size], axis=1)

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


def smooth_act_maps(all_act_maps, sigma):
    
    for act_maps in all_act_maps:
        for key in act_maps.keys():
            # print(act_maps[key].shape)
            reshape_val = int(np.sqrt(act_maps[key].shape[1]))
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], reshape_val, reshape_val)
            act_maps[key] = gaussian_filter(act_maps[key], sigma=sigma, mode='wrap')
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], -1)
    
    return all_act_maps


######################## PLOTTING FUNCTIONS ########################


COLOR_SETTINGS = {
    'exp_normal': "#1dde50",  # Vivid green, not too bright
    'exp_no_inh': "#1887d7",         # Original blue
    'exp_no_heb': "#7b1ed2",         # Original purple

    'control_normal': "#898989",  #  grey
    'control_no_inh': "#486b8d",    # Greyed-down blue
    'control_no_heb': "#5F4585",    # Greyed-down purple
}

 
def plot_pv_corr_distributions(groups, out1, out2, ax=None, p=(1,), colors=('green', '#808080')):
    fs = 20

    # Plot KDEs (smoothed histograms)
    for i, group in enumerate(groups):
        kde = sns.kdeplot(group, fill=True, color=colors[i], alpha=0.6, linewidth=1.5, ax=ax)
        ax.axvline(np.median(group), color=colors[i], linestyle='--', linewidth=1.5)

    # Labeling
    ax.set_xlabel('PV corr. coeff.', fontsize=fs)
    ax.set_ylabel('Frequency', fontsize=fs)
    ax.set_title(f'{out1} vs {out2}', fontsize =fs)
    ax.tick_params(labelsize=fs/1.5)

    sns.despine()

    if p is None:
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

    ax.hlines(0, color='grey', lw=0.5, xmin=-1, xmax=1.5)

    ax.set_title(f'{out1} vs {out2}', fontsize=fs)
    ax.set_ylabel("Spatial correlation", fontsize=fs)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.tick_params(labelsize=fs/1.5)
    if len(groups) == 2 or p is None:
        ax.set_ylim(-1.1, 1.3)
    else:
        ax.set_ylim(-1.1, 1.5)
    ax.set_xlim(-0.5, 1.5)
    sns.despine(ax=ax)
    ax.legend_.remove()

    if p is None:
        return

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

    return im 


def plot_cross_correlogram(smoothed, ax, title=None, cmap='jet', ylabel=None, vmin=None, vmax=None):

    im = ax.imshow(smoothed, cmap=cmap, origin='lower', interpolation='bilinear', vmin=vmin, vmax=vmax)

    # Crosshairs at center
    center_y, center_x = np.array(smoothed.shape) // 2
    ax.axhline(center_y, color='black', linewidth=2)
    ax.axvline(center_x, color='black', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=20)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)
