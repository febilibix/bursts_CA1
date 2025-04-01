import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from neuron import PyramidalCells
from itertools import product
from scipy.ndimage import gaussian_filter
import csv
import warnings
import pickle


ENVIRONMENTS_RUNS = {
    "F1": {'new_env': False, 'top_down': True},
    "F2": {'new_env': False, 'top_down': False},
    "N1": {'new_env': True,  'top_down': False},
    "F3": {'new_env': False, 'top_down': True}, 
    "N2": {'new_env': True,  'top_down': True},
    }


def simulate_2d_run_old(len_edge=20, av_running_speed=20, dt=0.01, tn=1000):
    """
    Simulate an animal running in a square box row by row.
    The mouse moves left to right, then right to left, shifting down 1 cm at each turn with a smooth transition.
    """
    fps = 1 / dt  # Frames per second
    total_time_steps = int(tn * fps)
    x_positions = []
    y_positions = []
    
    x, y = 0, 0
    direction, y_direction = 1, 1  # 1 for right, -1 for left
    
    while len(x_positions) < total_time_steps:
        stopping_time = np.random.uniform(0, 1, 2)
        stop1 = int(stopping_time[0] * fps)
        stop2 = int(stopping_time[1] * fps)
        
        # Stopping phase
        for _ in range(stop1):
            if len(x_positions) >= total_time_steps:
                break
            x_positions.append(x)
            y_positions.append(y)
        
        # Running phase (left-right or right-left)
        speed = av_running_speed + np.random.randn() * 5
        run_length = len_edge * fps / speed
        x_values = np.linspace(0 if direction == 1 else len_edge-1, 
                               len_edge-1 if direction == 1 else 0, 
                               int(run_length))
        
        for x in x_values:
            if len(x_positions) >= total_time_steps:
                break
            x_positions.append(x)
            y_positions.append(y)
        
        # Second stopping phase
        for _ in range(stop2):
            if len(x_positions) >= total_time_steps:
                break
            x_positions.append(x)
            y_positions.append(y)
        
        # Move smoothly to the next row (down 1 cm)
        speed = av_running_speed + np.random.randn() * 2  # Slightly slower for downward movement
        step_length = fps / speed  # Number of frames needed for 1 cm movement
        # y_target = y + y_direction
        y_values = np.linspace(y, y + y_direction, int(step_length))  # Smooth y-transition
        
        for y in y_values:
            if len(x_positions) >= total_time_steps or y > len_edge:
                break
            x_positions.append(x)
            y_positions.append(y)

        # Switch direction for next row
        direction *= -1
        if y >= len_edge:
            y_direction = -1 + np.random.randn() * 0.1
        if y <= 0:
            y_direction = 1 + np.random.randn() * 0.1
            
        if y + y_direction > len_edge:
            y_direction *= -1
        if y + y_direction < 0:
            y_direction *= -1
    
    t = np.arange(len(x_positions)) / fps
    return t, np.vstack((x_positions, y_positions))



def simulate_2d_run(len_edge=20, av_running_speed=20, dt=0.01, tn=1000, a=np.pi/20):
    """
    Simulates a mouse moving in a 2D square environment with smooth, random changes in direction.
    The mouse updates its movement direction gradually and bounces off walls naturally.
    """
    fps = 1 / dt  # Frames per second
    total_time_steps = int(tn * fps)
    x_positions, y_positions = [], []
    
    # Initial position and direction
    x, y = len_edge/2, len_edge/2  # Start in the middle
    phi = np.random.uniform(0, 2 * np.pi)  # Initial movement direction
    
    for _ in range(total_time_steps):
        # Update direction with small random change
        phi += np.random.uniform(-a, a)
        
        # Compute new position based on speed and direction
        dx = np.cos(phi) * (av_running_speed + np.random.randn() * 3) * dt 
        dy = np.sin(phi) * (av_running_speed + np.random.randn() * 3) * dt 
        
        # Check for boundary conditions
        if x + dx >= len_edge or x + dx <= 0:
            phi = np.pi - phi  # Reflect angle horizontally
        if y + dy >= len_edge or y + dy <= 0:
            phi = -phi  # Reflect angle vertically
        
        # Update position
        x = np.clip(x + dx, 0, len_edge)
        y = np.clip(y + dy, 0, len_edge)
        
        x_positions.append(x)
        y_positions.append(y)
    
    t = np.arange(len(x_positions)) / fps
    return t, np.vstack((x_positions, y_positions))



def get_firing_rates(pyramidal, event_count, x_run, n_bins = 2**13):

    firing_rates = np.zeros((event_count.shape[1], n_bins))
    x_run_reshaped = np.zeros((2, firing_rates.shape[1]))
    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)
        x_run_reshaped[:, i] = np.mean(x_run[:, i * step_size:(i + 1) * step_size], axis=1)

    return firing_rates, x_run_reshaped


def get_activation_map(firing_rates, len_edge, x_run_reshaped, n_bins = 225):

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


def plot_firing_rates(fig, ax, mean_firing_rates, out):

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{out}")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")

    return fig, ax


def cor_act_maps(act_map, out1, out2):
    act_map1, act_map2 = act_map[out1], act_map[out2]
    cor = np.zeros(act_map1.shape[0])
    condition = None
    for i in range(act_map1.shape[0]):
        if act_map1[i, :].sum() == 0 and act_map2[i, :].sum() == 0:
            cor[i] = np.nan
            condition = f"{out1} and {out2}"
        elif act_map1[i, :].sum() == 0:
            cor[i] = np.nan
            condition = out1
        elif act_map2[i, :].sum() == 0:
            cor[i] = np.nan
            condition = out2
        else:
            condition = None
            cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]
        if condition is not None:
            warnings.warn(f"Cell {i} has no activity in {condition}")
            
    return np.nanmean(cor)


def plot_act_maps(activation_maps, area, condition):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    for i, (out, act_map) in enumerate(activation_maps.items()):
        if out == 'F1': continue
        fig, axs[i-1] = plot_firing_rates(fig, axs[i-1], act_map, out)

    plt.tight_layout()
    fig.savefig(f"plots/full_experiment/firing_rates_{area}_{condition}.png")
    plt.close(fig)


def plot_burst_rates(burst_rates, tn, condition):
    key = list(burst_rates.keys())[0]
    t = np.arange(burst_rates[key].shape[1])*tn / (burst_rates[key].shape[1])
    
    plt.figure()
    for label, br in burst_rates.items():
        print(t[np.argmax(br.mean(axis=0))])
        plt.plot(t, br.mean(axis=0), label=label)
        
        # lower_bound = br.mean(axis=0) - br.std(axis=0)
        # upper_bound = br.mean(axis=0) + br.std(axis=0)
        # if label == 'N2':
        #     plt.fill_between(t, lower_bound, upper_bound, alpha=0.2)
    plt.legend(loc = 'upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Burst rate')
    out = 'burst_rates_ctrl' if condition == 'cont' else 'burst_rates'
    plt.savefig(f"plots/full_experiment/2d_case/{out}.png")
    plt.close()
    

def plot_2d_run(t_run, x_run):
    plt.figure()
    sc = plt.scatter(x_run[0,:], x_run[1,:], c=t_run, cmap='viridis', s=5, alpha=0.05)  # Use 'viridis' or another colormap
    plt.colorbar(sc, label='Time (s)') 
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('2D Run')
    plt.savefig('plots/full_experiment/2d_case/2d_run.png')


def plot_example(cell_idx, fr, br, tn, act_map, pos_bins, x_run_reshaped):

    time = np.arange(fr.shape[1]) * tn / fr.shape[1]
    plt.figure()
    plt.plot(time, fr[cell_idx,:], label='fr')
    # plt.plot(time, br[cell_idx,:], label='br')
    plt.plot(time, x_run_reshaped[0,:], label='x_pos')
    plt.plot(time, x_run_reshaped[1,:], label='y_pos')
    plt.legend()
    plt.title(f"Activity of cell {cell_idx}")
    plt.xlabel('Time (s)')
    plt.ylabel('Rate')
    plt.savefig(f"plots/full_experiment/2d_case/example_fr_time.png")
    plt.close()

    x, y = pos_bins[:, 0], pos_bins[:, 1]
    activity = act_map[cell_idx, :]

    plt.figure(figsize=(6,6))
    sc = plt.scatter(x,y, c=activity, cmap='viridis', s=500, marker='s')
    plt.colorbar(sc, label='Firing Rate (Hz)') 
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Activity Map for Cell {cell_idx}")
    plt.tight_layout()
    plt.savefig(f"plots/full_experiment/2d_case/example_fr.png")
    plt.close()  

    # Create a 2D grid from 1D arrays
    x_bins = np.linspace(x.min(), x.max(), 20)  # Adjust resolution
    y_bins = np.linspace(y.min(), y.max(), 20)
    X, Y = np.meshgrid(x_bins, y_bins)

    # Interpolate the activity onto the grid
    activity_grid = np.zeros((len(y_bins), len(x_bins)))
    for i in range(len(x)):
        xi = np.digitize(x[i], x_bins) - 1
        yi = np.digitize(y[i], y_bins) - 1
        activity_grid[yi, xi] = activity[i]

    # Apply Gaussian smoothing
    activity_smooth = gaussian_filter(activity_grid, sigma=1.5)  # Adjust sigma as needed

    # Plot using imshow for smooth visualization
    plt.figure(figsize=(6,6))
    plt.imshow(activity_smooth, cmap='jet', origin='lower', 
            extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()],
            interpolation='bilinear')  # Try 'bicubic' for more smoothing
    plt.colorbar(label='Firing Rate (Hz)')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Activity Map for Cell {cell_idx}")
    plt.tight_layout()
    plt.savefig(f"plots/full_experiment/2d_case/example_fr_smooth.png")
    plt.close()


def plot_smooth_activity_map(cell_idx, act_maps, all_pos_bins, condition, m_EC_orig, m_EC_new):

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    fig.suptitle(f"Activity Map for Cell {cell_idx}")

    for i, (out, act_map) in enumerate(act_maps.items()):
        if out == 'F1': continue
            # i = i + 1

        pos_bins = all_pos_bins[out]

        x, y = pos_bins[:, 0], pos_bins[:, 1]
        activity = act_map[cell_idx, :]
        
        x_bins = np.linspace(x.min(), x.max(), 20)  # Adjust resolution
        y_bins = np.linspace(y.min(), y.max(), 20)
        X, Y = np.meshgrid(x_bins, y_bins)

        # Interpolate the activity onto the grid
        activity_grid = np.zeros((len(y_bins), len(x_bins)))
        for j in range(len(x)):
            xi = np.digitize(x[j], x_bins) - 1
            yi = np.digitize(y[j], y_bins) - 1
            activity_grid[yi, xi] = activity[j]

        # Apply Gaussian smoothing
        activity_smooth = gaussian_filter(activity_grid, sigma=1.5)  # Adjust sigma as needed

        # Plot using imshow for smooth visualization
        im = axs[i-1].imshow(activity_smooth, cmap='jet', origin='lower', 
                extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()],
                interpolation='bilinear')  # Try 'bicubic' for more smoothing
        
        cbar_ax = axs[i-1].inset_axes([1.05, 0, 0.05, 1])  # [x-offset, y-offset, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        axs[i-1].set_xlabel("X Position")
        axs[i-1].set_ylabel("Y Position")

        pos = m_EC_orig[:, cell_idx] if out.startswith('F') else m_EC_new[:, cell_idx]
        x_pos, y_pos = np.round(pos, 2)
        axs[i-1].set_title(f"{out}, x = {x_pos}, y = {y_pos}")

    plt.tight_layout()
    plt.savefig(f"plots/full_experiment/2d_case/example_{cell_idx}_fr_smooth_{condition}.png")
    plt.close()



def run_simulation(alpha, a, lr, plot_burst = False):
    lr = 20 # 10
    t_epoch = 1
    speed = 30
    len_edge = 20
    dt = 0.001
    tn = 200 # 350
    # a = 0.3 # similarity between environments
    n_cells = {'pyramidal' : 400, 'inter_a' : 40, 'inter_b' : 40, 'CA3' : 225}
    seed = 98

    all_runs = {}

    for k in ENVIRONMENTS_RUNS.keys():
        t_run, x_run = simulate_2d_run(len_edge, speed, dt, tn)
        all_runs[k] = (t_run, x_run)

    for condition in ['exp', 
                      'cont'
                      ]:
        np.random.seed(seed)

        pyramidal = PyramidalCells(n_cells, len_edge = len_edge, learning_rate = lr, dt = dt, seed = seed)
        # m_EC_orig, m_CA3_orig = pyramidal.m_EC, pyramidal.m_CA3
        # m_EC_new, m_CA3_new = pyramidal.m_EC_new, pyramidal.m_CA3_new

        pyramidal.alpha = alpha

        act_maps = {}
        ca3_act_maps = {}
        burst_rates = {}
        firing_rates = {}
        all_pos_bins = {}
        burst_counts, event_counts = {}, {}

        for out, params in ENVIRONMENTS_RUNS.items():
            print(f"Running {out} {condition}...")
            
            if condition == 'cont':
                params['top_down'] = True            

            t_run, x_run = all_runs[out]
            
            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = a, t_per_epoch=t_epoch)

            fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
            br, _ = get_firing_rates(pyramidal, burst_count, x_run, n_bins = 128)
            fr_short, _ = get_firing_rates(pyramidal, event_count, x_run, n_bins = 128)
            
            act_map, pos_bins = get_activation_map(fr, len_edge, x_run_reshaped)
            ca3_act_maps[out], _ = get_activation_map(pyramidal.full_CA3_activities.T, len_edge, x_run)

            # if out == 'F1':
            #     plot_2d_run(t_run, x_run)
            #     plot_example(207, fr, br, tn, act_map, pos_bins, x_run_reshaped)
            
            # print(np.isnan(act_map), np.isnan(act_map).sum(), act_map.shape)
            act_maps[out] = act_map
            burst_rates[out] = br
            firing_rates[out] = fr_short
            all_pos_bins[out] = pos_bins

            burst_counts[out] = burst_count
            event_counts[out] = event_count


        # pos = np.random.randint(0, 400)
        # 
        # plot_smooth_activity_map(pos, act_maps, all_pos_bins, condition, m_EC_orig, m_EC_new)
        # plot_smooth_activity_map(296, act_maps, all_pos_bins, condition, m_EC_orig, m_EC_new)

        with open(f'plots/full_experiment/2d_case/act_maps_{condition}.pkl', 'wb') as f:
            pickle.dump((act_maps, all_pos_bins, burst_counts, event_counts), f)

        # print(np.isnan(act_maps['F2']).sum(), np.isnan(act_maps['F3']).sum(), np.isnan(act_maps['N1']).sum(), np.isnan(act_maps['N2']).sum())
        # mean_cor_F = cor_act_maps(act_maps, 'F2', 'F3')
        # mean_cor_N = cor_act_maps(act_maps, 'N1', 'N2')
        # mean_cor_T = cor_act_maps(act_maps, 'F2', 'N1')
        # mean_cor_T2 = cor_act_maps(act_maps, 'F2', 'N2')
        # mean_cor_CA3 = cor_act_maps(ca3_act_maps, 'F2', 'N2')
        # 
        # for k, v in act_maps.items():
        #     print(k, np.all(v == 0, axis = 1).sum())
        # 
        # print(condition)
        # print(f"Mean correlation between F2 and F3: {mean_cor_F}")
        # print(f"Mean correlation between N1 and N2: {mean_cor_N}")
        # print(f"Mean correlation between F2 and N1: {mean_cor_T}")
        # print(f"Mean correlation between F2 and N2: {mean_cor_T2}")
        # print('Baseline Correlation CA3:', mean_cor_CA3)
        # 
        # with open(f'plots/full_experiment/2d_case/correlations.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([condition, mean_cor_F, mean_cor_N, mean_cor_T, mean_cor_CA3])
        #     
        # if plot_burst:
        #     plot_burst_rates(burst_rates, tn, condition)


def main():

    alpha = 0.002 # 0.0025
    run_simulation(alpha, plot_burst = True)


if __name__ == '__main__':
    main()
    # Mock data generation
    np.random.seed(42)

    # Number of cells and bins
    num_cells = 10
    num_bins = 100

    # Simulated positions (random walk in a 2D space)
    pos_bins = np.cumsum(np.random.randn(num_bins, 2) * 2, axis=0)

    # Activity maps for each condition (random activity)
    act_maps = {
        "condition1": np.random.rand(num_cells, num_bins),
        "condition2": np.random.rand(num_cells, num_bins),
        "condition3": np.random.rand(num_cells, num_bins),
        "condition4": np.random.rand(num_cells, num_bins)
    }

    # Position bins dictionary
    all_pos_bins = {
        "condition1": pos_bins,
        "condition2": pos_bins + np.random.randn(*pos_bins.shape) * 0.5,
        "condition3": pos_bins + np.random.randn(*pos_bins.shape) * 0.5,
        "condition4": pos_bins + np.random.randn(*pos_bins.shape) * 0.5
    }

    # Pick a cell index to test
    cell_idx = 0

    m_EC_orig = np.random.rand(2, num_cells)
    m_EC_new = np.random.rand(2, num_cells)

    print(m_EC_orig.shape, m_EC_new.shape)

    # Test the function
    plot_smooth_activity_map(cell_idx, act_maps, all_pos_bins, "test", m_EC_orig, m_EC_new)