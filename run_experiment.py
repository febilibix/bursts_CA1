import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from neuron import PyramidalCells
from itertools import product


ENVIRONMENTS_RUNS = {
    "F1": {'new_env': False, 'top_down': True},
    "F2": {'new_env': False, 'top_down': False},
    "N1": {'new_env': True,  'top_down': False},
    "F3": {'new_env': False, 'top_down': True}, 
    "N2": {'new_env': True,  'top_down': True},
    }


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000):
    bins = np.arange(0., len_track)
    fps = 1/dt

    x = np.array([])
    i = 0
    while True:
        stopping_time = np.random.uniform(0, 1, 2)
        stop1 = np.ones((int(stopping_time[0]*fps),)) * 0.
        speed = av_running_speed + np.random.randn() * 5
        run_length = len(bins) * fps / speed
        run1 = np.linspace(0., float(len(bins)-1), int(run_length))
        stop2 = np.ones((int(stopping_time[1]*fps),)) * (len(bins)-1.)
        speed = av_running_speed + np.random.randn() * 5
        run_length = len(bins) * fps / speed
        run2 = np.linspace(len(bins)-1., 0., int(run_length))
        x = np.concatenate((x, stop1, run1, stop2, run2))
        if len(x) >= tn*fps:
            break
        i += 1

    x = x[:int(tn*fps)]
    t = np.arange(len(x))/fps

    return t, x


def get_firing_rates(pyramidal, event_count, x_run):

    firing_rates = np.zeros((event_count.shape[1], 1024))
    x_run_reshaped = np.zeros(1024)
    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)
        x_run_reshaped[i] = np.mean(x_run[i * step_size:(i + 1) * step_size])

    return firing_rates, x_run_reshaped


def get_activation_map(firing_rates, m_EC, x_run_reshaped, n_bins = 64):
    sort_TD = np.argsort(m_EC)
    sorted_fr = firing_rates[np.ix_(sort_TD, np.arange(firing_rates.shape[1]))]

    bins = np.arange(n_bins)
    n_cell = np.arange(sorted_fr.shape[0])
    out_collector = {k : [] for k in product(n_cell, bins)}
    out = np.zeros((sorted_fr.shape[0], n_bins))
    position_bins = np.linspace(0, x_run_reshaped.max(), n_bins)

    for idx, pos in enumerate(x_run_reshaped):
        bin_idx = np.argmin(np.abs(position_bins - pos))

        for i in range(sorted_fr.shape[0]):
            out_collector[(i, bin_idx)].append(sorted_fr[i, idx])

    for k, v in out_collector.items():
        out[k] = np.mean(v)

    return out


def plot_firing_rates(fig, ax, mean_firing_rates, out):

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{out}")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")

    return fig, ax


def cor_act_maps(act_map1, act_map2):
    cor = np.zeros(act_map1.shape[0])
    for i in range(act_map1.shape[0]):
        cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]

    return cor.mean()


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
    print(burst_rates[key].shape[1])
    t = np.arange(burst_rates[key].shape[1])*tn / (burst_rates[key].shape[1])
    plt.figure()
    for label, br in burst_rates.items():
        plt.plot(t, br.mean(axis=0), label=label)
        
        lower_bound = br.mean(axis=0) - br.std(axis=0)
        upper_bound = br.mean(axis=0) + br.std(axis=0)
        if label == 'N2':
            plt.fill_between(t, lower_bound, upper_bound, alpha=0.2)
    plt.legend(loc = 'upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Burst rate')
    out = 'burst_rates_ctrl' if condition == 'cont' else 'burst_rates'
    plt.savefig(f"plots/full_experiment/{out}.png")
    plt.close()


def run_simulation(alpha = 0.05, plot_burst = False):
    lr = 10
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.3 # similarity between environments
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}


    for condition in ['exp', 'cont']:

        pyramidal = PyramidalCells(n_cells, len_track = len_track, learning_rate = lr, dt = dt)
        m_EC_orig, m_CA3_orig = pyramidal.m_EC, pyramidal.m_CA3
        pyramidal.alpha = alpha
        t_run, x_run = simulate_run(len_track, speed, dt, tn)

        activation_maps = {}
        burst_rates = {}
        EC_act_maps = {}
        ca3_act_maps = {}

        for out, params in ENVIRONMENTS_RUNS.items():
            print(f"Running {out} {condition}...")
            if condition == 'cont':
                params['top_down'] = True
            
            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = a, t_per_epoch=t_epoch)

            fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
            br, _ = get_firing_rates(pyramidal, burst_count, x_run)

            ca3_act_maps[out] = get_activation_map(pyramidal.full_CA3_activities.T, m_CA3_orig, x_run)
            EC_act_maps[out] = get_activation_map(pyramidal.full_EC_activities.T, m_EC_orig, x_run)
            activation_maps[out] = get_activation_map(fr, m_EC_orig, x_run_reshaped)
            burst_rates[out] = br

        mean_cor_F = cor_act_maps(activation_maps['F2'], activation_maps['F3'])
        mean_cor_N = cor_act_maps(activation_maps['N1'], activation_maps['N2'])
        mean_cor_T = cor_act_maps(activation_maps['F2'], activation_maps['N1'])
        mean_cor_CA3 = cor_act_maps(ca3_act_maps['F2'], ca3_act_maps['N1'])

        print(f"Mean correlation between F1 and F2: {mean_cor_F}")
        print(f"Mean correlation between N1 and N2: {mean_cor_N}")
        print(f"Mean correlation between F2 and N1: {mean_cor_T}")
        print('Baseline Correlation CA3: ', mean_cor_CA3)

        plot_act_maps(activation_maps, 'CA1', condition)
        plot_act_maps(EC_act_maps, 'EC', condition)
        plot_act_maps(ca3_act_maps, 'CA3', condition)

        if plot_burst:
            plot_burst_rates(burst_rates, tn, condition)


def main():
    np.random.seed(1903)
    alpha = 0.0125
    run_simulation(alpha, plot_burst = True)


if __name__ == '__main__':
    main()