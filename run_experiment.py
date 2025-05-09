import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from neuron import PyramidalCells
import csv
import pandas as pd
from itertools import product


ENVIRONMENTS_RUNS = {
    # "F1": {'new_env': False, 'top_down': False, 'a': 0},
    "F2": {'new_env': False, 'top_down': False},
    "N1": {'new_env': True,  'top_down': False},
    "F3": {'new_env': False, 'top_down': True}, 
    "N2": {'new_env': True,  'top_down': True},
    }

ENVIRONMENTS_RUNS_CONTROL = {
   
    "F2": {'new_env': False, 'top_down': True},
    "N1": {'new_env': True,  'top_down': True},
    "F3": {'new_env': False, 'top_down': True}, 
    "N2": {'new_env': True,  'top_down': True},
    
    }


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000):
    ## TODO: Does it need to be this long?
    bins = np.arange(0., len_track)
    fps = 1/dt
    n_runs = int(2*tn/(len_track/av_running_speed))

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

    print('Event count shape:', event_count.shape)
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


def run_simulation(alpha = 0.05, plot_burst = False):
    lr = 15
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.3 # similarity between environments
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}
    cutoff = 0.004

    for condition in ['exp',
                      #  'control'
                      ]:

    #### TODO: Loop here over exp, control condition and for control just keep top down on but go through all 4 phases

        pyramidal = PyramidalCells(n_cells, len_track = len_track, learning_rate = lr, dt = dt)
        pyramidal.W_ip_a = pyramidal.W_ip_a
        pyramidal.alpha = alpha

        t_run, x_run = simulate_run(len_track, speed, dt, tn)


        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        figb, axsb = plt.subplots(2, 2, figsize=(12, 12))
        axsb = axsb.flatten()
        axs = axs.flatten()
        print('F1')
        event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, t_per_epoch=t_epoch, top_down=True)
        fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
        m_EC_orig, m_CA3_orig = pyramidal.m_EC, pyramidal.m_CA3
        # burst_collectors = {'F1': pyramidal.burst_collector}
        # pyramidal.burst_collector = []
        br, _ = get_firing_rates(pyramidal, burst_count, x_run)
        # pyramidal = plot_burst_collector(pyramidal, 'F1')
        activation_maps = {}
        burst_rates = {'F1': br}

        EC_act_maps = {}
        ca3_act_maps = {}

    

        for i, (out, params_orig) in enumerate(ENVIRONMENTS_RUNS.items()):
            print(out)
            params = params_orig.copy()
            if condition == 'control':
                params['top_down'] = True

            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = a, t_per_epoch=t_epoch)
            # burst_collectors[out] = pyramidal.burst_collector
            pyramidal.burst_collector = []
            fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
            br, _ = get_firing_rates(pyramidal, burst_count, x_run)
            mean_firing_rates = get_activation_map(fr, m_EC_orig, x_run_reshaped)
            mean_burst_rates = get_activation_map(br, m_EC_orig, x_run_reshaped)

            ca3_act_maps[out] = get_activation_map(pyramidal.full_CA3_activities.T, m_CA3_orig, x_run)
            EC_act_maps[out] = get_activation_map(pyramidal.full_EC_activities.T, m_EC_orig, x_run)
            activation_maps[out] = mean_firing_rates
            burst_rates[out] = br
            fig, axs[i] = plot_firing_rates(fig, axs[i], mean_firing_rates, out)
            figb, axsb[i] = plot_firing_rates(figb, axsb[i], mean_burst_rates, out)
            # pyramidal = plot_burst_collector(pyramidal, out)

        mean_cor_F = cor_act_maps(activation_maps['F2'], activation_maps['F3'])
        mean_cor_N = cor_act_maps(activation_maps['F2'], activation_maps['N2'])
        mean_cor_T = cor_act_maps(activation_maps['F2'], activation_maps['N1'])
        mean_cor_CA3 = cor_act_maps(ca3_act_maps['F2'], ca3_act_maps['N1'])

        print(f"Mean correlation between F2 and F3: {mean_cor_F}")
        print(f"Mean correlation between F2 and N2: {mean_cor_N}")
        print(f"Mean correlation between F2 and N1: {mean_cor_T}")
        
        print('Baseline Correlation CA3: ', mean_cor_CA3)

        with open(f'plots/full_experiment/cors_per_cutoff.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([condition, mean_cor_F, mean_cor_N, mean_cor_T])

        plt.tight_layout()
        fig.savefig(f"plots/full_experiment/firing_rates_CA1_{condition}.png")
        plt.close(fig)
        
        plt.tight_layout()
        figb.savefig(f"plots/full_experiment/burst_rates_CA1_{condition}.png")
        plt.close(figb)

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        ax = ax.flatten()
        for i, (out, act_map) in enumerate(ca3_act_maps.items()):
            im = ax[i].imshow(act_map, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[i])
            ax[i].set_title(f'CA3 {out}')
        plt.tight_layout()
        fig.savefig(f"plots/full_experiment/CA3_activities_{condition}.png")
        plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        ax = ax.flatten()
        for i, (out, act_map) in enumerate(EC_act_maps.items()):
            im = ax[i].imshow(act_map, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[i])
            ax[i].set_title(f'EC {out}')
        plt.tight_layout()
        fig.savefig(f"plots/full_experiment/EC_activities_{condition}.png")
        plt.close(fig)


        if plot_burst:
            plot_burst_rates(burst_rates, tn, condition)
            # plot_burst_collector(burst_collectors, sigmoid_params)

    return mean_cor_F, mean_cor_N, mean_cor_T




def run_simulation_control(alpha = 0.05, plot_burst = False):
    print('Running control condition')
    lr = 15
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.3 # similarity between environments
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}
    cutoff = 0.004


    pyramidal = PyramidalCells(n_cells, len_track=len_track, learning_rate = lr, dt = dt)
    pyramidal.alpha = alpha

    t_run, x_run = simulate_run(len_track, speed, dt, tn)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    figb, axsb = plt.subplots(2, 2, figsize=(12, 12))
    axsb = axsb.flatten()
    axs = axs.flatten()
    print('F1')
    event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, t_per_epoch=t_epoch, top_down=True)
    fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
    m_EC_orig, m_CA3_orig = pyramidal.m_EC, pyramidal.m_CA3
    br, _ = get_firing_rates(pyramidal, burst_count, x_run)

    activation_maps = {}
    burst_rates = {'F1': br}

    EC_act_maps = {}
    ca3_act_maps = {}

    for i, (out, params) in enumerate(ENVIRONMENTS_RUNS_CONTROL.items()):
        print(out)
        event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = a, t_per_epoch=t_epoch)
        pyramidal.burst_collector = []
        fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
        br, _ = get_firing_rates(pyramidal, burst_count, x_run)
        print(fr.shape, x_run_reshaped.shape)
        mean_firing_rates = get_activation_map(fr, m_EC_orig, x_run_reshaped)
        mean_burst_rates = get_activation_map(br, m_EC_orig, x_run_reshaped)

        ca3_act_maps[out] = get_activation_map(pyramidal.full_CA3_activities.T, m_CA3_orig, x_run)
        EC_act_maps[out] = get_activation_map(pyramidal.full_EC_activities.T, m_EC_orig, x_run)
        activation_maps[out] = mean_firing_rates
        burst_rates[out] = br
        fig, axs[i] = plot_firing_rates(fig, axs[i], mean_firing_rates, out)
        figb, axsb[i] = plot_firing_rates(figb, axsb[i], mean_burst_rates, out)
        # pyramidal = plot_burst_collector(pyramidal, out)

    
    mean_cor_F = cor_act_maps(activation_maps['F2'], activation_maps['F3'])
    mean_cor_N = cor_act_maps(activation_maps['F2'], activation_maps['N2'])
    mean_cor_T = cor_act_maps(activation_maps['F2'], activation_maps['N1'])
    mean_cor_CA3 = cor_act_maps(ca3_act_maps['F2'], ca3_act_maps['N1'])

    print(f"Mean correlation between F2 and F3: {mean_cor_F}")
    print(f"Mean correlation between F2 and N2: {mean_cor_N}")
    print(f"Mean correlation between F2 and N1: {mean_cor_T}")
    
    print('Baseline Correlation CA3: ', mean_cor_CA3)


    print(f"Mean correlation between F2 and N1: {mean_cor_T}")
    print('Baseline Correlation CA3: ', mean_cor_CA3)


    plt.tight_layout()
    fig.savefig(f"plots/full_experiment/firing_rates_CA1_ctrl.png")
    plt.close(fig)
    
    plt.tight_layout()
    figb.savefig(f"plots/full_experiment/burst_rates_CA1_ctrl.png")
    plt.close(figb)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.flatten()
    for i, (out, act_map) in enumerate(ca3_act_maps.items()):
        im = ax[i].imshow(act_map, aspect='auto', origin='lower')
        fig.colorbar(im, ax=ax[i])
        ax[i].set_title(f'CA3 {out}')
    plt.tight_layout()
    fig.savefig(f"plots/full_experiment/CA3_activities_ctrl.png")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.flatten()
    for i, (out, act_map) in enumerate(EC_act_maps.items()):
        im = ax[i].imshow(act_map, aspect='auto', origin='lower')
        fig.colorbar(im, ax=ax[i])
        ax[i].set_title(f'EC {out}')
    plt.tight_layout()
    fig.savefig(f"plots/full_experiment/EC_activities_ctrl.png")
    plt.close(fig)


    if plot_burst:
        plot_burst_rates(burst_rates, tn, 'ctrl')
        # plot_burst_collector(burst_collectors, sigmoid_params)

    return mean_cor_T


def plot_burst_rates(burst_rates, tn, condition):
    key = list(burst_rates.keys())[0]
    print(burst_rates[key].shape[1])
    t = np.arange(burst_rates[key].shape[1])*tn / (burst_rates[key].shape[1])
    plt.figure()
    for label, br in burst_rates.items():
        plt.plot(t, br.mean(axis=0), label=label)
        
        # lower_bound = br.mean(axis=0) - br.std(axis=0)
        # upper_bound = br.mean(axis=0) + br.std(axis=0)
        # if label == 'N2':
        #     plt.fill_between(t, lower_bound, upper_bound, alpha=0.2)
    plt.legend(loc = 'upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Burst rate')
    plt.savefig(f"plots/full_experiment/burst_rates_{condition}.png")
    plt.close()


def simulate_for_alphas():
    np.random.seed(1903)

    alphas_old = list(pd.read_csv('plots/full_experiment/correlations_act_maps.csv')['alpha'])
    alphas = list(np.arange(9, 21, 1)) 
    print(alphas)

    for alpha in alphas: # TODO: change
        alpha = round(alpha, 2)
        if alpha in np.round(alphas_old, 2):
            print(alpha)
            continue
        corF, corN, corT = run_simulation(alpha)

        with open(f'plots/full_experiment/correlations_act_maps.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([alpha, corF, corN, corT])


def plot_cors():
    plt.figure()
    df = pd.read_csv('plots/full_experiment/correlations_act_maps.csv')
    print(df.columns)
    label = [r'$\langle F_2, F_3 \rangle$', r'$\langle N_1, N_2 \rangle$', r'$\langle F_2, N_2 \rangle$']
    plt.plot(df['alpha'], df[['corF', 'corN', 'corT']], label=label)
    plt.legend(loc = 'upper right')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Correlation')
    plt.title('Correlation between activation maps')
    plt.savefig(f"plots/full_experiment/correlations_act_maps.png")


def create_activation_map(len_track, n_cells, sigma, m):

    x = np.arange(0, len_track, 0.01)
    x0 = np.linspace(0, len_track, n_cells)

    activation_maps = np.zeros((n_cells, len(x)))
    activation_maps = m * gauss(x, x0, sigma)

    activity_indices = np.arange(n_cells)
    np.random.shuffle(activity_indices)
    activation_maps = activation_maps[activity_indices, :]

    return activation_maps, x0[activity_indices]


def create_mixed_map(len_track, n_cells, sigma, m, x01, a):

    x = np.arange(0, len_track, 0.01)
    x02 = np.linspace(0, len_track, len(x01))
    np.random.shuffle(x02)

    activation_maps = np.zeros((n_cells, len(x)))
    activation_maps = m * ((a) * gauss(x, x01, sigma) + (1-a) * gauss(x, x02, sigma))

    return activation_maps, x02


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((mu[:, None] - x[None, :])**2) / sig**2)


def main():
    # simulate_for_alphas()
    # plot_cors()
    np.random.seed(1903)
    alpha = 0.025
    # run_simulation_control(alpha, plot_burst = True)
    run_simulation(alpha, plot_burst = True)
    test_activation_maps()


def test_activation_maps():
    a = 0.3
    len_track = 100
    n_cells = 200
    sigma = len_track/16
    m = 1
    activation_map1, x01 = create_activation_map(len_track, n_cells, sigma, m)
    activation_map2, x02 = create_activation_map(len_track, n_cells, sigma, m)

    mixed_map, x02 = create_mixed_map(len_track, n_cells, sigma, m, x01, a)

    order1 = np.argsort(x01)
    activation_map1 = activation_map1[np.ix_(order1, np.arange(activation_map1.shape[1]))]
    activation_map2 = activation_map2[np.ix_(order1, np.arange(activation_map2.shape[1]))]
    mixed_map = mixed_map[np.ix_(order1, np.arange(mixed_map.shape[1]))]

    mean_cor_CA3 = cor_act_maps(activation_map1, mixed_map)
    print('Baseline Correlation CA3: ', mean_cor_CA3)
    print(mixed_map.shape)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()

    im = axs[0].imshow(activation_map1, aspect='auto', origin='lower', extent = [0, 100, 0, n_cells])
    fig.colorbar(im, ax=axs[0])
    axs[0].set_title('Activation Map F CA3')

    im = axs[1].imshow(mixed_map, aspect='auto', origin='lower', extent = [0, 100, 0, n_cells])
    fig.colorbar(im, ax=axs[1])
    axs[1].set_title('Activation Map N CA3')

    plt.tight_layout()
    
    plt.savefig(f"plots/full_experiment/activation_maps_test.png")


    ## TODO: Run control condition with N1 EC on 

    
if __name__ == '__main__':
    main()