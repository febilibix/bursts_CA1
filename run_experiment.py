import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from neuron import PyramidalCells
import csv
import pandas as pd


ENVIRONMENTS_RUNS = {
    # "F1": {'new_env': False, 'top_down': False, 'a': 0},
    "F2": {'new_env': False, 'top_down': False},
    "N1": {'new_env': True,  'top_down': False},
    "F3": {'new_env': False, 'top_down': True },
    "N2": {'new_env': True,  'top_down': True },
    }


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000):
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


def get_activation_map(firing_rates, m_EC, x_run_reshaped):
    sort_TD = np.argsort(m_EC)
    sorted_fr = firing_rates[np.ix_(sort_TD, np.arange(firing_rates.shape[1]))]

    print(sorted_fr.shape, x_run_reshaped.shape)

    out = np.zeros((sorted_fr.shape[0], 32))
    position_bins = np.linspace(0, x_run_reshaped.max(), 32)

    for idx, pos in enumerate(x_run_reshaped):
        bin_idx = np.argmin(np.abs(position_bins - pos))
        if out[:, bin_idx].sum() == 0:
            out[:, bin_idx] = sorted_fr[:, idx] 
        else: 
            out[:, bin_idx] = (out[:, bin_idx] + sorted_fr[:, idx]) / 2

    return out


def plot_firing_rates(fig, ax, mean_firing_rates, out):

    

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{out}")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")

    return fig, ax


# def plot_burst_collector(pyramidal, out ='F1'):
# 
#     bc = np.array(pyramidal.burst_collector)
#     if len(bc) > 0:
#         print(bc.min(), bc.max(), bc.mean(), bc.std())
#         plt.figure()
#         plt.hist(bc, bins=8)
#         plt.title(f"F1 - All Burst Values")
#         plt.savefig(f"plots/full_experiment/burst_collector_{out}.png")
#         plt.close()
#     pyramidal.burst_collector = []
# 
#     return pyramidal
# 

def cor_act_maps(act_map1, act_map2):
    cor = np.zeros(act_map1.shape[0])
    for i in range(act_map1.shape[0]):
        cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]

    return cor.mean()


def run_simulation(alpha = 0.05, plot_burst = False):
    lr = 1
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.6 # 1 - similarity between environments
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}
    cutoff = 0.004
    sigmoid_params = {'a': 0.016, 'k': 1000, 'x0': 0.008}


    pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr, dt = dt)
    pyramidal.alpha = alpha
    pyramidal.cutoff = cutoff
    pyramidal.sigmoid_params = sigmoid_params

    t_run, x_run = simulate_run(len_track, speed, dt, tn)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    print('F1')
    _, burst_count = pyramidal.learn_place_cells(t_run, x_run, t_epoch)
    burst_collectors = {'F1': pyramidal.burst_collector}
    pyramidal.burst_collector = []
    br, _ = get_firing_rates(pyramidal, burst_count, x_run)
    # pyramidal = plot_burst_collector(pyramidal, 'F1')
    activation_maps = {}
    burst_rates = {'F1': br}

    for i, (out, params) in enumerate(ENVIRONMENTS_RUNS.items()):
        print(out)
        event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = 0.6, t_per_epoch=t_epoch)
        burst_collectors[out] = pyramidal.burst_collector
        pyramidal.burst_collector = []
        fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
        br, _ = get_firing_rates(pyramidal, burst_count, x_run)
        mean_firing_rates = get_activation_map(fr, pyramidal.m_EC, x_run_reshaped)
        activation_maps[out] = mean_firing_rates
        burst_rates[out] = br
        fig, axs[i] = plot_firing_rates(fig, axs[i], mean_firing_rates, out)
        # pyramidal = plot_burst_collector(pyramidal, out)

    mean_cor_F = cor_act_maps(activation_maps['F2'], activation_maps['F3'])
    mean_cor_N = cor_act_maps(activation_maps['N1'], activation_maps['N2'])
    mean_cor_T = cor_act_maps(activation_maps['F2'], activation_maps['N1'])
    print(f"Mean correlation between F1 and F2: {mean_cor_F}")
    print(f"Mean correlation between N1 and N2: {mean_cor_N}")
    print(f"Mean correlation between F2 and N1: {mean_cor_T}")

    with open(f'plots/full_experiment/cors_per_cutoff.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([cutoff, mean_cor_F, mean_cor_N, mean_cor_T])

    plt.tight_layout()
    plt.savefig(f"plots/full_experiment/firing_rates_CA1.png")
    plt.close()

    if plot_burst:
        plot_burst_rates(burst_rates, tn)
        plot_burst_collector(burst_collectors, sigmoid_params)

    return mean_cor_F, mean_cor_N, mean_cor_T


def plot_burst_rates(burst_rates, tn):
    key = list(burst_rates.keys())[0]
    print(burst_rates[key].shape[1])
    t = np.arange(burst_rates[key].shape[1])*tn / (burst_rates[key].shape[1])
    plt.figure()
    for label, br in burst_rates.items():
        plt.plot(t, br.mean(axis=0), label=label)
    plt.legend(loc = 'upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Burst rate')
    plt.savefig(f"plots/full_experiment/burst_rates.png")
    plt.close()


def plot_burst_collector(burst_collectors, sigmoid_params = {'a': 0.01, 'k': 10000, 'x0': 0.005}):
    
    for label, bc in burst_collectors.items():
        if len(bc) > 100:
            fig, ax1 = plt.subplots()

            # Plot the histogram on the first y-axis
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency', color='blue')
            ax1.hist(bc, bins=8, label=f"{label} Histogram", color='blue', alpha=0.6)
            ax1.tick_params(axis='y', labelcolor='blue')

            # Create a second y-axis
            ax2 = ax1.twinx()
            x = np.arange(0, max(bc) + 0.5*np.std(bc), 0.0001)
            ax2.set_ylabel('Sigmoid', color='red')
            ax2.plot(x, sigmoid(x, **sigmoid_params), label=f"{label} Sigmoid", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            # ax2.set_ylim(0, 0.015)

            # Add a shared legend
            
            plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
            plt.title('Burst Values')
            fig.tight_layout() 

            plt.savefig(f"plots/full_experiment/burst_collector_{label}.png")


def test():
    # Create some mock data
    t = np.arange(0.01, 10.0, 0.01)
    x = np.random.randn(100)
    data1 = np.exp(t)
    data2 = np.sin(2 * np.pi * t)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.hist(x, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('plots/full_experiment/test.png')



def sigmoid(x, a=1, k=1, x0=0):
    return a / (1 + np.exp(-k*(x-x0)))


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


def main():
    # simulate_for_alphas()
    # plot_cors()
    test()
    run_simulation(alpha = .1, plot_burst = True)

    
if __name__ == '__main__':
    main()