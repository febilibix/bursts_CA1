import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuron import PyramidalCells
from scipy.stats import pearsonr
import csv
import pandas as pd 


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000):
    bins = np.arange(0., len_track)
    fps = 1/dt
    n_runs = int(2*tn/(len_track/av_running_speed))

    running_speed_a = running_speed_b = np.ones(n_runs) * av_running_speed
    stopping_time_a = stopping_time_b = np.ones(n_runs) * 0

    x = np.array([])
    i = 0
    while True:
        stop1 = np.ones((int(stopping_time_a[i]*fps),)) * 0.
        run_length = len(bins) * fps / running_speed_a[i]
        run1 = np.linspace(0., float(len(bins)-1), int(run_length))
        stop2 = np.ones((int(stopping_time_b[i]*fps),)) * (len(bins)-1.)
        run_length = len(bins) * fps / running_speed_b[i]
        run2 = np.linspace(len(bins)-1., 0., int(run_length))
        x = np.concatenate((x, stop1, run1, stop2, run2))
        if len(x) >= tn*fps:
            break
        i += 1

    x = x[:int(tn*fps)]
    t = np.arange(len(x))/fps

    return t, x


def get_firing_rates(pyramidal, event_count):

    firing_rates = np.zeros((event_count.shape[1], 1024))

    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)

    print(firing_rates.shape, firing_rates)
    return firing_rates


def plot_firing_rates(fig, ax, firing_rates, m_EC, out, top_down = True):

    ###### IDEALLY I WOULD SOMEHOW PUT IT IN BINS ACCORDING TO POSITION AND THEN MEAN OVER THAT

    mean_firing_rates = get_activation_map(firing_rates, m_EC, top_down = top_down)

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{out}")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")

    return fig, ax


def get_activation_map(firing_rates, m_EC, top_down = True):
    lap = firing_rates.shape[1] // 16
    mean_firing_rates = sum([firing_rates[:, i*lap:(i+1)*lap] for i in range(16)]) / 16

    half = mean_firing_rates.shape[1] // 2
    mean_firing_rates = (firing_rates[:, :half] + firing_rates[:, :-half-1:-1]) / 2

    if top_down:
        sort_TD = np.argsort(m_EC)
    else:
        # m_EC = np.zeros(m_EC.shape)
        weighted_vals = mean_firing_rates * np.arange(mean_firing_rates.shape[1])[np.newaxis, :]
        m_EC = weighted_vals.sum(axis=1) / mean_firing_rates.sum(axis=1)
        print(m_EC)
        sort_TD = np.argsort(m_EC) 
        
    mean_firing_rates = mean_firing_rates[np.ix_(sort_TD, np.arange(mean_firing_rates.shape[1]))]
    return mean_firing_rates


def run_simulations():
    lr = {True: 1e-6, False : 5e-5}
    t_epoch = 0.5
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}

    tau_b = 0.125
    tau = 2*tau_b
    I_a = 8

    n_sim = 1

    all_cors = []
    last_cors = []
    for i in range( n_sim):
        all_cors_means = []
        last_cors_means = []

        for top_down in [True, 
                         False]: 
            pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr[top_down], dt = dt)
            pyramidal.p_active = 0.3
            # pyramidal.mb_pc = (1/pyramidal.p_active)* pyramidal.mb_pc
            t_run, x_run = simulate_run(len_track, speed, dt, tn)

            event_count_learning = pyramidal.learn_place_cells(t_run, x_run, t_epoch, top_down = top_down)
            firing_rates_learning = get_firing_rates(pyramidal, event_count_learning)
            mean_firing_rates_learning = get_activation_map(firing_rates_learning, pyramidal.m_EC, top_down = top_down)

            m_EC_1, m_CA3_1 = pyramidal.m_EC, pyramidal.m_CA3
            event_count = pyramidal.retrieve_place_cells(t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False)
            firing_rates = get_firing_rates(pyramidal, event_count)
            mean_firing_rates = get_activation_map(firing_rates, pyramidal.m_EC, top_down = top_down)

            event_count = pyramidal.retrieve_place_cells(t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False)
            firing_rates1 = get_firing_rates(pyramidal, event_count)
            mean_firing_rates1 = get_activation_map(firing_rates1, pyramidal.m_EC, top_down = top_down)

            cors1 = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates, mean_firing_rates1)]
            cors1_learning = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates_learning, mean_firing_rates1)]

            cors_means = [np.nanmean(cors1)]
            cors_last_means = [np.nanmean(cors1_learning)]

            with open(f"plots/multiple_envs/correlations.csv", 'a', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow([i+1, str(top_down), 1, np.nanmean(cors1)])

            with open(f"plots/multiple_envs/correlations_last.csv", 'a', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow([i+1, str(top_down), 1, np.nanmean(cors1)])    

            for j in range(1):
                events_learning = pyramidal.learn_place_cells(t_run, x_run, t_epoch, top_down=top_down)
                firing_rates_learning = get_firing_rates(pyramidal, events_learning)
                mean_firing_rates_learning = get_activation_map(firing_rates_learning, pyramidal.m_EC, top_down = top_down)
                m_CA3_learning, m_EC_learning = pyramidal.m_CA3, pyramidal.m_EC
                
                # event_count = pyramidal.retrieve_place_cells(t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False)
                # firing_rates = get_firing_rates(pyramidal, event_count)
                pyramidal.m_CA3, pyramidal.m_EC = m_CA3_1, m_EC_1
                event_count = pyramidal.retrieve_place_cells(t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False)
                firing_rates_new = get_firing_rates(pyramidal, event_count)
                mean_firing_rates_new = get_activation_map(firing_rates_new, pyramidal.m_EC, top_down = top_down)

                cors_new = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates, mean_firing_rates_new)]
                cors_means.append(np.nanmean(cors_new))

                fig, axs = plt.subplots(2, 2, figsize=(12, 12))
                axs = axs.flatten()
                plot_firing_rates(fig, axs[0], firing_rates, pyramidal.m_EC, "Original", top_down=top_down)
                plot_firing_rates(fig, axs[1], firing_rates1, pyramidal.m_EC, "Retrieval1", top_down=top_down)
                plot_firing_rates(fig, axs[2], firing_rates_new, pyramidal.m_EC, "Retrieval2", top_down=top_down)
                plt.tight_layout()
                plt.savefig(f"plots/multiple_envs/firing_rates_{'top_down' if top_down else 'bottom_up'}.png")
                plt.close()

                quit()
                pyramidal.m_CA3, pyramidal.m_EC = m_CA3_learning, m_EC_learning
                event_count = pyramidal.retrieve_place_cells(t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False)
                firing_rates_new = get_firing_rates(pyramidal, event_count)
                mean_firing_rates_new = get_activation_map(firing_rates_new, pyramidal.m_EC, top_down = top_down)

                cors_last = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates_learning, mean_firing_rates_new)]
                cors_last_means.append(np.nanmean(cors_last))


                with open(f"plots/multiple_envs/correlations.csv", 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow([i+1, str(top_down), j+2, np.nanmean(cors_new)])

                with open(f"plots/multiple_envs/correlations_last.csv", 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow([i+1, str(top_down), j+2, np.nanmean(cors_last)])


            all_cors_means.append(cors_means)
            last_cors_means.append(cors_last_means)

        all_cors.append(all_cors_means)
        last_cors.append(last_cors_means)
        # plot_cors_directly(last_cors, all_cors, cors_means)
    
    
def plot_cors_directly(last_cors, all_cors, cors_means):
    for name, cors in {'first': all_cors, 'last': last_cors}.items():
        all_cors = np.array(cors)
        all_cors_means = np.mean(all_cors, axis = 0)
        all_cors_std = np.std(all_cors, axis = 0)

        upper_bound = all_cors_means + all_cors_std
        lower_bound = all_cors_means - all_cors_std

        print(lower_bound.shape, upper_bound.shape)


        print("Correlations:", cors_means)

        plt.figure()
        plt.plot(np.arange(1, len(cors_means)+1), all_cors_means.T, label = ['top-down', 'no top-down'])
        for i in range(2):
            plt.fill_between(np.arange(1, len(cors_means)+1), lower_bound[i, :], upper_bound[i, :], alpha = 0.3)
        plt.xlabel("Number of environments")
        plt.ylabel("Mean correlation")
        plt.title(f"Correlation with {name} environment")
        plt.legend()
        plt.savefig(f'plots/multiple_envs/correlations_{name}.png')


def plot_cors():
    cors = pd.read_csv("plots/multiple_envs/correlations_last.csv")
    top_down = cors[cors['top_down'] == True]
    no_top_down = cors[cors['top_down'] == False]
    plt.figure()
    for type, df in {'top_down': top_down, 'no top-down': no_top_down}.items():
        df = df.pivot(index = 'n_env', columns = 'n_sim', values = 'cor')
        df['mean'] = df.mean(axis = 1)
        df['std'] = df.std(axis = 1)
        df['upper'] = df['mean'] + df['std']
        df['lower'] = df['mean'] - df['std']
        plt.plot(np.arange(1, df.shape[0]+1), df['mean'].values, label = type)
        plt.fill_between(np.arange(1, df.shape[0]+1), df['lower'].values, df['upper'].values, alpha = 0.3)

    plt.legend()
    plt.xlabel("Number of environments")
    plt.ylabel("Mean correlation")
    plt.title("Mean correlation with last environment")
    plt.savefig('plots/multiple_envs/correlations_last.png')

    print(cors)


def main():
    run_simulations()
    # plot_cors()


if __name__ == '__main__':
    main()