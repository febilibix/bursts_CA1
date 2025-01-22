import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuron import PyramidalCells
from scipy.stats import pearsonr
import csv
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000, regular = True):
    bins = np.arange(0., len_track)
    fps = 1/dt
    n_runs = int(2*tn/(len_track/av_running_speed))

    running_speed_a = running_speed_b = np.ones(n_runs) * av_running_speed
    stopping_time_a = stopping_time_b = np.ones(n_runs) * 0

    x = np.array([])
    i = 0

    std = 0 if regular else 5

    while True:
        stop1 = np.ones((int(stopping_time_a[i]*fps),)) * 0.
        running_speed = running_speed_a[i] + np.random.randn() * std
        run_length = len(bins) * fps / running_speed
        run1 = np.linspace(0., float(len(bins)-1), int(run_length))
        stop2 = np.ones((int(stopping_time_b[i]*fps),)) * (len(bins)-1.)
        running_speed = running_speed_b[i] + np.random.randn() * std
        run_length = len(bins) * fps / running_speed
        run2 = np.linspace(len(bins)-1., 0., int(run_length))
        x = np.concatenate((x, stop1, run1, stop2, run2))
        if len(x) >= tn*fps:
            break
        i += 1

    x = x[:int(tn*fps)]
    t = np.arange(len(x))/fps
# 
    # plt.figure()
    # plt.plot(t, x)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Position (cm)")
    # plt.title("Simulated run")
    # plt.savefig(f"plots/multiple_envs/simulated_run_{'retrieval' if regular else 'encoding'}.png")
    # plt.close()

    return t, x


def get_firing_rates(pyramidal, event_count):

    firing_rates = np.zeros((event_count.shape[1], 1024))

    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)

    return firing_rates


def plot_firing_rates(fig, ax, firing_rates, m_EC, out, top_down = True):

    ###### IDEALLY I WOULD SOMEHOW PUT IT IN BINS ACCORDING TO POSITION AND THEN MEAN OVER THAT

    mean_firing_rates = get_activation_map(firing_rates, m_EC, top_down = top_down)

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    n_active = np.sum(mean_firing_rates.sum(axis = 1) != 0)
    print(mean_firing_rates.max())
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{out}; neurons active: {n_active}")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")

    return fig, ax, n_active


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


def simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch, top_down, retrieval, new_env):
    t_run, x_run = simulate_run(len_track, speed, dt, tn, regular=True)

    if retrieval:
        event_count = pyramidal.retrieve_place_cells(t_run, x_run, new_env, a = 0, t_per_epoch = None, top_down = False)
    else:
        event_count= pyramidal.learn_place_cells(t_run, x_run, t_epoch, top_down = top_down)
    firing_rates = get_firing_rates(pyramidal, event_count)
    mean_firing_rates = get_activation_map(firing_rates, pyramidal.m_EC, top_down = top_down)

    max_per_neuron = mean_firing_rates.max(axis = 1)
    print(max_per_neuron.mean(), max_per_neuron.std(), max_per_neuron.min(), max_per_neuron.max(), mean_firing_rates.max())

    return t_run, x_run, firing_rates, mean_firing_rates


def run_simulations():
    lr = {True: 5e-1, False : 5e0}
    t_epoch = 0.5
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}

    n_sim = 10

    all_cors = []
    last_cors = []
    for i in range(n_sim):
        all_cors_means = []
        last_cors_means = []

        for top_down in [True, 
                         False]: 
            p_active = (.3, 0.6) if top_down else (.3, 0.2)
            pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr[top_down], dt = dt, p_active = p_active)

            pyramidal.alpha = 0 if top_down else pyramidal.alpha
            #                
            # print(pyramidal.CA3_act.max())

            _, _, fr_learning, mean_firing_rates_learning = simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch, 
                                                                   top_down, retrieval = False, new_env = False)
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs = axs.flatten()
            plot_firing_rates(fig, axs[0], fr_learning, pyramidal.m_EC, "Learning", top_down=top_down)
            
            m_CA3_1, m_EC_1 = pyramidal.m_CA3, pyramidal.m_EC
                       
            t_run, x_run, fr, mean_firing_rates = simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch,
                                                                  top_down, retrieval = True, new_env = False)

            # plt.figure()
            plot_firing_rates(fig, axs[1], fr, pyramidal.m_EC, "Retrieval Original", top_down=top_down)
            plt.tight_layout()
            plt.savefig(f"plots/multiple_envs/firing_rates_{'top_down' if top_down else 'bottom_up'}.png")
            plt.close()
            # 
            # plt.figure()
            # 
            # sort_TD = np.argsort(pyramidal.m_EC)
            # sort_CA3 = np.argsort(pyramidal.m_CA3)
            # W_CA3 = pyramidal.W_CA3[np.ix_(sort_TD, sort_CA3)]
            # plt.imshow(W_CA3, aspect='auto', origin='lower')
            # plt.colorbar()
            # plt.savefig(f"plots/multiple_envs/W_CA3_{pyramidal.p_active}.png")
            # 
            # plt.close()

            _, _, fr1, mean_firing_rates1 = simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch,
                                                            top_down, retrieval = True, new_env = False)
            n_active = np.sum(mean_firing_rates1.sum(axis = 1) != 0)

            cors1 = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates, mean_firing_rates1)]
            # cors1_learning = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates, mean_firing_rates1)]

            cors_means = [np.nanmean(cors1)]
            cors_last_means = [np.nan]

            with open(f"plots/multiple_envs/correlations_third_active_llr.csv", 'a', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow([i+1, str(top_down), 1, np.nanmean(cors1), n_active])

            with open(f"plots/multiple_envs/correlations_last_third_active_llr.csv", 'a', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow([i+1, str(top_down), 1, np.nan])    

            for j in range(20): # 20

                _, _, fr_l, mean_firing_rates_learning = simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch,
                                                                          top_down, retrieval = False, new_env = False)

                m_CA3_learning, m_EC_learning = pyramidal.m_CA3, pyramidal.m_EC

                _, _, fr_rt, mean_firing_rates_retriev = simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch,
                                                                          top_down, retrieval = True, new_env = False)

                pyramidal.m_CA3, pyramidal.m_EC = m_CA3_1, m_EC_1

                _, _, fr_new, mean_firing_rates_new = simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch,
                                                                       top_down, retrieval = True, new_env = False)
                
                n_active = np.sum(mean_firing_rates_new.sum(axis = 1) != 0)

                cors_new = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates, mean_firing_rates_new)]
                cors_means.append(np.nanmean(cors_new))

                # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
                # axs = axs.flatten()
                # plot_firing_rates(fig, axs[0], fr, pyramidal.m_EC, "Original", top_down=top_down)
                # plot_firing_rates(fig, axs[1], fr1, pyramidal.m_EC, "Retrieval1", top_down=top_down)
                # plot_firing_rates(fig, axs[2], fr_new, pyramidal.m_EC, "Retrieval2", top_down=top_down)
                # plt.tight_layout()
                # plt.savefig(f"plots/multiple_envs/firing_rates_{'top_down' if top_down else 'bottom_up'}.png")
                # plt.close()

                pyramidal.m_CA3, pyramidal.m_EC = m_CA3_learning, m_EC_learning
                _, _, fr_new, mean_firing_rates_new = simulate_for_env(pyramidal, len_track, speed, dt, tn, t_epoch,
                                                                       top_down, retrieval = True, new_env = False)

                cors_last = [pearsonr(row1, row2)[0] for row1, row2 in zip(mean_firing_rates_learning, mean_firing_rates_new)]
                cors_last_means.append(np.nanmean(cors_last))


                with open(f"plots/multiple_envs/correlations_third_active_llr.csv", 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow([i+1, str(top_down), j+2, np.nanmean(cors_new), n_active])

                with open(f"plots/multiple_envs/correlations_last_third_active_llr.csv", 'a', newline = '') as f:
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


def plot_cors(var = 'cor', which = 'third_active', title = 'Mean correlation with first environment'):
    cors = pd.read_csv(f"plots/multiple_envs/correlations_{which}.csv")
    top_down = cors[cors['top_down'] == True].copy()
    cors = pd.read_csv(f"plots/multiple_envs/correlations_{which[:-4]}.csv")
    no_top_down = cors[cors['top_down'] == False].copy()
    
    plt.figure()
    for type, df in {'top_down': top_down, 'no top-down': no_top_down}.items():
        df = df.pivot(index = 'n_env', columns = 'n_sim', values = var)
        # df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns = df.columns, index = df.index)
        df['mean'] = df.mean(axis = 1)
        df['std'] = df.std(axis = 1)
        df['upper'] = df['mean'] + df['std']
        df['lower'] = df['mean'] - df['std']
        plt.plot(np.arange(1, df.shape[0]+1), df['mean'].values, label = type)
        plt.fill_between(np.arange(1, df.shape[0]+1), df['lower'].values, df['upper'].values, alpha = 0.3)

    plt.legend()
    plt.xlabel("Number of environments")
    ylab = "Mean correlation" if var == 'cor' else "Number of active neurons"
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(f'plots/multiple_envs/correlations_{which}_{var}.png')


def main():
    run_simulations()
    plot_cors('cor', which = 'third_active_llr', title = 'Mean correlation with first environment')
    plot_cors('n_active', which = 'third_active_llr', title = 'Number of active neurons')
    plot_cors('cor', which = 'last_third_active_llr', title = 'Mean correlation with last environment')
# 

if __name__ == '__main__':
    main()