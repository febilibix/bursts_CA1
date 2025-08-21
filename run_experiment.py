import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from neuron import PyramidalCells
from itertools import product
import seaborn as sns
import multiprocessing as mp
import os
import pickle



ENVIRONMENTS_RUNS = {
    "F1": {'new_env': False, 'top_down': True},
    "F2": {'new_env': False, 'top_down': False},
    "N1": {'new_env': True,  'top_down': False},
    "F3": {'new_env': False, 'top_down': True}, 
    "N2": {'new_env': True,  'top_down': True},
    }

# ENVIRONMENTS_RUNS_CONTROL = {
#    
#     "F2": {'new_env': False, 'top_down': True},
#     "N1": {'new_env': True,  'top_down': True},
#     "F3": {'new_env': False, 'top_down': True}, 
#     "N2": {'new_env': True,  'top_down': True},
#     
#     }


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000):
    bins = np.arange(0., len_track)
    fps = 1/dt

    x = np.array([])
    i = 0
    while True:
        stopping_time = np.random.uniform(0, 1, 2)
        stop1 = np.ones((int(stopping_time[0]*fps),)) * 0.
        speed = av_running_speed + np.random.randn() * 5
        speed = speed if speed > 0 else av_running_speed # ensure speed is positive
        run_length = len(bins) * fps / speed
        run1 = np.linspace(0., float(len(bins)-1), int(run_length))
        stop2 = np.ones((int(stopping_time[1]*fps),)) * (len(bins)-1.)
        speed = av_running_speed + np.random.randn() * 5
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
    cor = np.zeros(act_map1.shape[1])
    for i in range(act_map1.shape[1]):
        cor[i] = pearsonr(act_map1[:, i], act_map2[:, i])[0]

    return cor


def run_simulation(alpha = 0.2, lr = 15, ma_pc = 40, mb_pc = 32, w_ip_b = 4000, w_pi_b = 30, w_pi_a = 200, w_ip_a = 7000, plot_burst = False):
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.3 # .3  # similarity between environments
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}

    pvs_per_condition = {}
    all_act_maps = {}

    for condition in ['exp', 'control']:

        pyramidal = PyramidalCells(n_cells, len_track = len_track, learning_rate = lr, dt = dt)
        pyramidal.ma_pc, pyramidal.mb_pc = ma_pc, mb_pc
        pyramidal.alpha = 0.05 # alpha

        pyramidal.W_ip_a = w_ip_a*np.ones((n_cells['inter_a'], n_cells['pyramidal']))/(n_cells['pyramidal']) # 7000
        pyramidal.W_ip_b = w_ip_b*np.random.rand(n_cells['inter_b'], n_cells['pyramidal'])/(n_cells['pyramidal']) # 4000
        pyramidal.W_pi_a = w_pi_a*np.ones((n_cells['pyramidal'], n_cells['inter_a']))/n_cells['inter_a'] # 200
        pyramidal.W_pi_b = w_pi_a*np.random.rand(n_cells['pyramidal'], n_cells['inter_b'])/n_cells['inter_b'] # 30

        t_run, x_run = simulate_run(len_track, speed, dt, tn)

        # pyramidal = plot_burst_collector(pyramidal, 'F1')
        activation_maps = {}
        burst_rates = {}
        EC_act_maps = {}
        ca3_act_maps = {}


        for i, (out, params_orig) in enumerate(ENVIRONMENTS_RUNS.items()):
            print(out)
            params = params_orig.copy()
            if condition == 'control':
                params['top_down'] = True

            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = a, t_per_epoch=t_epoch)
            if out == 'F1':
                m_EC_orig, m_CA3_orig = pyramidal.m_EC, pyramidal.m_CA3
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
            # fig, axs[i] = plot_firing_rates(fig, axs[i], mean_firing_rates, out)
            # figb, axsb[i] = plot_firing_rates(figb, axsb[i], mean_burst_rates, out)
            # pyramidal = plot_burst_collector(pyramidal, out)

        pv_f1_f2 = cor_act_maps(activation_maps['F1'], activation_maps['F2'])
        pv_f2_n1 = cor_act_maps(activation_maps['F2'], activation_maps['N1'])
        pv_n1_n2 = cor_act_maps(activation_maps['N1'], activation_maps['N2'])

        pvs_per_condition[condition] = [pv_f1_f2, pv_f2_n1, pv_n1_n2]
        all_act_maps[condition] = activation_maps

        
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    # axs = axs.flatten()
    #   
    # for i, comp in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):
    #     plot_pv_corr_distributions(pvs_per_condition['exp'][i], pvs_per_condition['control'][i], comp[0], comp[1], axs[i])
    # 
    # plt.tight_layout()
    # plt.savefig(f"plots/full_experiment/pop_vec/pv_corr_distributions_inpb_{mb_pc}_inpa_{ma_pc}_lr_{lr}.png", dpi=300)
    # plt.close()

    

    for condition in all_act_maps.keys():
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()
        activation_maps = all_act_maps[condition]
        for i, out in enumerate(['F2', 'N1', 'F3', 'N2']):
            plot_firing_rates(fig, axs[i], activation_maps[out], out)
        plt.tight_layout()
        plt.savefig(f"plots/full_experiment/activation_maps/{condition}_act_map.png", dpi=300) # _ipa_{w_ip_a}_ipb_{w_ip_b}_pia_{w_pi_a}_pib_{w_pi_b}
        plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        axs = axs.flatten()
        im = axs[0].imshow(pyramidal.W_ip_b, aspect='auto', origin='lower')
        fig.colorbar(im, ax=axs[0])
        axs[0].set_title(f"W_ip_b")
        im = axs[1].imshow(pyramidal.W_pi_b, aspect='auto', origin='lower')
        fig.colorbar(im, ax=axs[1])
        axs[1].set_title(f"W_pi_b")
        plt.tight_layout()
        plt.savefig(f"plots/full_experiment/weights/{condition}_weights.png", dpi=300)
        plt.close()


def plot_act_maps(activation_maps, activation_maps_ca3, a):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs = axs.flatten()

    all_act_maps = activation_maps + activation_maps_ca3
    titles = ['F CA1', 'N CA1', 'F CA3', 'N CA3']

    for i, ax in enumerate(axs):

        im = ax.imshow(all_act_maps[i], aspect='auto', origin='lower', extent=[0, 100, 0, all_act_maps[i].shape[0]])
        fig.colorbar(im, ax=ax)
        ax.set_title('Activation Map ' + titles[i])


    plt.tight_layout()
    plt.savefig(f"plots/full_experiment/activation_maps/act_map_{a}.png", dpi=300)
    plt.close()


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
    plt.savefig(f"plots/full_experiment/weights/w_ca3.png", dpi=300)



def test_similarities():
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.3 # .3  # similarity between environments
    lr = 15
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}

    aas = np.arange(0, 1.1, 0.1)

    for idx in range(1, 50):

        for lr_inh in [False, 1, 10, 100]:

            cors = {}

            inh_plast = True if lr_inh else False

            for (top_down, plasticity) in [(True, False), (True, True), (False, True)]:

                cors_ca3 = []
                cors[(f'{top_down}_{plasticity}')] = []

                for a in aas:

                    pyramidal = PyramidalCells(n_cells, len_edge = len_track, learning_rate = lr, dt = dt, inh_plasticity= inh_plast)
                    
                    pyramidal.alpha = 0.05 # alpha
                    pyramidal.eta_inh = lr_inh

                    pyramidal.W_ip_a = 7000*np.ones((n_cells['inter_a'], n_cells['pyramidal']))/(n_cells['pyramidal']) # 7000
                    pyramidal.W_ip_b = 1000*np.random.rand(n_cells['inter_b'], n_cells['pyramidal'])/(n_cells['pyramidal']) # 4000
                    pyramidal.W_pi_a = 200*np.ones((n_cells['pyramidal'], n_cells['inter_a']))/n_cells['inter_a'] # 200
                    pyramidal.W_pi_b = 200*np.random.rand(n_cells['pyramidal'], n_cells['inter_b'])/n_cells['inter_b'] # 30

                
                    t_run, x_run = simulate_run(len_track, speed, dt, tn)

                    activation_maps, activation_maps_ca3 = [], []

                    pyramidal.retrieve_place_cells(t_run, x_run, top_down=True, new_env=False, a=a, t_per_epoch=t_epoch)

                    if plasticity:
                        pyramidal.retrieve_place_cells(t_run, x_run, top_down=top_down, new_env=1, a=a, t_per_epoch=t_epoch)

                    for env in [0, 1]:

                        event_count, _ = pyramidal.retrieve_place_cells(
                            t_run, x_run, top_down=False, new_env=env, a=a, t_per_epoch=t_epoch, plasticity=False
                            )
                        # ca3_act = pyramidal.full_CA3_activities.T
                        if env == 0: ## Save ordering of first env, although sorting is arbitrary, to make them comparable 
                            m_EC_orig, m_CA3_orig = pyramidal.m_EC, pyramidal.m_CA3

                        fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
                        mean_firing_rates = get_activation_map(fr, m_EC_orig, x_run_reshaped)

                        m_fr_ca3 = pyramidal.get_input_map(area='CA3', env=env, a=a)

                        activation_maps.append(mean_firing_rates)
                        activation_maps_ca3.append(m_fr_ca3)

                    # plot_act_maps(activation_maps, activation_maps_ca3, a)
                    # plot_w_ca3(pyramidal.W_CA3, m_EC_orig, m_CA3_orig)

                    pv_corr = cor_act_maps(activation_maps[0], activation_maps[1])
                    pv_corr_ca3 = cor_act_maps(activation_maps_ca3[0], activation_maps_ca3[1])

                    cors[(f'{top_down}_{plasticity}')].append(np.mean(pv_corr))
                    cors_ca3.append(np.mean(pv_corr_ca3))
                    print(f"a: {a}, Inh Plast: {inh_plast}, PV Corr: {np.mean(pv_corr)}, CA3 PV Corr: {np.mean(pv_corr_ca3)}")

            with open(f'plots/test_similarities/pv_corr_vs_similarity_inh_plast_{inh_plast}_lrinh_{lr_inh}_sim_{idx}.pkl', mode='wb') as file:
                pickle.dump({'aas': aas, 'cors': cors, 'cors_ca3': cors_ca3}, file)

        plt.figure()
        for key, cor in cors.items():
            key = key.split('_')
            plt.plot(aas, cor, label=f'CA1: EC {"on" if key[0] == "True" else "off"}, exc plast {"on" if key[1] == "True" else "off"}')

        plt.plot(aas, cors_ca3, label=f'CA3')
        plt.xlabel('Similarity (a)')
        plt.ylabel('PV Correlation')
        plt.title(f'PV Correlation vs Similarity Inh Plast {inh_plast}')
        plt.legend()
        plt.savefig(f'plots/test_similarities/pv_corr_vs_similarity_inh_plast_{inh_plast}.png', dpi=300)
        plt.close()




    
def plot_pv_corr_distributions(pv_corr_exp, pv_corr_cont, out1, out2, ax):

    color_exp, color_cont = 'green', 'gray'

    # Plot KDEs (smoothed histograms)
    sns.kdeplot(pv_corr_exp, fill=True, color=color_exp, alpha=0.6, linewidth=1.5, ax=ax)
    sns.kdeplot(pv_corr_cont, fill=True, color=color_cont, alpha=0.6, linewidth=1.5, ax=ax)

    # Plot medians as dashed lines
    ax.axvline(np.median(pv_corr_exp), color=color_exp, linestyle='--', linewidth=1.5)
    ax.axvline(np.median(pv_corr_cont), color=color_cont, linestyle='--', linewidth=1.5)

    # Labeling
    ax.set_xlabel('PV corr. coeff.')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{out1} vs {out2}')
    ax.legend([f'exp', f'control'], loc='upper right')
    # ax.set_xlim(-.2, 1)
    # .set_ylim(0, 45)
    sns.despine()
    # plt.tight_layout()
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/pv_corr_distributions_test.png', dpi=300)


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


def run_single_experiment(params):
    w_ip_b, w_pi_b, w_pi_a, w_ip_a = params
    print(f"Running experiment with w_ip_b: {w_ip_b}, w_pi_b: {w_pi_b}, w_pi_a: {w_pi_a}, w_ip_a: {w_ip_a}")
    run_simulation(w_ip_b=w_ip_b, w_pi_b=w_pi_b, w_pi_a=w_pi_a, w_ip_a=w_ip_a)
    # run_simulation_control(w_ip_b=w_ip_b, w_pi_b=w_pi_b, w_pi_a=w_pi_a, w_ip_a=w_ip_a)


def main():
    np.random.seed(1903)

    test_similarities()

    quit()
    alpha = 0.2

    w_ip_bs = [1000] # 4000
    w_pi_bs = [5] #30
    w_pi_as = [200] # 200
    w_ip_as = [7000] # 7000

    param_combinations = list(product(w_ip_bs, w_pi_bs, w_pi_as, w_ip_as))

    run_single_experiment(param_combinations[0])

    # with mp.Pool(processes=mp.cpu_count()//6) as pool:
    #     pool.map(run_single_experiment, param_combinations)
    # mb_pcs = [20,30,50,100]
    # run_simulation_control(alpha, plot_burst = True)
    # for mb_pc in mb_pcs:
    # run_simulation(alpha, plot_burst = True)

    # ma_pcs = [10,20,30,50,100]
# 
    # for ma_pc in ma_pcs:
    #     run_simulation(alpha, ma_pc=ma_pc, plot_burst = True)
    # 
    # lrs = [5, 10, 15, 20, 25, 50]
# 
    # for lr in lrs:
    #     run_simulation(alpha, lr=lr, plot_burst = True)

    # test_activation_maps()


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