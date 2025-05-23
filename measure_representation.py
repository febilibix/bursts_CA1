import numpy as np
import matplotlib.pyplot as plt
from neuron import PyramidalCells
from scipy.spatial import ConvexHull
from scipy.special import gamma
from sklearn.decomposition import PCA
from itertools import product
import pickle
import os



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

def pca_ellipsoid_volume(data):
    # Apply PCA
    pca = PCA()
    pca.fit(data)
    
    # Get singular values (square root of eigenvalues)
    singular_values = np.sqrt(pca.explained_variance_)
    
    # Dimensionality of the data
    n = data.shape[1]
    
    # Compute the volume of the ellipsoid
    volume = (np.pi ** (n / 2)) / gamma((n / 2) + 1) * np.prod(singular_values)
    return volume


t_epoch = 1
speed = 20
len_track = 100. 
dt = 0.001
tn = len_track/speed*32
a = 0.3
n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}
len_track = 100
n_env = 50
a = 0 ## Similarity between maps
lr = 15
n_bins = 100

t_run, x_run = simulate_run(len_track=len_track, av_running_speed=speed, dt=dt, tn=tn)
x_pos = np.arange(0, len_track, 1)


methods = {
    # 'mvee' : mvee_ellipsoid_volume,
    'pca' : pca_ellipsoid_volume,
    # 'aabb' : aabb_volume,
}

plt.figure(figsize=(10, 5))

for a in np.round(np.linspace(0,1,11),1):

    pyramidal = PyramidalCells(
        n_cells = n_cells,
        len_track = len_track,
        learning_rate=lr,
        dt = dt,
        n_env = n_env # TODO: I am not sure how this parameter behaves
    )

    m_EC_orig = pyramidal.m_EC.copy()
        
    all_envs = np.zeros((n_env*x_pos.shape[0], n_cells['CA3'] )) #
    all_envs_CA1 = np.zeros((n_env*n_bins, n_cells['pyramidal'] )) #
    vols, vols_CA1 = [], []

    out_CA1 = f'plots/measure_representation/volumes_CA1_{a}.pkl'
    # if os.path.exists(out_CA1):
    #     continue

    for env_idx in range(n_env):
    
        # act = pyramidal.create_activity_pc(x_pos, len_track, pyramidal.n_cells['CA3'],
        #                                    pyramidal.mb_pc, pyramidal.m_CA3, pyramidal.all_m_CA3[env_idx], a)[:, :-5]
        
        sc, bc = pyramidal.retrieve_place_cells(t_run, x_run, new_env=env_idx, a=a, t_per_epoch=t_epoch, top_down=False, plasticity = True)
        fr, x_run_reshaped = get_firing_rates(pyramidal, sc, x_run)
        mean_firing_rates = get_activation_map(fr, m_EC_orig, x_run_reshaped, n_bins=n_bins)

        all_envs_CA1[env_idx*n_bins:(env_idx+1)*n_bins, :] = mean_firing_rates.T
        # all_envs[env_idx*x_pos.shape[0]:(env_idx+1)*x_pos.shape[0], :] = act.T
        # print(pca_ellipsoid_volume(all_envs_CA1[:(env_idx+1)*n_bins, :]))
        # print(pca_ellipsoid_volume(all_envs[:(env_idx+1)*x_pos.shape[0], :]))

        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(all_envs_CA1[:(env_idx+1)*n_bins, :], aspect='auto', cmap='hot')
        # axs[0].set_title(f'Activation map CA1 for {env_idx} environments')
        # axs[0].set_xlabel('Cell index')
        # axs[0].set_ylabel('Position bin')
        # axs[0].colorbar = plt.colorbar(axs[0].images[-1], ax=axs[0])
        # 
        # axs[1].imshow(all_envs[:(env_idx+1)*x_pos.shape[0], :], aspect='auto', cmap='hot')
        # axs[1].set_title(f'Activation map CA3 for {env_idx} environments')
        # axs[1].set_xlabel('Cell index')
        # axs[1].set_ylabel('Position bin')
        # axs[1].colorbar = plt.colorbar(axs[1].images[-1], ax=axs[1])
        # plt.savefig(f'plots/measure_representation/activation_maps.png')
        # plt.close()
        # vols.append(pca_ellipsoid_volume(all_envs[:(env_idx+1)*x_pos.shape[0], :]))
        vols_CA1.append(pca_ellipsoid_volume(all_envs_CA1[:(env_idx+1)*n_bins, :]))

    # with open(out_CA1, 'wb') as f:
    #     pickle.dump(vols_CA1, f)
    # 
    # with open(out_CA1.replace('CA1', 'CA3'), 'wb') as f:
    #     pickle.dump(vols, f)

    with open(f'plots/measure_representation/all_envs_no_EC_{a}.pkl', 'wb') as f:
        pickle.dump((all_envs_CA1), f)
    
    plt.plot((vols_CA1), label=f'similarity = {a}')

    del pyramidal
    import gc 
    gc.collect()


plt.title(f'Volume of  Ellipsoid')
plt.xlabel('Number of environments')
plt.yscale('log')
plt.ylabel('Volume')
plt.legend()
plt.savefig(f'plots/measure_representation/volume_pca_ellipsoid.png')
plt.close()
