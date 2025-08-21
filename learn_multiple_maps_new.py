import sys
# sys.path.append('../../')  # Temporary fix, will need to structure the code properly later
sys.path.append('final_docs/') 

from neuron import PyramidalCells
from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pickle


#### TODO: Write some test to see if the model can store more than one map and retrieve them correctly.

def test_similarities():
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.3 # .3  # similarity between environments
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}

    aas = np.arange(0, 1.1, 0.1)
    aas =[0.3] ## TODO: Delete

    for idx in range(0, 50):

        for lr_inh in [False, 1, 10, 100]:

            cors = {}

            inh_plast = True if lr_inh else False

            for (top_down, plasticity) in [# (False, False), 
                                           (True, True), (False, True)]:

                cors_ca3 = []
                cors[(f'{top_down}_{plasticity}')] = []

                for a in aas:

                    pyramidal = PyramidalCells(n_cells, len_edge = len_track, dt = dt, inh_plasticity=inh_plast, n_dim=1)
                    m_EC_1, m_EC_2 = pyramidal.all_m_EC[0], pyramidal.all_m_EC[1]

                    # pyramidal.W_ip_b = 2*pyramidal.W_ip_b # 2*7000*np.ones((n_cells['inter_a'], n_cells['pyramidal']))/(n_cells['pyramidal']) # 7000
                    pyramidal.W_pi_b = 2*pyramidal.W_pi_b # 2*

                    pyramidal.eta = 30
                    
                    # pyramidal.alpha = 0.01 # alpha
                    # pyramidal.eta_inh = lr_inh
                    # pyramidal.eta = 5

                    # pyramidal.W_ip_a = 7000*np.ones((n_cells['inter_a'], n_cells['pyramidal']))/(n_cells['pyramidal']) # 7000
                    # pyramidal.W_ip_b = 1000*np.random.rand(n_cells['inter_b'], n_cells['pyramidal'])/(n_cells['pyramidal']) # 4000
                    # pyramidal.W_pi_a = 200*np.ones((n_cells['pyramidal'], n_cells['inter_a']))/n_cells['inter_a'] # 200
                    # pyramidal.W_pi_b = 200*np.random.rand(n_cells['pyramidal'], n_cells['inter_b'])/n_cells['inter_b'] # 30
                
                    t_run, x_run = simulate_run(len_track, speed, dt, tn)

                    activation_maps, activation_maps_ca3 = [], []

                    

                    pyramidal.retrieve_place_cells(t_run, x_run, top_down=True, new_env=0, a=a, t_per_epoch=t_epoch)

                    if plasticity:
                        pyramidal.retrieve_place_cells(t_run, x_run, top_down=top_down, new_env=1, a=a, t_per_epoch=t_epoch)

                    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                          
                    for env in [0, 1]:

                        # if not (not plasticity and env == 1):
                        #     td = True if env == 0 else top_down
                        #     pyramidal.retrieve_place_cells(t_run, x_run, top_down=td, new_env=env, a=a, t_per_epoch=t_epoch)

                        event_count, _ = pyramidal.retrieve_place_cells(
                            t_run, x_run, top_down=False, new_env=env, a=a, t_per_epoch=t_epoch, plasticity=False
                            )
     
                        fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
                        mean_firing_rates = get_activation_map(fr, m_EC_1, x_run_reshaped)
                        m_fr_2 = get_activation_map(fr, m_EC_2, x_run_reshaped)

                        m_fr_ca3 = pyramidal.get_input_map(area='CA3', env=env, a=a)

                        activation_maps.append(mean_firing_rates)
                        activation_maps_ca3.append(m_fr_ca3)

                        im1 = plot_firing_rates(axs[0, env], mean_firing_rates, f'env {env+1} sorted by 1', vmin=None, vmax=None)
                        im2 = plot_firing_rates(axs[1, env], m_fr_2, f'env {env+1} sorted by 2', vmin=None, vmax=None)

                        # Add colorbars (legends) for each subplot
                        plt.colorbar(im1, ax=axs[0, env], fraction=0.046, pad=0.04, label='Firing rate')
                        plt.colorbar(im2, ax=axs[1, env], fraction=0.046, pad=0.04, label='Firing rate')
                    
                    plt.tight_layout()
                    plt.savefig(f'plots/test_similarities/activation_map_td_{top_down}_plast_{plasticity}_a_{a}_inhplast_{str(inh_plast)}.png')
                    plt.close(fig)
 
                    # plot_act_maps(activation_maps, activation_maps_ca3, a)
                    # plot_w_ca3(pyramidal.W_CA3, m_EC_orig, m_CA3_orig)

                    pv_corr = cor_act_maps(activation_maps[0], activation_maps[1], which='pv')
                    pv_corr_ca3 = cor_act_maps(activation_maps_ca3[0], activation_maps_ca3[1], which='pv')

                    sp_corr = cor_act_maps(activation_maps[0], activation_maps[1], which='spatial')
                    sp_corr_ca3 = cor_act_maps(activation_maps_ca3[0], activation_maps_ca3[1], which='spatial')
# 
                    # cors[(f'{top_down}_{plasticity}')].append(np.mean(pv_corr))
                    # cors_ca3.append(np.mean(pv_corr_ca3))
                    print(f"a: {a}, Inh Plast: {inh_plast}, PV Corr: {round(np.nanmean(pv_corr),3)}, CA3 PV Corr: {round(np.nanmean(pv_corr_ca3),3)}, Spatial Corr: {round(np.nanmean(sp_corr),3)}, CA3 Spatial Corr: {round(np.nanmean(sp_corr_ca3),3)}")


            ## TODO: Change this path
            # with open(f'data/test_similarities/spatial_corr_vs_similarity_inh_plast_{inh_plast}_lrinh_{lr_inh}_sim_{idx}.pkl', mode='wb') as file:
            #     pickle.dump({'aas': aas, 'cors': cors, 'cors_ca3': cors_ca3}, file)

if __name__ == '__main__':
    test_similarities()
    # run_single_experiment((0.01, 0.3, 15, 5000, 2000, 1000, 3000, 1.0, 5, 0.1))
    # run_simulation(0.01, 0.3, 15, 5000, 2000, 1000, 3000, 1.0, 5, 0.1)