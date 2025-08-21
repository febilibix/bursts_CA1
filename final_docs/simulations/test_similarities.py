import sys
sys.path.append('../../')  # Temporary fix, will need to structure the code properly later
sys.path.append('../')  # Temporary fix, will need to structure the code properly later

from neuron import PyramidalCells
from helpers import *
import numpy as np
import pickle
import multiprocessing as mp
import os
from itertools import product


def run_for_a(a, top_down, plasticity, inh_plast, lr_inh, idx):

    out_path = f'data/test_similarities_test2/pv_corr_vs_similarity_a_inh_plast_{inh_plast}_lrinh_{lr_inh}_sim_{idx}' ## TODO: Change to 1d_2envs
    os.makedirs(out_path, exist_ok=True)

    pyramidal = PyramidalCells(N_CELLS['1D'], len_edge = LEN_TRACK_1D, dt = DT, inh_plasticity=inh_plast, n_dim=1, seed=idx)
    pyramidal.eta_inh = lr_inh if inh_plast else 0
    m_EC_1 = pyramidal.all_m_EC[0]

    t_run, x_run = simulate_run(LEN_TRACK_1D, SPEED, DT, TN_1D, seed=idx)
    activation_maps, activation_maps_ca3 = [], []

    for env in [0, 1]:

        if not (not plasticity and env == 1):
            td = True if env == 0 else top_down
            if not td: 
                pyramidal.alpha = 7.5
            pyramidal.retrieve_place_cells(t_run, x_run, top_down=td, new_env=env, a=a, t_per_epoch=T_EPOCH)

        event_count, _ = pyramidal.retrieve_place_cells(
            t_run, x_run, top_down=False, new_env=env, a=a, t_per_epoch=T_EPOCH, plasticity=False
            )

        fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
        mean_firing_rates = get_activation_map(fr, m_EC_1, x_run_reshaped)

        m_fr_ca3 = pyramidal.get_input_map(area='CA3', env=env, a=a)

        activation_maps.append(mean_firing_rates)
        activation_maps_ca3.append(m_fr_ca3)

    pv_corr = cor_act_maps(activation_maps[0], activation_maps[1], which='pv')
    pv_corr_ca3 = cor_act_maps(activation_maps_ca3[0], activation_maps_ca3[1], which='pv')

    with open(f'{out_path}/a_{a}_top_down_{top_down}_plasticity_{plasticity}.pkl', mode='wb') as file:
        pickle.dump((pv_corr, pv_corr_ca3), file)

    print(f"lr_inh: {lr_inh}, a: {a}, top_down: {top_down}, plasticity: {plasticity}, pv_corr: {np.mean(pv_corr):.3f}, pv_corr_ca3: {np.mean(pv_corr_ca3):.3f}")


def test_similarities(params):

    lr_inh, a = params

    for idx in range(1, 50):    ### TODO: change to 50

        inh_plast = True if lr_inh else False

        for (top_down, plasticity) in [(False, False), (True, True), (False, True)]:

            run_for_a(a, top_down, plasticity, inh_plast, lr_inh, idx)

 
def main():
    lr_inhs = [False, 1, 10, 100]  
    aas = list(np.arange(0, 1.1, 0.1).round(1))

    params = product(lr_inhs, aas)

    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        pool.map(test_similarities, params)


if __name__ == '__main__':
    main()
