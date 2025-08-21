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


def run_simulation(params):
    lr_inh, alpha, a = params

    out_path = f'data/test_plast/{a}/' 
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/pv_corr_lr_inh_{lr_inh}_alpha_{alpha}.pkl'

    if os.path.exists(out_file):
        print(f"Skipping {out_file}, already exists")
        return

    pyramidal = PyramidalCells(N_CELLS['1D'], len_edge = LEN_TRACK_1D, dt = DT, inh_plasticity=True, n_dim=1, seed=SEED)
    pyramidal.eta_inh = lr_inh
    pyramidal.alpha = alpha
    m_EC_1 = pyramidal.all_m_EC[0]

    t_run, x_run = simulate_run(LEN_TRACK_1D, SPEED, DT, TN_1D, seed=SEED)
    activation_maps = []

    for env in [0, 1]:

        pyramidal.retrieve_place_cells(t_run, x_run, top_down=True, new_env=env, a=a, t_per_epoch=T_EPOCH)

        event_count, _ = pyramidal.retrieve_place_cells(
            t_run, x_run, top_down=False, new_env=env, a=a, t_per_epoch=T_EPOCH, plasticity=False
            )

        fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
        mean_firing_rates = get_activation_map(fr, m_EC_1, x_run_reshaped)

        activation_maps.append(mean_firing_rates)

    pv_corr = cor_act_maps(activation_maps[0], activation_maps[1], which='pv')

    with open(out_file, mode='wb') as file:
        pickle.dump(pv_corr, file)

    print(f"lr_inh: {lr_inh}, alpha: {alpha}, a: {a}, pv_corr: {np.nanmean(pv_corr):.3f}")

    for var in list(locals().keys()):
        if var not in ['gc']:
            del locals()[var]

    import gc
    gc.collect()


def main():
    lr_inhs = np.arange(0, 200, 2).round(1) 
    alphas = np.arange(0, 20, .2).round(1)  # Example range for alpha
    aas = [0.2, 0.4, 0.6, 0.8]  # Example range for a

    for a in aas:
        print(f"Running simulations for a = {a}")
        params = product(lr_inhs, alphas, [a])

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(run_simulation, params)


if __name__ == '__main__':
    main()
