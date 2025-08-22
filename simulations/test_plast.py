import sys
import numpy as np
import pickle
import multiprocessing as mp
import os
from itertools import product

sys.path.append('../')  

from neuron import PyramidalCells
from helpers import *


def run_simulation(params):
    """Run one simulation for a single (lr_inh, alpha, a) parameter triple.

    Workflow:
        - Let model learn 2 maps for the parameters provided
        - Recall both maps
        - Compute PV correlations between env 0 and env 1 activation maps.
        - Save the PV correlation vector as a pickle file and print a summary.

    Args:
        params: Tuple (lr_inh, alpha, a) where
            lr_inh: Inhibitory learning rate (eta_inh).
            alpha: Spike-driven plasticity weight.
            a: Mixing factor between old/new fields when switching environments.

    Side Effects:
        - Creates directories under 'data/test_plast/{a}/' if needed.
        - Writes a pickle file with PV correlations for this parameter triple.

    Notes:
        - If the output file already exists, the run is skipped.
        - Prints the mean PV correlation (ignoring NaNs) for quick monitoring.
    """

    lr_inh, alpha, a = params

    out_path = f'data/test_plast_test/{a}/' 
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

        fr, x_run_reshaped = get_firing_rates(pyramidal.dt, event_count, x_run)
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
    """Sweep over (lr_inh, alpha, a) combinations and parallelize runs.

    Grid:
        lr_inh: 0, 1, 2, ..., 99 (step 1)
        alpha:  0.0, 0.1, 0.2, ..., 9.9 (step 0.1)
        a:      0.2, 0.4, 0.6, 0.8

    For each a:
        - Forms the Cartesian product of lr_inh × alpha × {a}.
        - Uses a multiprocessing Pool with cpu_count() processes.
        - Maps parameter triples to run_simulation.

    Prints:
        Progress per a and per-run summaries from run_simulation.
    """

    lr_inhs = np.arange(0, 100, 1).round(1) 
    alphas = np.arange(0, 10, .1).round(1) 
    aas = [0.2, 0.4, 0.6, 0.8]  

    for a in aas:
        print(f"Running simulations for a = {a}")
        params = product(lr_inhs, alphas, [a])

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(run_simulation, params)


if __name__ == '__main__':
    main()
