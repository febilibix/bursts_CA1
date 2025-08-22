import sys
import numpy as np
import pickle
import multiprocessing as mp
import os
from itertools import product

sys.path.append('../')

from neuron import PyramidalCells
from helpers import *

def run_for_a(a, top_down, plasticity, inh_plast, lr_inh, idx):

    """Run one simulation for a given similarity 'a' and condition flags.

    Args:
        a: Mixing factor between old and new place fields (0 = old only, 1 = new only).
        top_down: Whether EC top-down input is present for env=1.
        plasticity: If True, allow plasticity updates; if False, freeze weights.
        inh_plast: Enable inhibitory plasticity (True/False).
        lr_inh: Inhibitory learning rate (ignored if inh_plast=False).
        idx: Random seed index (used for both rng and output path disambiguation).

    Workflow:
        - Builds output directory based on parameters.
        - Initializes PyramidalCells model with given inhibitory plasticity.
        - Runs one trajectory (simulate_run).
        - For each environment (0,1):
            * If plasticity allowed, run with chosen top_down setting.
            * Always run a plasticity-free pass (top_down=False) to collect events.
            * Compute firing rate activation maps (CA1) and CA3 input maps.
        - Compute PV correlations across env 0 and env 1 for both CA1 and CA3.
        - Saves results as pickle: (pv_corr, pv_corr_ca3).

    Side Effects:
        - Creates output directories if missing.
        - Writes pickles to disk.
        - Prints summary with mean PV correlations.
    """

    out_path = f'data/test_similarities/pv_corr_vs_similarity_a_inh_plast_{inh_plast}_lrinh_{lr_inh}_sim_{idx}' 
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

        fr, x_run_reshaped = get_firing_rates(pyramidal.dt, event_count, x_run)
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

    """Sweep top_down/plasticity settings for given (lr_inh, a) across seeds.

    Args:
        params: Tuple (lr_inh, a)
            lr_inh: Inhibitory learning rate (0 disables inh_plast).
            a: Input similarity factor (0-1).

    Workflow:
        - Sets inh_plast True if lr_inh > 0 else False.
        - Loops over seeds idx = 1..49
        - Runs run_for_a with three conditions:
            (top_down=False, plasticity=False),
            (top_down=True, plasticity=True),
            (top_down=False, plasticity=True).
    """

    lr_inh, a = params

    for idx in range(50):

        inh_plast = True if lr_inh else False

        for (top_down, plasticity) in [(False, False), (True, True), (False, True)]:

            run_for_a(a, top_down, plasticity, inh_plast, lr_inh, idx)

 
def main():
    """Main entry point: run similarity tests in parallel.

    Parameter grid:
        lr_inhs = [False, 1, 10, 100]
        aas     = np.arange(0.0, 1.1, 0.1)

    Creates Cartesian product of lr_inhs × aas and maps each
    pair to test_similarities using multiprocessing with half
    of available CPU cores.

    Side Effects:
        - Writes pickle results per (lr_inh, a, seed, condition).
        - Prints progress per run.
    """
    
    lr_inhs = [False, 1, 10, 100]  
    aas = list(np.arange(0, 1.1, 0.1).round(1))

    params = product(lr_inhs, aas)

    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        pool.map(test_similarities, params)


if __name__ == '__main__':
    main()
