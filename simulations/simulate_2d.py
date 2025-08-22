import sys 
import pickle
import os

sys.path.append('../') 

from helpers import *
from neuron import PyramidalCells, CONFIGS


def run_simulation():
    """Run 2D CA1 simulations for all environments and conditions.

    Workflow:
        - Generate random 2D trajectories (simulate_2d_run) for each environment key in ENVIRONMENTS_RUNS.
        - For each condition ('exp', 'cont'):
            * Initialize a PyramidalCells model (2D).
            * For each environment (F1, F2, N1, F3, N2):
                - Skip if output pickle already exists.
                - If control condition, force top_down=True in params.
                - Run retrieve_place_cells with given environment parameters.
                - Compute firing rates with high spatial resolution (n_bins=2**15).
                - Convert firing rates into a 2D activation map (get_activation_map_2d).
                - Save the activation map as a pickle file.

    Side Effects:
        - Creates 'data/2d_test/' directory if missing.
        - Writes one pickle per (condition, environment).

    Notes:
        - Seed is incremented per trajectory for reproducibility.
        - Activation maps are stored as numpy arrays for later analysis/plotting.
    """
   
    seed = 101 
    all_runs = {}

    for condition in ['exp', 'cont']:
        for k in ENVIRONMENTS_RUNS.keys():
            t_run, x_run = simulate_2d_run(LEN_EDGE_2D, SPEED, DT, TN_2D, seed=seed)
            seed += 1
            all_runs[k] = (t_run, x_run)

    out_path = f'data/2d/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for condition in ['exp', 'cont']:

        pyramidal = PyramidalCells(N_CELLS['2D'], len_edge = LEN_EDGE_2D, dt = DT, seed = seed, n_dim=2, inh_plasticity=True)

        for out in ENVIRONMENTS_RUNS.keys():
            if os.path.exists(f'{out_path}{condition}_{out}.pkl'):
                print(f"Skipping {out_path}{condition}_{out}.pkl, already exists")
                continue
            print(f"Running {out} {condition}...")

            params = ENVIRONMENTS_RUNS[out].copy()
            if condition == 'cont': params['top_down'] = True           

            t_run, x_run = all_runs[out]
            event_count, _ = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = A, t_per_epoch=T_EPOCH) 

            fr, x_run_reshaped = get_firing_rates(pyramidal.dt, event_count, x_run, n_bins = 2**15, n_dim=2)
            act_map, _ = get_activation_map_2d(fr, LEN_EDGE_2D, x_run_reshaped)

            with open(f'{out_path}{condition}_{out}.pkl', 'wb') as f:
                pickle.dump(act_map, f)


if __name__ == '__main__':
    run_simulation()

    