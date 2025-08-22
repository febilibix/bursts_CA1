import sys 
import pickle
import multiprocessing as mp 
import os

sys.path.append('../') 

from helpers import *
from neuron import PyramidalCells, CONFIGS


def run_single_experiment(lr=CONFIGS['1D']['eta'], lr_inh=CONFIGS['1D']['eta_inh'], alpha=CONFIGS['1D']['alpha'],
                          inh_plast=True, out_name='test'):
    
    """Run one full experiment (exp vs. control) in a 1D environment.

    For each condition:
      - Simulates CA1 pyramidal cells across environments (F1, F2, N1, F3, N2).
      - Records activation maps, firing rates, and burst rates.
      - Computes population vector (pv) and spatial correlations.
      - Stores results to disk as a pickle file.

    Args:
        lr: Learning rate for CA3→CA1 weights.
        lr_inh: Learning rate for inhibitory plasticity.
        alpha: Weight for spike-driven plasticity term.
        inh_plast: If True, enable inhibitory plasticity.
        out_name: Suffix for output file name.

    Saves:
        data/1d_2envs/act_maps_and_pvs_inh_{out_name}.pkl
    """

    pvs_per_condition = {}
    sp_per_condition = {}
    all_act_maps = {}
    all_act_maps_split = {}
    all_brs = {}
    all_frs = {}
    x_runs = {}

    dir_name = f'data/1d_2envs/' 

    for idx, condition in enumerate(['exp', 'control']):
        print(f"Running {condition}...")

        pyramidal = PyramidalCells(N_CELLS['1D'], len_edge = LEN_TRACK_1D, dt = DT, seed=SEED, inh_plasticity=inh_plast, n_dim=1)

        pyramidal.eta = lr
        pyramidal.eta_inh = lr_inh
        pyramidal.alpha = alpha

        activation_maps, act_maps_split = {}, {}
        burst_rates, firing_rates = {}, {}
        for i, (out, params_orig) in enumerate(ENVIRONMENTS_RUNS.items()):
            t_run, x_run = simulate_run(LEN_TRACK_1D, SPEED, DT, TN_1D, seed=i*idx) 
            
            with open(f"data/1d_2envs/x_run_{out}.pkl", 'wb') as f:  
                pickle.dump((t_run, x_run), f)

            print(out)
            params = params_orig.copy()
            if condition == 'control':
                params['top_down'] = True

            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a=A, t_per_epoch=T_EPOCH)
                                                            
            if out == 'F1':
                m_EC_orig = pyramidal.m_EC 
                                                    
            fr, x_run_reshaped = get_firing_rates(pyramidal.dt, event_count, x_run)
            mean_firing_rates = get_activation_map(fr, m_EC_orig, x_run_reshaped)
            activation_maps[out] = mean_firing_rates

            act_maps_split[out] = []
            for k in range(8):
                fr_split = fr[:, k*fr.shape[1]//8:(k+1)*fr.shape[1]//8]
                xrun_split = x_run_reshaped[:, k*fr.shape[1]//8:(k+1)*fr.shape[1]//8]
                mfr = get_activation_map(fr_split, m_EC_orig, xrun_split)
                act_maps_split[out].append(mfr)
                
            br, _ = get_firing_rates(pyramidal.dt, burst_count, x_run)
            burst_rates[out] = br
            firing_rates[out] = fr

        pv_f1_f2 = cor_act_maps(activation_maps['F1'], activation_maps['F2'], which='pv')
        pv_f2_n1 = cor_act_maps(activation_maps['F2'], activation_maps['N1'], which='pv')
        pv_n1_n2 = cor_act_maps(activation_maps['N1'], activation_maps['N2'], which='pv')
        pv_f3_n2 = cor_act_maps(activation_maps['F3'], activation_maps['N2'], which='pv')

        sp_f1_f2 = cor_act_maps(activation_maps['F1'], activation_maps['F2'], which='spatial')
        sp_f2_n1 = cor_act_maps(activation_maps['F2'], activation_maps['N1'], which='spatial')
        sp_n1_n2 = cor_act_maps(activation_maps['N1'], activation_maps['N2'], which='spatial')
        sp_f3_n2 = cor_act_maps(activation_maps['F3'], activation_maps['N2'], which='spatial')

        pvs_per_condition[condition] = [pv_f1_f2, pv_f2_n1, pv_n1_n2, pv_f3_n2]
        sp_per_condition[condition] = [sp_f1_f2, sp_f2_n1, sp_n1_n2, sp_f3_n2]
        all_act_maps[condition] = activation_maps
        all_act_maps_split[condition] = act_maps_split
        all_brs[condition] = burst_rates
        all_frs[condition] = firing_rates
    
    os.makedirs(dir_name, exist_ok=True)

    with open(f"{dir_name}/act_maps_and_pvs_inh_{out_name}.pkl", 'wb') as f:
        pickle.dump({'pvs_per_condition': pvs_per_condition, 'sp_per_condition':sp_per_condition,
                     'all_act_maps': all_act_maps, 'x_runs': x_runs, 'all_brs' : all_brs, 'all_frs': all_frs,
                     'all_act_maps_split': all_act_maps_split}, f)


def run_simulation():
    """Run baseline simulations for three learning-rule conditions.

    Conditions:
        - 'normal': Hebbian + inhibitory plasticity.
        - 'no_inh': Hebbian only, no inhibitory plasticity.
        - 'no_heb': Inhibitory plasticity only (alpha=0).

    Saves results via run_single_experiment for each case.
    """
                    
    learn_rules = {
        'normal': {'alpha': CONFIGS['1D']['alpha'], 'inh_plast': True},
        'no_inh': {'alpha': CONFIGS['1D']['alpha'], 'inh_plast': False},
        'no_heb': {'alpha': 0, 'inh_plast': True}
    }

    for learn_rule, params_lr in learn_rules.items():

        run_single_experiment(alpha=params_lr['alpha'], inh_plast=params_lr['inh_plast'], out_name=learn_rule)


def vary_parameters():

    """Sweep over parameter values (lr, lr_inh, alpha) in parallel.

    For each parameter:
        - Varies it over a set of values.
        - Spawns multiple processes using multiprocessing.Pool.
        - Runs run_one_simulation for each value.

    Note:
        Output files are named according to parameter and value.
    """
                        
    params = {
        'lr': [0.1, 1, 10, 100, 1000],
        'lr_inh': [0.1, 1, 10, 100, 1000],
        'alpha': [0.05, 0.25, 0.75, 1, 5],
    }

    for param, values in params.items():

        params = [{param: val} for val in values]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(run_one_simulation, params)


def run_one_simulation(params):
    """Helper to run one experiment with given parameter overrides.

    Args:
        params: Dict of {parameter_name: value}, e.g. {'lr': 10}.

    Saves:
        Pickled results with filename including parameter name and value.
    """

    print(params)
    run_single_experiment(**params, out_name=f"{list(params.keys())[0]}_{list(params.values())[0]}")


if __name__ == "__main__":
    # run_simulation()
    vary_parameters()