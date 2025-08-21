import sys 
sys.path.append('../../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
sys.path.append('../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
from helpers import *
from neuron import PyramidalCells, CONFIGS
import pickle
import numpy as np
import multiprocessing as mp # TODO: DELETE
from itertools import product # TODO: DELETE
import os


# TODO: I will need to join the two functions somehow but that i will only do once parameters are set.

def run_single_experiment(lr=CONFIGS['1D']['eta'], lr_inh=CONFIGS['1D']['eta_inh'], alpha=CONFIGS['1D']['alpha'],
                          ma_pc=CONFIGS['1D']['ma_pc'], mb_pc=CONFIGS['1D']['mb_pc'],
                          inh_plast=True, out_name='test'):

    pvs_per_condition = {}
    sp_per_condition = {}
    all_act_maps = {}
    all_act_maps_split = {}
    all_brs = {}
    all_frs = {}
    x_runs = {}
    all_act_maps_inh = {}

    # dir_name = f"data/1d_2envs/param_tuning7/alpha_{alpha}_lr_{lr}_lr_inh_{lr_inh}_ma_pc_{ma_pc}_mb_pc_{mb_pc}_w_pi_a_{w_pi_a}_w_ip_a_{w_ip_a}"
    dir_name = f'data/1d_2envs_test2/alpha_{alpha}_lr_{lr}_mapc_{ma_pc}_mbpc_{mb_pc}_lrinh_{lr_inh}/' # TODO: Change to 1d_2envs

    if os.path.exists(f"{dir_name}/act_maps_and_pvs_inh_{out_name}.pkl"):
        print(f"Skipping {dir_name}/act_maps_and_pvs_inh_{out_name}.pkl, already exists")
        return


    for idx, condition in enumerate(['exp', 'control']):
        print(f"Running {condition}...")
        # print(f"Running {condition} {'with' if params_lr['inh_plast'] else 'without'} inhibitory plasticity and {'no ' if learn_rule == 'no_heb' else ''}Hebbian learning")
        # t_run, x_run = simulate_run(len_track, speed, dt, tn, seed=SEED)
        

        pyramidal = PyramidalCells(N_CELLS['1D'], len_edge = LEN_TRACK_1D, dt = DT, seed=SEED, inh_plasticity=inh_plast, n_dim=1)
        # pyramidal.alpha = params_lr['alpha']

        # # rng = np.random.default_rng(seed=SEED)
        pyramidal.eta = lr
        pyramidal.eta_inh = lr_inh
        pyramidal.alpha = alpha
        pyramidal.ma_pc = ma_pc
        pyramidal.mb_pc = mb_pc

        ## TODO:::::::
        pyramidal.tau_fE = .020 # s

        pyramidal.tau_fI = .020 # 
        # pyramidal.W_ip_a = w_ip_a*np.ones((N_CELLS['1D']['inter_a'], N_CELLS['1D']['pyramidal']))/(N_CELLS['1D']['pyramidal'])
        # pyramidal.W_pi_a = w_pi_a*np.ones((N_CELLS['1D']['pyramidal'], N_CELLS['1D']['inter_a']))/N_CELLS['1D']['inter_a'] # 1000
        
        activation_maps, act_maps_inh, act_maps_split = {}, {}, {}
        burst_rates, firing_rates = {}, {}
        for i, (out, params_orig) in enumerate(ENVIRONMENTS_RUNS.items()):
            t_run, x_run = simulate_run(LEN_TRACK_1D, SPEED, DT, TN_1D, seed=(1+i)*(1+idx))
            # t_run, x_run = simulate_run(LEN_TRACK_1D, SPEED, DT, TN_1D, seed=i*idx) ### TODO: THis is the version in the thesis
            
            with open(f"data/1d_2envs_test/x_run_{out}.pkl", 'wb') as f:  # TODO: Maybe change path here
                pickle.dump((t_run, x_run), f)

            print(out)
            params = params_orig.copy()
            if condition == 'control':
                params['top_down'] = True

            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a=A, t_per_epoch=T_EPOCH)
                                                            
            if out == 'F1':
                m_EC_orig, m_CA3_orig = pyramidal.m_EC, pyramidal.m_CA3
                                                    
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

            fr_inh, x_run_reshaped_inh = get_firing_rates(pyramidal.dt, pyramidal.full_spike_count_int_b, x_run)

            if out == 'F1':
                mfr_inh, m_inh = get_activation_map(fr_inh, None, x_run_reshaped_inh)
            else:
                mfr_inh = get_activation_map(fr_inh, m_inh, x_run_reshaped_inh)
            act_maps_inh[out] = mfr_inh

        ## TODO: Will I?
        ## TODO: I DON'T THINK I WILL: I will do this computation with the plots and save whole spike history here

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
        all_act_maps_inh[condition] = act_maps_inh

        # del pyramidal, t_run, x_run, event_count, fr, x_run_reshaped
        # import gc
        # gc.collect()

    # with open(f"data/1d_2envs/act_maps_and_pvs_inh_{inh_plast}.pkl", 'wb') as f:
    
    os.makedirs(dir_name, exist_ok=True)

    with open(f"{dir_name}/act_maps_and_pvs_inh_{out_name}.pkl", 'wb') as f:
        ## TODO: Check here what i will actually save, maybe actually the whole spike history and then do computation in the plotting script
        pickle.dump({'pvs_per_condition': pvs_per_condition, 'sp_per_condition':sp_per_condition, 'all_act_maps': all_act_maps, 'x_runs': x_runs, 'all_brs' : all_brs, 'all_frs': all_frs, 'all_act_maps_split': all_act_maps_split, 'all_act_maps_inh': all_act_maps_inh}, f)

    del pyramidal, t_run, x_run, event_count, fr, x_run_reshaped, all_act_maps, pvs_per_condition, sp_per_condition, all_act_maps_split, all_brs, all_frs, act_maps_inh, act_maps_split
    import gc
    gc.collect()



def run_simulation(beta = 0.01, tau_f=20, tau_intb = 0.1, alpha = 0.05, lr = 15, lr_inh = 15, ma_pc = 80, mb_pc = 32, w_ip_b = 1000, w_pi_b = 200, w_pi_a = 200, w_ip_a = 7000):
                    
    ## TODO: All these parameters should be set i guess more globally?! 
    ## MAybe just globally in this file or i import them from the helpers file
    ## or maybe I add additional config file with all the parameters

    learn_rules = {
        'normal': {'alpha': CONFIGS['1D']['alpha'], 'inh_plast': True},
        'no_inh': {'alpha': CONFIGS['1D']['alpha'], 'inh_plast': False},
        'no_heb': {'alpha': 0, 'inh_plast': True}
    }

    for learn_rule, params_lr in learn_rules.items():

        run_single_experiment(alpha=params_lr['alpha'], inh_plast=params_lr['inh_plast'], out_name=learn_rule)

   


def vary_parameters():
                        
    params = {
        # TODO: See which values show nice effect
        'lr': [0.1, 1, 10, 100, 1000],
        'lr_inh': [0.1, 1, 10, 100, 1000],
        'alpha': [0.05, 0.25, 0.75, 1, 5],
    }

    for param, values in params.items():

        params = [{param: val} for val in values]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(run_one_simulation, params)

        # for val in values: ### TODO: Do this in parallel
        #     run_single_experiment(**{param: val}, out_name=f"{param}_{val}")


def run_one_simulation(params):
    print(params)
    run_single_experiment(**params, out_name=f"{list(params.keys())[0]}_{list(params.values())[0]}")


def run_single_experiment_pt(params):
    alpha, lr, inh_lr, ma_pc, mb_pc = params
    # w_ip_b, w_pi_b = params
    for out_name, inh_plast in [('normal', True), ('no_inh', False)]:
        run_single_experiment(alpha=alpha, lr=lr, lr_inh=inh_lr, ma_pc=ma_pc, mb_pc=mb_pc, out_name=out_name, inh_plast=inh_plast)

if __name__ == "__main__":
    ## TODO: maybe you can run 4 conditions in parallel
    # vary_parameters()
    # run_simulation()
    # # vary_parameters()
    # quit()
    # tau_intbs = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009] # , 0.6, 0.7, 0.8, 0.9, 1.0]
    # tau_fs = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # alphas = [0.05, 0.025, 0.075]
    # lrs = [15, 20, 25]
    # inh_lrs = [10, 15, 20, 25]
    # ma_pcs = [160, 200, 240, 280]
    # mb_pcs = [20, 24, 28, 32]
    # w_pi_as = [50, 75, 100, 125]
    # w_ip_as = [3000, 5000, 7000]

    alphas = [0.1, 0.5, 0.7, 0.9, 1, 1.5] # TODO: I think this one is not tuned yet
    lrs = [0.5, 1, 2.5, 5, 10, 15, 20, 25]   
    ma_pcs = [180, 240, 360, 500, 750] # , 400, 800, 1200]# [5000] 
    mb_pcs = [50, 64, 75, 100] # , 64, 128, 256, 512]# [2000]
    inh_lrs = [5, 10, 15, 25, 50, 75, 100] ## TODO: MAybe even less?! probably need to tune again, last time i was running with weird normalization


    params = list(product(alphas, lrs, inh_lrs, ma_pcs, mb_pcs))
    np.random.shuffle(params)

    # params = product(wipbs, wpibs)

    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        pool.map(run_single_experiment_pt, params)
