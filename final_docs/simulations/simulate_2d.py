import sys 
sys.path.append('../../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
sys.path.append('../') 
from helpers import *
from neuron import PyramidalCells, CONFIGS
import pickle
import os
import multiprocessing as mp
from itertools import product


def run_simulation(alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a,  W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh):
    t_epoch = 1
    speed = 20 
    len_edge = 50 
    dt = 0.001
    tn = 250 # 1D Default # 250 # 1000
    n_cells = N_CELLS['2D']
     
    seed = 101 # TODO: This will need to be set globally
    ### TODO: It worked with 98 but i feel like i will need to use the global one, check !!!!!

    all_runs = {}

    ### TODO: 

    for i, condition in enumerate(['exp', 'cont']):
        for j, k in enumerate(ENVIRONMENTS_RUNS.keys()):
            t_run, x_run = simulate_2d_run(len_edge, speed, DT, tn, seed=seed)
            seed += 1
            all_runs[k] = (t_run, x_run)
            # TODO: uncomment lines below
            with open(f'data/2d_test2/run_{k}.pkl', 'wb') as f:
                pickle.dump((t_run, x_run), f)

    # out_path = f'data/2d/alpha_{alpha}_a_{a}_lr_{lr}_ma_{ma_pc}_mb_{mb_pc}_W_pi_a_{W_pi_a}_W_ip_a_{W_ip_a}_W_pi_b_{W_pi_b}_W_ip_b_{W_ip_b}_tau_a_{tau_a}_beta_{beta}_inhlr_{inh_lr}_inh_True/'
    out_path = f'data/2d_test5/alpha_{alpha}_lr_{lr}_mapc_{ma_pc}_mbpc_{mb_pc}_lrinh_{inh_lr}/'
    # out_path = 'data/2d_test3/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ### TODO: I think I will be able to run exp and control in parallel but while parameter tuning is done sequentially.
    for condition in ['exp', 
                      'cont'
                      ]:
        # np.random.seed(seed)

        pyramidal = PyramidalCells(n_cells, len_edge = len_edge, dt = DT, seed = seed, n_dim=2, inh_plasticity=True)
        # 
        pyramidal.eta = lr
        pyramidal.eta_inh = inh_lr
        pyramidal.alpha = alpha
        pyramidal.ma_pc = ma_pc
        pyramidal.mb_pc = mb_pc

        ## TODO:::::::
        pyramidal.tau_fE = 0.1 # s - Updated to biological range

        pyramidal.tau_fI = 0.1 # s - Updated to biological range 
        # pyramidal.W_ip_a = W_ip_a*np.ones((n_cells['inter_a'], n_cells['pyramidal']))/n_cells['pyramidal']
        # pyramidal.W_pi_a = W_pi_a*np.ones((n_cells['pyramidal'], n_cells['inter_a']))/n_cells['inter_a'] # 1000
        # pyramidal.W_ip_b = W_ip_b*np.ones((n_cells['inter_b'], n_cells['pyramidal']))/n_cells['pyramidal']
        # pyramidal.W_pi_b = W_pi_b*np.ones((n_cells['pyramidal'], n_cells['inter_b']))/n_cells['inter_b'] # 200
        # pyramidal.pa['tau'] = tau_a
        # pyramidal.constant_depression_term = beta
        # pyramidal.pib['tau'] = tau_inh

        # pyramidal.pib['tau'] = 0.01

        for out in ENVIRONMENTS_RUNS.keys():
            if os.path.exists(f'{out_path}{condition}_{out}.pkl'):
                print(f"Skipping {out_path}{condition}_{out}.pkl, already exists")
                continue
            print(f"Running {out} {condition}...")

            params = ENVIRONMENTS_RUNS[out].copy()
            if condition == 'cont': params['top_down'] = True           

            t_run, x_run = all_runs[out]
            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = a, t_per_epoch=T_EPOCH) 

            fr, x_run_reshaped = get_firing_rates(pyramidal.dt, event_count, x_run, n_bins = 2**15, n_dim=2)
            act_map, _ = get_activation_map_2d(fr, len_edge, x_run_reshaped)
            burst_prop = burst_count.sum(axis = 0)/event_count.sum(axis = 0) # TODO: Will I use this?

            act_map_EC1 = pyramidal.get_input_map(area='EC', env=0, a=a, n_bins = 15)
            act_map_EC2 = pyramidal.get_input_map(area='EC', env=1, a=a, n_bins = 15)

            with open(f'{out_path}{condition}_{out}.pkl', 'wb') as f:
                pickle.dump((act_map, burst_prop), f)

            del act_map, burst_prop, event_count, burst_count, fr, x_run_reshaped
            import gc
            gc.collect()

        del pyramidal
        import gc
        gc.collect()



def run_F1(alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a,  W_pi_b, W_ip_b, tau_a, inh_lr, beta):
    t_epoch = 1
    speed = 30
    len_edge = 50 # 50
    dt = 0.001
    tn = 300 # 1000
    n_cells =  {'pyramidal' : 2500, 'inter_a' : 250, 'inter_b' : 250, 'CA3' : 1444} # {'pyramidal' : 900, 'inter_a' : 90, 'inter_b' : 90, 'CA3' : 484}
     
    seed = 98 # TODO: This will need to be set globally

    all_runs = {}

    for k in ['F1']:
        t_run, x_run = simulate_2d_run(len_edge, speed, dt, tn)
        all_runs[k] = (t_run, x_run)
        # plot_2d_run(t_run, x_run, k) # TODO: DO I NEED THIS?! MAYBE WILL USE FOR FIGURE 2 TO KIND OF SHOW THE RUNS. THEN I MIGHT WANT TO COLOR THEM IN SOME WAY SHOWING WHETHER IT IS FAM OR NOV

    out_path = f'data/2d/F1/alpha_{alpha}_a_{a}_lr_{lr}_ma_{ma_pc}_mb_{mb_pc}_W_pi_a_{W_pi_a}_W_ip_a_{W_ip_a}_W_pi_b_{W_pi_b}_W_ip_b_{W_ip_b}_tau_a_{tau_a}_beta_{beta}_inhlr_{inh_lr}_inh_False/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ### TODO: I think I will be able to run exp and control in parallel but while parameter tuning is done sequentially.
    for condition in ['exp', 
                      'cont'
                      ]:
        np.random.seed(seed)

        pyramidal = PyramidalCells(n_cells, len_edge = len_edge, dt = dt, seed = seed, n_dim=2, inh_plasticity=False)

        pyramidal.eta = lr
        pyramidal.eta_inh = inh_lr
        pyramidal.alpha = alpha
        pyramidal.ma_pc = ma_pc
        pyramidal.mb_pc = mb_pc
        pyramidal.W_ip_a = W_ip_a*np.ones((n_cells['inter_a'], n_cells['pyramidal']))/n_cells['pyramidal']
        pyramidal.W_pi_a = W_pi_a*np.ones((n_cells['pyramidal'], n_cells['inter_a']))/n_cells['inter_a'] # 1000
        pyramidal.W_ip_b = W_ip_b*np.ones((n_cells['inter_b'], n_cells['pyramidal']))/n_cells['pyramidal']
        pyramidal.W_pi_b = W_pi_b*np.ones((n_cells['pyramidal'], n_cells['inter_b']))/n_cells['inter_b'] # 200
        pyramidal.pa['tau'] = tau_a
        pyramidal.constant_depression_term = beta

        # pyramidal.pib['tau'] = 0.01

        for out in ['F1']:
            print(f"Running {out} {condition}...")    

            t_run, x_run = all_runs[out]
            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, top_down=True, a = a, t_per_epoch=t_epoch) 

            fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run, n_bins = 2**14, n_dim=2)
            act_map, _ = get_activation_map_2d(fr, len_edge, x_run_reshaped)



            with open(f'{out_path}{condition}_{out}.pkl', 'wb') as f:
                pickle.dump((act_map, pyramidal.m_EC), f)

            del act_map, event_count, burst_count, fr, x_run_reshaped
            import gc
            gc.collect()



def run_single_experiment(params):
    alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh = params

    run_simulation(alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a,  W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh)
    # run_F1(alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a,  W_pi_b, W_ip_b, tau_a, inh_lr, beta)



if __name__ == '__main__':
    # run_simulation()

    alphas = [0.1, 0.05, 0.25, 0.7, 1.5, 5] # TODO: I think this one is not tuned yet
    lrs = [1, 10,15, 25, 50, 75]   
    ma_pcs = [180, 360, 500, 750] # , 400, 800, 1200]# [5000] 
    mb_pcs = [64, 100, 128] # , 64, 128, 256, 512]# [2000]
    inh_lrs = [1, 5, 10, 15, 25, 50, 75] # Scaled by 5x to maintain iSTDP curve area with tau=0.1s
    W_ip_a = [1000]
    W_pi_a = [500] 
    W_ip_b = [4000]
    W_pi_b = [30] 
    tau_a = [1.0]
    betas = [0.1] ## TODO: This one could still be tuned.  probably need to tune again, last time i was running with weird normalization
    tau_inh = [0.1]
    aas = [0.3]

    param_combinations = list(product(alphas, aas, lrs, ma_pcs, mb_pcs, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, inh_lrs, betas, tau_inh))
    np.random.shuffle(param_combinations)
    # run_single_experiment(param_combinations[0])  # Run the first combination to test
    # quit()

    with mp.Pool(processes=50) as pool:
        pool.map(run_single_experiment, param_combinations)