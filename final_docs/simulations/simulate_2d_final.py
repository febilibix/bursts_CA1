import sys 
sys.path.append('../../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
sys.path.append('../') 
from helpers import *
from neuron import PyramidalCells, CONFIGS
import pickle
import os
import multiprocessing as mp
from itertools import product


def run_simulation(inh_plast=1):
  
    seed = 101 

    all_runs = {}


    for i, condition in enumerate(['exp', 'cont']):
        for j, k in enumerate(ENVIRONMENTS_RUNS.keys()):
            t_run, x_run = simulate_2d_run(LEN_EDGE_2D, SPEED, DT, TN_2D, seed=seed)
            seed += 1
            all_runs[k] = (t_run, x_run)
            # TODO: uncomment lines below
            with open(f'data/2d_test2/run_{k}.pkl', 'wb') as f:
                pickle.dump((t_run, x_run), f)

    out_path = 'data/2d/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    for condition in ['exp', 
                      'cont'
                      ]:
        

        pyramidal = PyramidalCells(N_CELLS['2D'], len_edge = LEN_EDGE_2D, dt = DT, seed = seed, n_dim=2, inh_plasticity=True)


        for out in ENVIRONMENTS_RUNS.keys():
            # if os.path.exists(f'{out_path}{condition}_{out}.pkl'):
            #     print(f"Skipping {out_path}{condition}_{out}.pkl, already exists")
            #     continue
            print(f"Running {out} {condition}...")

            params = ENVIRONMENTS_RUNS[out].copy()
            if condition == 'cont': params['top_down'] = True           

            t_run, x_run = all_runs[out]

            if inh_plast != 1:
                if out == str(inh_plast).split('_')[0]: pyramidal.inh_plasticity = False
                elif str(inh_plast).split('_')[0] == 'both' and out in ['N1', 'N2']: pyramidal.inh_plasticity = False
                elif inh_plast == 0: pyramidal.inh_plasticity = False
                else: pyramidal.inh_plasticity = True

            print(f'{out}, {inh_plast}, Inh plast: {pyramidal.inh_plasticity}')
            
            event_count, burst_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = A, t_per_epoch=T_EPOCH) 

            fr, x_run_reshaped = get_firing_rates(pyramidal.dt, event_count, x_run, n_bins = 2**15, n_dim=2)
            act_map, _ = get_activation_map_2d(fr, LEN_EDGE_2D, x_run_reshaped)

            with open(f'{out_path}{condition}_{out}_inhplast_{inh_plast}_test.pkl', 'wb') as f:
                pickle.dump(act_map, f)

            del act_map, event_count, burst_count, fr, x_run_reshaped
            import gc
            gc.collect()

        del pyramidal
        import gc
        gc.collect()



if __name__ == '__main__':
    with mp.Pool(processes=50) as pool:
        # pool.map(run_simulation, ['N1_off', 'N2_off', 'both_off', 1])
        pool.map(run_simulation, [1])
 