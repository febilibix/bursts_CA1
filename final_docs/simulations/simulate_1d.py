import sys 
sys.path.append('../../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
sys.path.append('../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
from helpers import *
from neuron import PyramidalCells
import pickle
import numpy as np
import multiprocessing as mp # TODO: DELETE
from itertools import product # TODO: DELETE


def run_simulation():

    ## TODO: All these parameters should be set i guess more globally?! 
    ## MAybe just globally in this file or i import them from the helpers file
    ## or maybe I add additional config file with all the parameters
    
    t_epoch = 1
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.3
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}

    t_run, x_run = simulate_run(len_track, speed, dt, tn)

    pyramidal = PyramidalCells(n_cells, len_edge = len_track, dt = dt, seed=SEED, inh_plasticity=False, n_dim=1)
    pyramidal.alpha = 1
            
    event_count, _ = pyramidal.retrieve_place_cells(t_run, x_run, t_per_epoch=t_epoch, top_down=False)                          
    fr, x_run_reshaped = get_firing_rates(pyramidal, event_count, x_run)
    act_map = get_activation_map(fr, None, x_run_reshaped)


    with open(f"data/1d/act_maps_hebbian.pkl", 'wb') as f:
        pickle.dump(act_map, f)


if __name__ == "__main__":
    run_simulation()
   