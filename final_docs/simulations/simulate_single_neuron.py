import sys
sys.path.append('../../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
sys.path.append('../') 
from neuron import PyramidalCells
from helpers import *
import pickle
import multiprocessing as mp


def run_simulation(params):

    n_int_a, n_int_b, inh_plasticity = params

    basal_strengths, apical_strengths = np.linspace(0, 50, 100), np.linspace(0, 100, 100) 
    frs = np.zeros((len(basal_strengths), len(apical_strengths)))
    brs = frs.copy()

    print(f"Running with n_int_a: {n_int_a}, n_int_b: {n_int_b}")

    for i, basal_strength in enumerate(basal_strengths):
        for j, apical_strength in enumerate(apical_strengths):
            print( 'basal_strength:', basal_strength, 'apical_strength:', apical_strength)
            sr, br = run_single_neuron(apical_strength, basal_strength, n_int_a, n_int_b, inh_plasticity)
            frs[i, j], brs[i, j] = sr, br
        
    with open(f'data/single_neuron/single_neuron_inha_{n_int_a}_inhb_{n_int_b}_inh_{inh_plasticity}.pkl', 'wb') as f:
        pickle.dump((frs, brs), f)


def run_single_neuron(apical_strength, basal_strength, n_int_a=1, n_int_b=1, inh_plasticity=False):
    dt = 0.001
    t_epoch = 10
    n_cells = {'pyramidal' : 1, 'inter_a' : n_int_a, 'inter_b' : n_int_b, 'CA3' : 1} 

    pyramidal = PyramidalCells(n_cells, len_edge=0, dt=dt, seed=SEED, inh_plasticity=inh_plasticity, n_dim=1)
    pyramidal.I_a = apical_strength*np.ones((n_cells['pyramidal'], int(t_epoch/dt)+1))
    pyramidal.I_b = basal_strength*np.ones((n_cells['CA3'], int(t_epoch/dt)+1))
    pyramidal.W_CA3 = np.ones((n_cells['pyramidal'], n_cells['CA3']))
            
    pyramidal.run_one_epoch(t_epoch, plasticity=False)                          
    return pyramidal.spike_count.sum()/t_epoch, pyramidal.burst_count.sum()/t_epoch


def main():

    params = [(0,0,False), (1,1,False), (1,1,True)] # TODO: Possibly more?!
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(run_simulation, params)


if __name__ == "__main__":
    main()
