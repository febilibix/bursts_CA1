import numpy as np 
import matplotlib.pyplot as plt
from neuron import PyramidalCells



def get_firing_rates(pyramidal, event_count, x_run):

    firing_rates = np.zeros((event_count.shape[1], 1024))
    x_run_reshaped = np.zeros(1024)
    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)
        x_run_reshaped[i] = np.mean(x_run[i * step_size:(i + 1) * step_size])

    return firing_rates, x_run_reshaped


def create_activity_pc(x, len_track, m,  m_cell): 
    sigma_pf = len_track/8

    activity = np.zeros((1, x.shape[0] + 5))
    activity[:, :-5] = m * np.exp(-0.5 * ((m_cell - x[None, :])**2) / sigma_pf**2)          

    return activity


def run_simulation(alpha = 0.05, plot_burst = False):
    lr = 1
    speed = 20
    len_track = 20. 
    dt = 0.001
    tn = len_track/speed*64
    n_cells = {'pyramidal' : 1, 'inter_a' : 1, 'inter_b' : 0, 'CA3' : 1}

    delta_xs = np.arange(0.1, 3, .1)
    inh_factors = np.concatenate((np.arange(0, 5, 1), np.arange(5, 20, 4)))
    taus =  np.logspace(-2, 0, 5)

    mean_burst_rates = np.zeros((len(inh_factors), len(delta_xs)))
    x_run = len_track/2*np.ones(int(tn/dt))


    for tau in taus:
        for i, inh_factor in enumerate(inh_factors):
            for j, delta_x in enumerate(delta_xs):

                neuron = PyramidalCells(n_cells, weights = dict(), learning_rate = lr, dt = dt)
                
                m_CA3, m_EC = neuron.mb_pc, 3*neuron.ma_pc
                ## ONLY SHIFT EC Center
                act_EC = create_activity_pc(x_run, len_track, m_EC, len_track/2 ) 
                act_CA3 = create_activity_pc(x_run, len_track, m_CA3, len_track/2 + delta_x/2) 

                neuron.W_CA3 = np.ones(1)
                neuron.W_pi_a = inh_factor*neuron.W_pi_a*np.ones(1)*1000
                neuron.W_ip_a = inh_factor*neuron.W_ip_a*np.ones(1)*6000

                neuron.I_a = lambda t: act_EC[:, int(t/dt)]
                neuron.I_b = lambda t: act_CA3[:, int(t/dt)]
                neuron.pi['tau'] = tau

                neuron.run_one_epoch(tn, plasticity=False)

                burst_rate, x_run_reshaped = get_firing_rates(neuron, neuron.burst_count, x_run)
                spike_rate, _ = get_firing_rates(neuron, neuron.spike_count, x_run)

                print('inh factor: ', inh_factor, 'delta x: ', delta_x)
                print(burst_rate.mean(), spike_rate.mean())
                mean_burst_rates[i,j] = burst_rate.mean()
                
                # plt.figure()
                # plt.plot(np.arange(len(x_run_reshaped)), burst_rate.T)
                # plt.savefig(f'plots/single_neuron/EC_constant/burst_rate_tau_{tau}.png')

        plt.figure()
        for idx, inh_factor in enumerate(inh_factors):
            plt.plot(delta_xs, mean_burst_rates[idx, :], label=f'Inh factor {inh_factor}')
        plt.xlabel('Delta x')
        plt.ylabel('Mean burst rate')
        plt.legend()
        plt.savefig(f'plots/single_neuron/EC_constant/mean_burst_rates_tau_{tau}.png')

        


if __name__ == '__main__':
    run_simulation()