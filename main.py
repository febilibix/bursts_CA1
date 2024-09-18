from neuron import PyramidalCells, sparsify_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr


def sinusoidal(t, A = 10, f = 0.05, phi = 0):
    return A*np.sin(2*np.pi*f*t + phi)


def plot_single_neuron(t_values, v_values, params_inter, bursts, events):

    fig, ax = plt.subplots(6,1, figsize = (10,8), dpi = 800)

    ax[0].set_title('Basal dynamics')
    ax[0].plot(t_values, v_values[0])

    ax[1].set_title('Apical dynamics')
    ax[1].plot(t_values, v_values[1])

    ax[2].set_title('Interneuron apical dynamics')
    ax[2].plot(t_values, np.array(v_values)[2,:,0].T)
    ax[2].plot(t_values, [params_inter['v_th']]*len(t_values), 'r--', label = 'Threshold')
    ax[2].plot(t_values, [params_inter['E_L']]*len(t_values), '--', color = 'black', label = 'Reset')
    ax[2].legend()

    ax[3].set_title('Interneuron basal dynamics')
    ax[3].plot(t_values,  np.array(v_values)[3,:,0].T)
    ax[3].plot(t_values, [params_inter['v_th']]*len(t_values), 'r--', label = 'Threshold')
    ax[3].plot(t_values, [params_inter['E_L']]*len(t_values), '--', color = 'black', label = 'Reset')
    ax[3].legend()

    ax[4].set_title('Bursts')
    ax[4].plot(t_values, bursts)

    ax[5].set_title('Events')
    ax[5].plot(t_values, events)

    plt.tight_layout()
    plt.savefig('plots/single_neuron.png')



def run_single_neuron(I_b, I_a):
    

    n_cells = {'pyramidal' : 1, 'inter' : 1}

    params_basal = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
    params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5}
    params_inter = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

    pyramidal = PyramidalCells(
                        params_basal,
                        I_b,
                        params_apical,
                        I_a,
                        params_inter,
                        n_cells,
                        W_pi = 1000*np.ones(1),
                        W_ip_a = 15000000*np.ones(1),    
                        W_ip_b = 0*np.ones(1),
                        W_pp = np.zeros(1)
                    )
    
    v0 = {
        "basal": np.ones(1)*params_basal['E_L'],
        "apical": np.ones(1)*params_apical['E_L'],
        "inter": np.ones(1)*params_inter['E_L']
        }
    
    t0 = 0
    tn = 100
    dt = .01

    t_values, v_values, I_values, events, bursts, _, _ = pyramidal.run_simulation(v0, t0, tn, dt)

    plot_single_neuron(t_values, v_values, params_inter, bursts, events)


def run_simulation_constant_inputs(basal_input, apical_input, params_basal, params_apical, params_inter):

    I_b = lambda t: np.array([basal_input])
    I_a = lambda t: np.array([apical_input])

    n_cells = {'pyramidal' : 1, 'inter' : 1}

    pyramidal = PyramidalCells(       
                        params_basal,
                        I_b,
                        params_apical,
                        I_a,
                        params_inter,
                        n_cells,
                        W_pi = np.ones(1),
                        W_ip_a = np.ones(1),    
                        W_ip_b = np.ones(1),
                        W_pp = np.zeros(1)
                    )

    v0 = {
        "basal": np.ones(1)*params_basal['E_L'],
        "apical": np.ones(1)*params_apical['E_L'],
        "inter": np.ones(1)*params_inter['E_L']
        }

    t0 = 0
    tn = 50
    dt = 0.01

    t_values, v_values, I_values, events, bursts, _, _ = pyramidal.run_simulation(v0, t0, tn, dt)

    event_rate = sum(events)/tn
    burst_rate = sum(bursts)/tn

    return event_rate[0], burst_rate[0], t_values, v_values, I_values, events, bursts


def test_basal_apical_inputs(basal_inputs, apical_inputs):

    event_rates = np.zeros((len(basal_inputs),len(apical_inputs)))
    burst_rates = np.zeros((len(basal_inputs),len(apical_inputs)))

    params_basal = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
    params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5}
    params_inter = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

    for i, basal_input in tqdm(enumerate(basal_inputs)):
        for j, apical_input in enumerate(apical_inputs):
            event_rates[i,j], burst_rates[i,j], _, _, _, _, _ = run_simulation_constant_inputs(
                basal_input, apical_input, params_basal, params_apical, params_inter)
            
    fig, ax = plt.subplots(1,3, figsize = (10,3), dpi = 800)

    burst_event_ratios = np.where(np.isnan(burst_rates/event_rates), 2, burst_rates/event_rates)

    burst_event_ratios = burst_rates/event_rates

    extent = [apical_inputs.min(), apical_inputs.max(), basal_inputs.min(), basal_inputs.max()]

    ax[0].set_title('Burst/Spike ratio')
    fig.colorbar(ax[0].imshow(burst_event_ratios, cmap='hot', origin='lower', extent=extent), ax=ax[0])
    ax[0].set_xlabel('Apical input')
    ax[0].set_ylabel('Basal input')

    ax[1].set_title('Burst rate')
    fig.colorbar(ax[1].imshow(burst_rates, cmap='hot', origin='lower', extent=extent), ax=ax[1])
    ax[1].set_xlabel('Apical input')
    ax[1].set_ylabel('Basal input')

    ax[2].set_title('Spike rate')
    fig.colorbar(ax[2].imshow(event_rates, cmap='hot', origin='lower', extent=extent), ax=ax[2])
    ax[2].set_xlabel('Apical input')
    ax[2].set_ylabel('Basal input')

    plt.tight_layout()
    plt.savefig('plots/input_space.png')


def main():
    I_b = lambda t: 4
    I_a = lambda t: 10

    run_single_neuron(I_b, I_a)

    basal_inputs = np.linspace(0, 50, 50)
    apical_inputs = np.linspace(0, 50, 50)
    # test_basal_apical_inputs(basal_inputs, apical_inputs)


if __name__ == "__main__":
    main()