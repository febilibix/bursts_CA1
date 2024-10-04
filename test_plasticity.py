import numpy as np 
from neuron import PyramidalCells
import matplotlib.pyplot as plt
import pandas as pd


def create_inputs_constant(N = 50, dt = 0.01, tn = 100, m = 8, sigma = 0.01, choice = 'random'):

    ts, ws, Is = [], [], []
    if choice == 'random':
        I = np.random.choice([0, m], N, p = [.5, .5])
    elif choice == 'odd':
        I = np.array([m if i % 2 != 0 else 0 for i in range(N)])
    elif choice == 'even':
        I = np.array([0 if i % 2 != 0 else m for i in range(N)])

    w = I

    for t in np.arange(0.0001,tn,dt):
        ts.append(t)
        ws.append(w)   
        Is.append(I)
        delta_w = np.random.normal(0, np.sqrt(sigma), N)
        w = I + delta_w

    return np.array(ts), np.array(ws), I


def get_output_CA3(N = 50, dt = 0.01, tn = 100, m = 8, sigma = 0.01, n_patterns = 10):
    
    I_b_all = []
    choices = ['odd', 'even']

    for i in range(n_patterns):
        tn_pattern = tn//n_patterns
        choice = choices[i] if i < len(choices) else 'random'
        _, I_b, p = create_inputs_constant(N = N, dt = dt, tn = tn_pattern, m = m, sigma=sigma, choice=choice)
        I_b_all.append(I_b)

    I_b_all = np.concatenate(I_b_all, axis = 0)

    return I_b_all


def plot_raster(events, bursts, delta_t, mean_event_rate, mean_burst_rate, tn, t, I_CA3, top_down = True):

    fig, axs = plt.subplots(4,1, figsize = (12,8), dpi = 800, sharex=True)

    fig.suptitle(f"")

    axs[0].set_title("CA3 output")
    axs[0].plot(t, I_CA3, lw = 0.5)
    axs[0].set_ylabel("Input")

    axs[1].set_title("Raster plot of spikes")
    axs[1].eventplot(events, linelengths = 0.5, linewidths=0.5, color = 'blue')
    axs[1].eventplot(bursts, linelengths = 0.5, linewidths=0.5, color = 'red')
    axs[1].set_ylabel("Neuron")

    axs[1].plot([], [], color='blue', label='Events')  # Empty plot for 'Events' legend
    axs[1].plot([], [], color='red', label='Bursts')  # Empty plot for 'Bursts' legend
    axs[1].legend(loc = 'upper left')

    axs[2].set_title("Mean event rate")
    

    axs[2].plot(np.arange(0, tn, delta_t), mean_event_rate[0], label=f'MER type 0')
    axs[2].plot(np.arange(0, tn, delta_t), mean_event_rate[1], label=f'MER type 1')
    axs[2].legend(loc = 'upper left')

    axs[3].set_title("Mean burst rate")
    axs[3].plot(np.arange(0, tn, delta_t), mean_burst_rate[0], label=f'MBR type 0')
    axs[3].plot(np.arange(0, tn, delta_t), mean_burst_rate[1], label=f'MBR type 1')

    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/raster_plot_{"learning" if top_down else "recall"}.png')


def run_simulation(n_CA3 = 30, n_pyr = 50, n_int_a = 5, n_int_b = 5, tn = 1000,
                   delta_t = 1, n_patterns = 5, top_down = True, W_CA3 = None, I_a_params = None, 
                   I_CA3_arr = None):

    n_cells = {'pyramidal' : n_pyr, 'inter_a' : n_int_a, 'inter_b' : n_int_b, 'CA3' : n_CA3}

    params_basal  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
    params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5 }
    params_inter  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

    m_a, m_b = 6.5, 8*n_pyr
    sigma_a, sigma_b = 2*m_a/1000, 2*m_b/1000

    # Setting the simulation time parameters 
    t0 = 0
    tn = tn
    dt = 0.01

    if top_down:
        t_a, I_a_arr, I_a_no_noise = create_inputs_constant(N = n_pyr, dt = dt, tn = tn, m = m_a, sigma=sigma_a)
        I_a = lambda t: I_a_arr[int(t/dt)-1, :]
    else:
        t_a, I_a_arr, I_a_no_noise = I_a_params
        I_a = lambda t: np.zeros(n_pyr)

    W_ip_a = 2000*np.ones((n_int_a, n_pyr))/np.sqrt(n_pyr)
    W_ip_b = 1000*np.ones((n_int_b, n_pyr))/np.sqrt(n_pyr)
    W_pi_a = 1000*np.ones((n_pyr, n_int_a))/np.sqrt(n_pyr)
    W_pi_b = 1000*np.ones((n_pyr, n_int_b))/np.sqrt(n_pyr)

    weights = {'pi_a' : W_pi_a, 'pi_b' : W_pi_b, 'ip_a' : W_ip_a, 'ip_b' : W_ip_b, 
               'pp_a' : np.zeros((n_int_a, n_int_a)), 'pp_b' : np.zeros((n_int_b, n_int_b)),
               'CA3' : W_CA3}
    
    v0 = {
        "basal":  np.random.normal(params_basal['E_L'], sigma_a, n_cells['pyramidal']),
        "apical": np.random.normal(params_apical['E_L'], sigma_a, n_cells['pyramidal']),
        "inter_a":  params_inter['E_L'],
        "inter_b":  params_inter['E_L']
        }

    if top_down:
        I_CA3_arr = get_output_CA3(N = n_CA3, dt = dt, tn = tn, m = m_b, sigma=sigma_b, n_patterns=n_patterns)

    I_CA3 = lambda t: I_CA3_arr[int(t/dt)-1, :]

    pyramidal = PyramidalCells(  
                    params_basal=params_basal,
                    input_basal=I_CA3,
                    params_apical=params_apical,
                    input_apical=I_a,
                    params_inter=params_inter,   
                    n_cells=n_cells,
                    weights=weights,
                    plasticity=top_down 
                    )

    _, _, _, event_count, burst_count, events, bursts = pyramidal.run_simulation(v0, t0, tn, dt)

    neuron_type = np.where(I_a_no_noise > 0, 1, 0)

    event_rate = np.zeros((int(tn/delta_t), n_pyr))
    burst_rate = np.zeros((int(tn/delta_t), n_pyr))
    totals = {'event': event_rate, 'burst': burst_rate}

    for i, t in enumerate(range(int(tn/delta_t))):
        for name, count in {'event' : event_count, 'burst' : burst_count}.items():
            total = np.sum(count[int(t*delta_t/dt):int((t+1)*delta_t/dt), :], axis = 0)
            totals[name][i, :] = total
    
    mer0, mbr0 = np.mean(event_rate[:, neuron_type == 0], axis=1), np.mean(burst_rate[:, neuron_type == 0], axis=1)
    mer1, mbr1 = np.mean(event_rate[:, neuron_type == 1], axis=1), np.mean(burst_rate[:, neuron_type == 1], axis=1)
    mer, mbr = (mer0, mer1), (mbr0, mbr1)

    plot_raster(events, bursts, delta_t, mer, mbr, tn, t_a, I_CA3_arr, top_down)

    return pyramidal.W_CA3, t_a, I_a_arr, I_a_no_noise, I_CA3_arr


def plot_weights(W_CA3, I_a_no_noise):
    neuron_type = np.where(I_a_no_noise > 0, 1, 0)

    W_CA3_plot = np.vstack([W_CA3[neuron_type == 0], W_CA3[neuron_type == 1]])

    plt.figure(figsize=(8,12))
    plt.title(f"CA3 to Pyramidal weights, N_0 = {sum(neuron_type == 0)}, N_1 = {sum(neuron_type == 1)}")
    plt.imshow(W_CA3_plot)
    plt.xlabel("CA3")
    plt.ylabel("CA1")
    plt.colorbar()
    plt.savefig('plots/W_CA3.png')

    print(np.where(I_a_no_noise > 0, 1, 0))

    W_CA3_df = pd.DataFrame(W_CA3.round(5))
    W_CA3_df.index = np.where(I_a_no_noise > 0, 1, 0)

    W_CA3_df.to_csv('W_CA3.csv', index=True)


def main():

    n_CA3 = 30
    n_pyr = 50
    n_int_a = 5
    n_int_b = 5
    tn = 1000
    delta_t = 5
    n_patterns = 50

    W_CA3, t_a, I_a_arr, I_a_no_noise, I_CA3_arr = run_simulation(
        n_CA3=n_CA3, n_pyr=n_pyr, n_int_a=n_int_a, n_int_b=n_int_b, tn=tn, delta_t=delta_t, n_patterns=n_patterns)
     
    plot_weights(W_CA3, I_a_no_noise)

    I_a = (t_a, I_a_arr, I_a_no_noise)

    run_simulation(n_CA3=n_CA3, n_pyr=n_pyr, n_int_a=n_int_a, n_int_b=n_int_b, tn=tn, delta_t=delta_t,
                   top_down = False, W_CA3 = W_CA3, I_a_params = I_a, I_CA3_arr = I_CA3_arr)
    # TODO: I guess what i need to do now is to check memory capacity of the network
    # Sth like not giving it any top down learning signal and see how many patterns it can remember 
    # I guess in terms of if it still has low burst rate if i give it the same pattern it had seen before
    


if __name__ == "__main__":
    main()

