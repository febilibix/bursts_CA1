import numpy as np 
from neuron import PyramidalCells
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def create_inputs_constant(N = 50, dt = 0.01, tn = 100, m = 8, sigma = 0.01):

    ts, ws, Is = [], [], []
    I = np.random.choice([0, m], N, p = [.5, .5])
    # I = np.array([0,m])
    w = I

    for t in np.arange(0.0001,tn,dt):
        ts.append(t)
        ws.append(w)   
        Is.append(I)
        delta_w = np.random.normal(0, np.sqrt(sigma), N)
        w = I + delta_w

    return np.array(ts), np.array(ws), I


def create_inputs_sinusoidal(I_b, f = 10, N = 50, dt = 0.01, tn = 100, m_s = 6.5, m_c = 8, sigma = 0.01):

    ts, ws = [], []
    I_b_star = -(I_b - m_c)
    w = I_b

    w = 1/2*(m_s + np.sin(2*np.pi*f*0) * I_b - np.sin(2*np.pi*f*0) * I_b_star)

    for t in np.arange(0.0001,tn,dt):
        I = 1/2*(m_s + np.sin(2*np.pi*f*t) * I_b - np.sin(2*np.pi*f*t) * I_b_star)

        ts.append(t)
        ws.append(w)   
        delta_w = np.random.normal(0, np.sqrt(sigma), N)
        w = I + delta_w

    return np.array(ts), np.array(ws)


def plot_raster(events, bursts, delta_t, mean_event_rate, mean_burst_rate, tn, t, w_b, w_a, wa, wb, a_const):

    fig, axs = plt.subplots(5,1, figsize = (12,8), dpi = 800, sharex=True)

    fig.suptitle(f"w_b = {wb}, w_a = {wa}, {'apical constant' if a_const else 'basal constant'}")

    axs[0].set_title("Basal input")
    axs[0].plot(t, w_b, lw = 0.5)
    axs[0].set_ylabel("Input")

    axs[1].set_title("Apical input")
    axs[1].plot(t, w_a, lw = 0.5)
    axs[1].set_ylabel("Input")

    axs[2].set_title("Raster plot of spikes")
    axs[2].eventplot(events, linelengths = 0.5, linewidths=0.5, color = 'blue')
    axs[2].eventplot(bursts, linelengths = 0.5, linewidths=0.5, color = 'red')
    axs[2].set_ylabel("Neuron")

    axs[2].plot([], [], color='blue', label='Events')  # Empty plot for 'Events' legend
    axs[2].plot([], [], color='red', label='Bursts')  # Empty plot for 'Bursts' legend
    axs[2].legend(loc = 'upper left')

    axs[3].set_title("Mean event rate")
    axs[3].plot(np.arange(0, tn, delta_t), mean_event_rate[0], label='MER type 0')
    axs[3].plot(np.arange(0, tn, delta_t), mean_event_rate[1], label='MER type 1')
    axs[3].legend(loc = 'upper left')

    axs[4].set_title("Mean burst rate")
    axs[4].plot(np.arange(0, tn, delta_t), mean_burst_rate[0], label='MBR type 0')
    axs[4].plot(np.arange(0, tn, delta_t), mean_burst_rate[1], label='MBR type 1')

    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/raster_plot_wa_{wa}_{"ac" if a_const else "bc"}.png')


def compute_burst_prob(burst_rate, n_cycles):
    mbr = np.mean(burst_rate, axis=1)

    mbr = mbr.reshape((int(mbr.shape[0]/n_cycles), n_cycles), order = 'F').mean(axis=1)
    mbr_slices = np.split(mbr, 4)
    mbr_slices = [mbr_slices[-1]] + mbr_slices[:-1]
    mbr_slices = [np.concatenate(mbr_slices[:2]), np.concatenate(mbr_slices[2:])]

    # mbr_slices = np.split(mbr, n_cycles)
    # print(len(mbr_slices))
    burst_rates = np.array([slice[::-1] if i % 2 == 1 else slice for i, slice in enumerate(mbr_slices)])

    # print(np.array(burst_rates).shape)
    mean_burst_rate = np.array(burst_rates).mean(axis=0)

    return mean_burst_rate


def plot_burst_prob(burst_rates, bins_per_cycle, n_cycles, wa, wb, test_condition):

    plt.figure()

    for i, burst_rate in enumerate(burst_rates):
        mean_burst_rate = compute_burst_prob(burst_rate, n_cycles)

        plt.plot(np.linspace(-1, 1, int(bins_per_cycle/2)), mean_burst_rate, label = fr"$w_a = {wa[i]}$, $w_b = {wb[i]}$")
        plt.xlabel(f"Correlation")
        plt.ylabel("Burst Rate")
        plt.title(f"Burst Rates {test_condition}")

    plt.legend()
    plt.savefig(f'plots/burst_rates_{test_condition}_long_simulation.png')


def test_PyramidalCells(n_pyr, n_int_a, n_int_b, tn, n_cycles, bins_per_cycle, wa, wb, a_const = True):

    n_cells = {'pyramidal' : n_pyr, 'inter_a' : n_int_a, 'inter_b' : n_int_b}

    params_basal  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
    params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5 }
    params_inter  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

    W_ip_a = wa*np.ones((n_int_a, n_pyr))/np.sqrt(n_pyr)
    W_ip_b = wb*np.ones((n_int_b, n_pyr))/np.sqrt(n_pyr)
    W_pi_a = 1000*np.ones((n_pyr, n_int_a))/np.sqrt(n_pyr)
    W_pi_b = 1000*np.ones((n_pyr, n_int_b))/np.sqrt(n_pyr)

    weights = {'pi_a' : W_pi_a, 'pi_b' : W_pi_b, 'ip_a' : W_ip_a, 'ip_b' : W_ip_b, 
                    'pp_a' : np.zeros((n_int_a, n_int_a)), 'pp_b' : np.zeros((n_int_b, n_int_b))}
    
    m_a, m_b = 6.5, 8
        
    sigma_a, sigma_b = 2*m_a/1000, 2*m_a/1000

    # Setting the simulation time parameters 
    t0 = 0
    tn = tn
    dt = 0.01

    if a_const:
        _, w_a, I_b_no_noise = create_inputs_constant(N = n_pyr, dt = dt, tn = tn, m = m_a, sigma=sigma_b)
        I_a = lambda t: w_a[int(t/dt)-1, :]
        f_a = np.round(n_cycles/tn, 6) 
        t_b, w_b = create_inputs_sinusoidal(I_b_no_noise, f = f_a, N = n_pyr, dt = dt, tn = tn, m_s = m_b, m_c = m_a, sigma = sigma_a)
        I_b = lambda t: w_b[int(t/dt)-1, :]

    else:
        _, w_b, I_b_no_noise = create_inputs_constant(N = n_pyr, dt = dt, tn = tn, m = m_b, sigma=sigma_b)
        I_a = lambda t: w_a[int(t/dt)-1, :]
        f_a = np.round(n_cycles/tn, 6) 
        t_b, w_a = create_inputs_sinusoidal(I_b_no_noise, f = f_a, N = n_pyr, dt = dt, tn = tn, m_s = m_a, m_c = m_b, sigma = sigma_a)
        I_b = lambda t: w_b[int(t/dt)-1, :]

    neuron_type = np.where(I_b_no_noise > 0, 1, 0)

    # Setting the time step for computation of statistics
    delta_t = 1/(f_a*bins_per_cycle) 

    # Initializing the initial membrane potentials for all cells
    v0 = {
        "basal":  np.random.normal(params_basal['E_L'], sigma_b, n_cells['pyramidal']),
        "apical": np.random.normal(params_apical['E_L'], sigma_a, n_cells['pyramidal']),
        "inter_a":  params_inter['E_L'],
        "inter_b":  params_inter['E_L']
        }
    
    pyramidal = PyramidalCells(  
                    params_basal= params_basal,
                    input_basal=I_b,
                    params_apical=params_apical,
                    input_apical=I_a,
                    params_inter=params_inter,   
                    n_cells=n_cells,
                    weights=weights,
                    )

    _,_, _, event_count, burst_count, events, bursts = pyramidal.run_simulation(v0, t0, tn, dt)

    event_rate = np.zeros((int(tn/delta_t), n_pyr))
    burst_rate = np.zeros((int(tn/delta_t), n_pyr))
    totals = {'event': event_rate, 'burst': burst_rate}

    for i, t in enumerate(range(int(tn/delta_t))):
        for name, count in {'event' : event_count, 'burst' : burst_count}.items():
            total = np.sum(count[int(t*delta_t/dt):int((t+1)*delta_t/dt), :], axis = 0)
            # total = np.where(total > 0, 1, 0) # TODO: is this it?
            totals[name][i, :] = total
    
    mer0, mbr0 = np.mean(event_rate[:, neuron_type == 0], axis=1), np.mean(burst_rate[:, neuron_type == 0], axis=1)
    mer1, mbr1 = np.mean(event_rate[:, neuron_type == 1], axis=1), np.mean(burst_rate[:, neuron_type == 1], axis=1)
    mer, mbr = (mer0, mer1), (mbr0, mbr1)


    # plot_raster(events, bursts, delta_t, mer, mbr, tn, t_b, w_b, w_a, wa, wb, a_const)

    return burst_rate


def main():
    n_pyr = 50
    n_int_a, n_int_b = int(n_pyr/10), int(n_pyr/10)
    n_cycles = 4
    bins_per_cycle = 64
    tn = 10000
    Wa, Wb = [2000, 1000], [1000, 1500]

    w_values = list(range(500, 2500, 500))

    test_conditions = {
        'wa_const': ([1000]*len(w_values), w_values),
        'wb_const': (w_values, [1000]*len(w_values)),
        'sum_const': (w_values, w_values[::-1])
    }

    for test_condition, (Wa, Wb) in tqdm(test_conditions.items()):
        burst_rates = []

        for wa, wb in tqdm(zip(Wa, Wb)):
            burst_rate = test_PyramidalCells(n_pyr, n_int_a, n_int_b, tn, n_cycles, bins_per_cycle, wa, wb)
            burst_rates.append(burst_rate)
        
        plot_burst_prob(burst_rates, bins_per_cycle, n_cycles, Wa, Wb, test_condition)


if __name__ == "__main__":
    main()