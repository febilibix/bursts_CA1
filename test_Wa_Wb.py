import numpy as np 
import matplotlib.pyplot as plt
from neuron import PyramidalCells, sparsify_matrix
from tqdm import tqdm
from scipy.stats import pearsonr


def white_noise(N = 50, sigma = .1, dt = 0.01, tn = 100, m = 30):

    ts, ws = [], []
    w = np.ones(N) * m
    for t in np.arange(0.0001,tn,dt):
        ts.append(t)
        ws.append(w)   
        # apply model
        delta_w = np.random.normal(0, np.sqrt(sigma), N)
        # print(delta_w)
        w = m + delta_w

    return np.array(ts), np.array(ws)


def plot_results(events, bursts, delta_t = 1, mean_event_rate = None, mean_burst_rate = None, tn = 100):

    fig, axs = plt.subplots(2,1, figsize = (10,6), dpi = 800, sharex=True)

    axs[0].set_title("Raster plot of spikes")
    axs[0].eventplot(events, linelengths = 0.5, linewidths=0.5, color = 'blue')
    axs[0].eventplot(bursts, linelengths = 0.5, linewidths=0.5, color = 'red')
    axs[0].set_ylabel("Neuron")

    axs[0].plot([], [], color='blue', label='Events')  # Empty plot for 'Events' legend
    axs[0].plot([], [], color='red', label='Bursts')  # Empty plot for 'Bursts' legend

    axs[1].plot(np.arange(0, tn, delta_t), mean_event_rate, label='Mean event rate')
    axs[1].plot(np.arange(0, tn, delta_t), mean_burst_rate, label='Mean burst rate')
    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.savefig('plots/raster_plot.png')


def run_network_simulation(n_pyr, n_int, m_b, m_a, W_pi, W_ip_a, W_ip_b, tn = 100, dt = 0.01, plot = False):
    n_cells = {'pyramidal' : n_pyr, 'inter' : n_int}

    params_basal  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
    params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5 }
    params_inter  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

    sigma_a, sigma_b = 1, 1

    # Setting the time step for computation of statistics
    delta_t = 1 # TODO: might adapt

    # Setting the simulation time parameters 
    t0 = 0
    tn = tn
    dt = 0.01

    _, w_b = white_noise(N = n_pyr, sigma = sigma_b, m = m_b, tn = tn, dt = dt)
    I_b = lambda t: w_b[int(t/dt)-1, :]

    _, w_a = white_noise(N = n_pyr, sigma = sigma_a, m = m_a, tn = tn, dt = dt)
    I_a = lambda t: w_a[int(t/dt)-1, :]

    # Initializing the initial membrane potentials for all cells
    v0 = {
        "basal":  np.random.normal(params_basal['E_L'], sigma_b, n_cells['pyramidal']),
        "apical": np.random.normal(params_apical['E_L'], sigma_a, n_cells['pyramidal']),
        "inter":  params_inter['E_L']
        }
    
    pyramidal = PyramidalCells(  
                    params_basal= params_basal,
                    input_basal=I_b,
                    params_apical=params_apical,
                    input_apical=I_a,
                    params_inter=params_inter,   
                    n_cells=n_cells,
                    W_pi = W_pi,     
                    W_ip_a = W_ip_a,  
                    W_ip_b = W_ip_b,   
                    W_pp = np.zeros((n_int, n_int))
                    )

    _,_, _, event_count, burst_count, events, bursts = pyramidal.run_simulation(v0, t0, tn, dt)

    # Computing the mean event rate and burst rate for each cell
    mean_event_rate = np.zeros(int(tn/delta_t))
    mean_burst_rate = np.zeros(int(tn/delta_t))
    event_rate = np.zeros((int(tn/delta_t), n_pyr))
    burst_rate = np.zeros((int(tn/delta_t), n_pyr))
    mean_rates = {'event': mean_event_rate, 'burst': mean_burst_rate}
    totals = {'event': event_rate, 'burst': burst_rate}

    for i, t in enumerate(range(int(tn/delta_t))):
        for name, count in {'event' : event_count, 'burst' : burst_count}.items():
            total = np.sum(count[int(t*delta_t/dt):int((t+1)*delta_t/dt), :], axis = 0)
            total = np.where(total > 0, 1, 0)
            mean_rates[name][i] = np.mean(total)
            totals[name][i, :] = total

    if plot:
        plot_results(events, bursts, delta_t, mean_event_rate, mean_burst_rate, tn = tn)

    return mean_event_rate, mean_burst_rate, event_rate, burst_rate


def test_weights_pi(W_ip_as, W_ip_bs, m_a, m_b, n_pyr, n_int):

    mean_event_rates_mean = np.zeros((len(W_ip_as),len(W_ip_bs)))
    mean_burst_rates_mean = np.zeros((len(W_ip_as),len(W_ip_bs)))

    mean_event_rates_var = np.zeros((len(W_ip_as),len(W_ip_bs)))
    mean_burst_rates_var = np.zeros((len(W_ip_as),len(W_ip_bs)))

    mean_event_rates_cor = np.zeros((len(W_ip_as),len(W_ip_bs)))
    mean_burst_rates_cor = np.zeros((len(W_ip_as),len(W_ip_bs)))

    W_ip_as = 10*W_ip_as/np.sqrt(n_pyr)
    W_ip_bs = 10*W_ip_bs/np.sqrt(n_pyr)
    W_pi    = 1000 * np.ones((n_pyr, n_int))/np.sqrt(n_pyr)
    
    for i, W_ip_a_sc in tqdm(enumerate(W_ip_as)):
        for j, W_ip_b_sc in enumerate(W_ip_bs):
            W_ip_a = W_ip_a_sc*np.ones((n_int, n_pyr))
            W_ip_b = W_ip_b_sc*np.ones((n_int, n_pyr))
            mean_event_rate, mean_burst_rate, ec, bc = run_network_simulation(n_pyr, n_int, m_b, m_a, W_pi, W_ip_a, W_ip_b, tn = 20)
            mean_event_rates_mean[i,j] = np.mean(mean_event_rate)
            mean_burst_rates_mean[i,j] = np.mean(mean_burst_rate)
            mean_event_rates_var[i,j]  = np.var(mean_event_rate)
            mean_burst_rates_var[i,j]  = np.var(mean_burst_rate)
            mean_event_rates_cor[i,j]  = np.mean(np.corrcoef(ec.T)[np.triu_indices(n_pyr, k=1)]) 
            mean_burst_rates_cor[i,j]  = np.mean(np.corrcoef(bc.T)[np.triu_indices(n_pyr, k=1)])

    statistics = {'mean' : (mean_event_rates_mean, mean_burst_rates_mean), 
                  'var'  : (mean_event_rates_var, mean_burst_rates_var),
                  'cor'  : (mean_event_rates_cor, mean_burst_rates_cor)}
    
    for stat_name, stat in statistics.items():
        fig, ax = plt.subplots(1, 2, figsize = (8,3), dpi = 800)

        extent = [W_ip_as.min(), W_ip_as.max(), W_ip_bs.min(), W_ip_bs.max()]

        ax[0].set_title(f'Mean event rate {stat_name}')
        fig.colorbar(ax[0].imshow(stat[0], cmap='hot', origin='lower', extent=extent), ax=ax[0])
        ax[0].set_xlabel(r'$W_a$')
        ax[0].set_ylabel(r'$W_b$')

        ax[1].set_title(f'Mean Burst rate {stat_name}')
        fig.colorbar(ax[1].imshow(stat[1], cmap='hot', origin='lower', extent=extent), ax=ax[1])
        ax[1].set_xlabel(r'$W_a$')
        ax[1].set_ylabel(r'$W_b$')

        plt.tight_layout()
        plt.savefig(f'plots/event_burst_rates_{stat_name}_ma_{m_a}_mb_{m_b}.png')


def main():

    n_pyr = 50
    n_int = 5

    W_ip_as = np.linspace(0, 1000, 50) 
    W_ip_bs = np.linspace(0, 1000, 50)
# 
    for (ma, mb) in [(10, 4), (6.5, 8), (4, 10)]:
        test_weights_pi(W_ip_as, W_ip_bs, ma, mb, n_pyr, n_int)


    m_b = 4
    m_a = 10
    W_ip_a = 2000*np.ones((n_int, n_pyr))/np.sqrt(n_pyr)
    W_ip_b = 1000*np.ones((n_int, n_pyr))/np.sqrt(n_pyr)
    W_pi   = 1000*np.ones((n_pyr, n_int))/np.sqrt(n_pyr) ## Normalize by normalizing the rows of W (normalize post-synaptic weights)


    print(W_pi, W_ip_a, W_ip_b)

    run_network_simulation(n_pyr, n_int, m_b, m_a, W_pi, W_ip_a, W_ip_b, tn = 200, plot = True)


if __name__ == "__main__":
    main()