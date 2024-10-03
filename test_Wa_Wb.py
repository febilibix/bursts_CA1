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

    fig, axs = plt.subplots(1,1, figsize = (10,6), dpi = 800, sharex=True)

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


def run_network_simulation(n_pyr, n_int_a, n_int_b, m_b, m_a, weights, tn = 100, dt = 0.01, plot = False):
    n_cells = {'pyramidal' : n_pyr, 'inter_a' : n_int_a, 'inter_b' : n_int_b}

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

    # Computing the event rate and burst rate for each cell
    event_rate = np.zeros((int(tn/delta_t), n_pyr))
    burst_rate = np.zeros((int(tn/delta_t), n_pyr))
    totals = {'event': event_rate, 'burst': burst_rate}

    for i, t in enumerate(range(int(tn/delta_t))):
        for name, count in {'event' : event_count, 'burst' : burst_count}.items():
            total = np.sum(count[int(t*delta_t/dt):int((t+1)*delta_t/dt), :], axis = 0)
            total = np.where(total > 0, 1, 0)
            totals[name][i, :] = total

    if plot:
        plot_results(events, bursts, delta_t, np.mean(event_rate, axis=1), np.mean(burst_rate, axis=1), tn = tn)

    return event_rate, burst_rate


def test_weights_pi(W_ip_as, W_ip_bs, m_a, m_b, n_pyr, n_int_a, n_int_b, post_inter = False):

    burst_measures = {m: np.zeros((len(W_ip_as),len(W_ip_bs))) for m in ['mean', 'var', 'cor']}
    event_measures = {m: np.zeros((len(W_ip_as),len(W_ip_bs))) for m in ['mean', 'var', 'cor']}
    be_r_measures = {m: np.zeros((len(W_ip_as),len(W_ip_bs))) for m in ['mean', 'var', 'cor']}

    m_fcts = {'mean' : lambda x : np.mean(np.mean(x, axis=1)),
              'var'  : lambda x: np.var(np.mean(x, axis=1)), 
              'cor'  : lambda x : np.mean(np.corrcoef(x.T)[np.triu_indices(n_pyr, k=1)])}

    W_as = 10*W_ip_as/np.sqrt(n_pyr)
    W_bs = 10*W_ip_bs/np.sqrt(n_pyr)
    W_ca = 1000/np.sqrt(n_pyr)
    W_cb = 1000/np.sqrt(n_pyr)

    for i, W_a_sc in tqdm(enumerate(W_as)):
        for j, W_b_sc in enumerate(W_bs):

            if post_inter:
                W_ip_a = W_ca*np.ones((n_int_a, n_pyr))
                W_ip_b = W_cb*np.ones((n_int_b, n_pyr))
                W_pi_a = W_a_sc*np.ones((n_pyr, n_int_a))
                W_pi_b = W_b_sc*np.ones((n_pyr, n_int_b))

            else:
                W_ip_a = W_a_sc*np.ones((n_int_a, n_pyr))
                W_ip_b = W_b_sc*np.ones((n_int_b, n_pyr))
                W_pi_a = W_ca*np.ones((n_pyr, n_int_a))
                W_pi_b = W_cb*np.ones((n_pyr, n_int_b))

            weights = {'pi_a' : W_pi_a, 'pi_b' : W_pi_b, 'ip_a' : W_ip_a, 'ip_b' : W_ip_b, 
                           'pp_a' : np.zeros((n_int_a, n_int_a)), 'pp_b' : np.zeros((n_int_b, n_int_b))}
            
            er, br = run_network_simulation(
                n_pyr, n_int_a, n_int_b, m_b, m_a, weights=weights, tn=20)
            
            for measure, fct in m_fcts.items():
                burst_measures[measure][i,j] = fct(br)
                event_measures[measure][i,j] = fct(er)
                be_r_measures[measure][i,j] = fct(br)/fct(er)

    statistics = {m : (event_measures[m], burst_measures[m], be_r_measures[m]) for m in ['mean', 'var', 'cor']}
    
    for stat_name, stat in statistics.items():
        fig, ax = plt.subplots(1, 3, figsize = (10,3), dpi = 800)

        extent = [W_ip_as.min(), W_ip_as.max(), W_ip_bs.min(), W_ip_bs.max()]

        ax[0].set_title(f'Mean event rate {stat_name}')
        fig.colorbar(ax[0].imshow(stat[0], cmap='hot', origin='lower', extent=extent), ax=ax[0])
        ax[0].set_xlabel(r'$W_b$')
        ax[0].set_ylabel(r'$W_a$')

        ax[1].set_title(f'Mean Burst rate {stat_name}')
        fig.colorbar(ax[1].imshow(stat[1], cmap='hot', origin='lower', extent=extent), ax=ax[1])
        ax[1].set_xlabel(r'$W_b$')
        ax[1].set_ylabel(r'$W_a$')

        ax[2].set_title(f'Mean burst/event ratio {stat_name}')
        fig.colorbar(ax[2].imshow(stat[2], cmap='hot', origin='lower', extent=extent), ax=ax[2])
        ax[2].set_xlabel(r'$W_b$')
        ax[2].set_ylabel(r'$W_a$')

        plt.tight_layout()
        plt.savefig(f'plots/event_burst_rates_{stat_name}_ma_{m_a}_mb_{m_b}_{"post" if post_inter else "pre"}.png')


def main():

    n_pyr = 50
    n_int_a = 5
    n_int_b = 5

    W_ip_as = np.linspace(0, 200, 50) 
    W_ip_bs = np.linspace(0, 200, 50)

    # for (ma, mb) in [#(10, 4), (6.5, 8), 
    #     (4, 10)]:
    #     test_weights_pi(W_ip_as, W_ip_bs, ma, mb, n_pyr, n_int_a, n_int_b)
        # test_weights_pi(W_ip_as, W_ip_bs, ma, mb, n_pyr, n_int_a, n_int_b, post_inter=True)

    m_b = 8
    m_a = 9
    W_ip_a = 2000*np.ones((n_int_a, n_pyr))/np.sqrt(n_pyr)
    W_ip_b = 1000*np.ones((n_int_b, n_pyr))/np.sqrt(n_pyr)
    W_pi_a = 1000*np.ones((n_pyr, n_int_a))/np.sqrt(n_pyr) ## Normalize by normalizing the rows of W (normalize post-synaptic weights)
    W_pi_b = 1000*np.ones((n_pyr, n_int_b))/np.sqrt(n_pyr) 

    weights = {'pi_a' : W_pi_a, 'pi_b' : W_pi_b, 'ip_a' : W_ip_a, 'ip_b' : W_ip_b, 
               'pp_a' : np.zeros((n_int_a, n_int_a)), 'pp_b' : np.zeros((n_int_b, n_int_b))}

    print(W_pi_a, W_pi_b, W_ip_a, W_ip_b)

    run_network_simulation(
        n_pyr, n_int_a, n_int_b, m_b, m_a, weights = weights, tn = 200, plot = True)

    values = np.tile(np.arange(100).reshape(100, 1), (1, 100)).T

    i,j = 1, 20
    print(values, values[i,j])

    fig, ax = plt.subplots(1, 1, figsize = (8,3), dpi = 800)

    fig.colorbar(ax.imshow(values, cmap='hot', origin='lower'), ax=ax)

    plt.savefig('plots/test.png')



if __name__ == "__main__":
    main()