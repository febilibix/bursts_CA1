import numpy as np 
from neuron import PyramidalCells
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine


def create_inputs_constant(N = 50, dt = 0.01, tn = 100, m = 8, sigma = 0.01):

    ts, ws, Is = [], [], []
    I = np.random.choice([0, m], N, p = [.5, .5])
    w = I

    for t in np.arange(0.0001,tn,dt):
        ts.append(t)
        ws.append(w)   
        Is.append(I)
        delta_w = np.random.normal(0, np.sqrt(sigma), N)
        w = I + delta_w

    return np.array(ts), np.array(ws), I


def get_input_EC(N = 50, dt = 0.01, tn_pattern = 100, m = 8, sigma = 0.01, n_patterns = 2, n_presentations = 10):

    params_per_pattern = [create_inputs_constant(N, dt, tn_pattern, m, sigma) for i in range(n_patterns)]

    I_a_all, ts = [], []
    ps = [i[2] for i in params_per_pattern]
    T = 0

    for i in range(n_presentations):

        for j in range(n_patterns):
            ## learning
            
            t_a, I_a, _ = params_per_pattern[j] 
            I_a_all.append(I_a)
            ts.extend(T + t_a)
            T = ts[-1]

        for j in range(n_patterns):
            ## recall    

            I_a = np.zeros((int(tn_pattern/dt), N))
            t_a = np.arange(0, tn_pattern, dt)

            I_a_all.append(I_a)
            ts.extend(T + t_a)
            T = ts[-1]

    I_a_all = np.concatenate(I_a_all, axis = 0)

    return np.array(ts), np.array(I_a_all), ps


def get_output_CA3(N = 30, dt = 0.01, tn_pattern = 100, m = 8, sigma = 0.01, n_patterns = 2, n_presentations = 10):
    
    params_per_pattern = [create_inputs_constant(N, dt, tn_pattern, m, sigma) for i in range(n_patterns)]

    I_b_all = []

    for i in range(n_presentations):
        for j in range(n_patterns):
            ## learning
            _, I_CA3, _ = params_per_pattern[j] 
            I_b_all.append(I_CA3)
            
        for j in range(n_patterns):
            ## recall    
            _, I_CA3, _ = params_per_pattern[j]
            I_b_all.append(I_CA3)

    I_b_all = np.concatenate(I_b_all, axis = 0)

    return I_b_all


def plot_raster(events, bursts, delta_t, mean_event_rate, mean_burst_rate, tn, t, I_CA3, top_down, cos_dist = None):

    fig, axs = plt.subplots(5,1, figsize = (12,8), dpi = 800, sharex=True)

    fig.suptitle(f"")

    # axs[0].set_title("CA3 output")
    # axs[0].plot(t, I_CA3, lw = 0.5)
    # axs[0].set_ylabel("Input")

    axs[0].set_title("Top Down learning signal")
    axs[0].plot(t, top_down, lw = 0.5)
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
    axs[3].legend(loc = 'upper left')

    axs[4].set_title("Cosine distance")
    axs[4].plot(np.linspace(0, tn, cos_dist.shape[0]), cos_dist, label=[f'pattern{i}' for i in range(cos_dist.shape[1])])
    axs[4].legend(loc = 'upper left')

    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/raster_plot.png')


def get_active_neurons(event_count, n_patterns, dt, tn):
    
    active_neurons = np.zeros((n_patterns, event_count.shape[1]))
    time_step = tn/n_patterns/dt
    
    for i in range(n_patterns):
        activity = np.sum(event_count[int((i+1)*time_step - time_step/2): int((i+1)*time_step), :], axis = 0) / time_step
        active_neurons[i, :] = activity 

    return active_neurons


def run_simulation(n_CA3 = 30, n_pyr = 50, n_int_a = 5, n_int_b = 5, tn = 1000,
                   delta_t = 1, n_patterns = 2, n_presentations = 10, lr = 0.05, event_plot = True):


    tn_pattern = tn//n_patterns//n_presentations//2
    n_cells = {'pyramidal' : n_pyr, 'inter_a' : n_int_a, 'inter_b' : n_int_b, 'CA3' : n_CA3}

    # TODO: I want to hardcode all these parameters into the neuron class:
    params_basal  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
    params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5 }
    params_inter  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

    m_a, m_b = 6.5, 8*n_pyr
    sigma_a, sigma_b = 2*m_a/1000, 2*m_b/1000

    # Setting the simulation time parameters 
    t0 = 0
    tn = tn
    dt = 0.01

    t_a, I_a_arr, I_a_no_noise = get_input_EC(N = n_pyr, dt = dt, tn_pattern = tn_pattern,
                                              m = m_a, sigma=sigma_a, n_patterns=n_patterns,
                                              n_presentations=n_presentations)
    I_a = lambda t: I_a_arr[int(t/dt)-1, :]

    neuron_type = np.where(np.array(I_a_no_noise) > 0, 1, 0)
    neuron_type = np.repeat(neuron_type, int(tn/(delta_t*n_patterns*n_presentations*2)), axis=0).astype('bool')
    neuron_type = np.tile(neuron_type, (n_presentations*2, 1)).astype('bool')    

    # TODO: I want to hardcode all these parameters into the neuron class:

    W_ip_a = 2000*np.ones((n_int_a, n_pyr))/np.sqrt(n_pyr)
    W_ip_b = 1000*np.ones((n_int_b, n_pyr))/np.sqrt(n_pyr)
    W_pi_a = 1000*np.ones((n_pyr, n_int_a))/np.sqrt(n_pyr)
    W_pi_b = 1000*np.ones((n_pyr, n_int_b))/np.sqrt(n_pyr)

    weights = {'pi_a' : W_pi_a, 'pi_b' : W_pi_b, 'ip_a' : W_ip_a, 'ip_b' : W_ip_b, 
               'pp_a' : np.zeros((n_int_a, n_int_a)), 'pp_b' : np.zeros((n_int_b, n_int_b)),
               'CA3' : None}
    
    v0 = {
        "basal":  np.random.normal(params_basal['E_L'], sigma_a, n_cells['pyramidal']),
        "apical": np.random.normal(params_apical['E_L'], sigma_a, n_cells['pyramidal']),
        "inter_a":  params_inter['E_L'],
        "inter_b":  params_inter['E_L']
        }
    
    
    
    I_CA3_arr = get_output_CA3(N = n_CA3, dt = dt, tn_pattern = tn_pattern, m = m_b, sigma=sigma_b, n_patterns=n_patterns, 
                               n_presentations=n_presentations)
    I_CA3 = lambda t: I_CA3_arr[int(t/dt)-1, :]

    

    pyramidal = PyramidalCells(  
                    params_basal=params_basal,
                    input_basal=I_CA3,
                    params_apical=params_apical,
                    input_apical=I_a,
                    params_inter=params_inter,   
                    n_cells=n_cells,
                    weights=weights,
                    plasticity=True,
                    learning_rate=lr
                    )

    _, _, _, event_count, burst_count, events, bursts = pyramidal.run_simulation(
                v0, t0, tn, dt, n_patterns= n_patterns, n_presentations = n_presentations, selected_neurons = I_a_no_noise, 
                event_plot=event_plot)
    
    cos_dist = pyramidal.cosine_distances

    if event_plot:
        plot_results(events, bursts, delta_t, event_count, burst_count, tn, t_a, I_CA3_arr, I_a_arr, neuron_type, n_pyr, dt, cos_dist)

    # print(cos_dist)

    return cos_dist
    

def plot_results(events, bursts, delta_t, event_count, burst_count, tn, t_a, I_CA3_arr, I_a_arr, neuron_type, n_pyr, dt, cos_dist):
    
    event_rate = np.zeros((int(tn/delta_t), n_pyr))
    burst_rate = np.zeros((int(tn/delta_t), n_pyr))
    totals = {'event': event_rate, 'burst': burst_rate}

    for i, t in enumerate(range(int(tn/delta_t))):
        for name, count in {'event' : event_count, 'burst' : burst_count}.items():
            total = np.sum(count[int(t*delta_t/dt):int((t+1)*delta_t/dt), :], axis = 0)
            totals[name][i, :] = total

    mer0, mbr0 = np.mean(np.where(neuron_type, 0, event_rate), axis=1), np.mean(np.where(neuron_type, burst_rate, 0), axis=1)
    mer1, mbr1 = np.mean(np.where(neuron_type, event_rate, 0), axis=1), np.mean(np.where(neuron_type, burst_rate, 0), axis=1)

    mer, mbr = (mer0, mer1), (mbr0, mbr1)

    plot_raster(events, bursts, delta_t, mer, mbr, tn, t_a, I_CA3_arr, I_a_arr, cos_dist = cos_dist)



def plot_weights(W_CA3, I_a_no_noise):

    # TODO: I guess i will delete this function
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

    delta_t = 10
    eta = .001

    N_patterns = [200]
    print(N_patterns)
    N_iter = 10
    n_presentations = 10

    mean_cos = np.zeros((N_iter, len(N_patterns)))


    for i in range(N_iter):
        print(i)
        for j, n_patterns in enumerate(N_patterns):
            tn = n_presentations * n_patterns * 100
            cos_dist = run_simulation(
                n_CA3=n_CA3, n_pyr=n_pyr, n_int_a=n_int_a, n_int_b=n_int_b, tn=tn, delta_t=delta_t,
                n_patterns=n_patterns, lr = eta, n_presentations= n_presentations, event_plot=False)
            
            print(cos_dist[-1, :])
            print(np.mean(cos_dist[-1, :]))

            mean_cos[i, j] = np.mean(cos_dist[-1, :])

    mean_cos_mean = mean_cos.mean(axis=0)
    mean_cos_std = mean_cos.std(axis=0)

    plt.figure()
    plt.plot(N_patterns, mean_cos_mean, label='Mean Cosine Distance')
    plt.fill_between(N_patterns, mean_cos_mean - mean_cos_std, mean_cos_mean + mean_cos_std, alpha=0.3, label='Std Dev')
    plt.title("Cosine distances as function of number of patterns")
    plt.xlabel("Number of patterns")
    plt.ylabel("Cosine distance")
    plt.legend()
    plt.savefig('plots/cosine_distance_low.png')

    # test_cosine(n_CA3, n_pyr, n_int_a, n_int_b, tn, delta_t, n_patterns, n_presentations, eta)

    # repeat over different n_patterns and see how recall performance scales with number of patterns
    

def test_cosine(n_CA3, n_pyr, n_int_a, n_int_b, tn, delta_t, n_patterns, n_presentations, eta):

    cos = []

    for _ in range(10):
        cos_dist = run_simulation(
            n_CA3=n_CA3, n_pyr=n_pyr, n_int_a=n_int_a, n_int_b=n_int_b, tn=tn, delta_t=delta_t,
            n_patterns=n_patterns, lr = eta, n_presentations= n_presentations)
        
        cos.append(cos_dist[-1, :])

    print(np.array(cos).mean(axis = 0), np.array(cos).std(axis = 0))
    quit()

    cos = []
    for _ in range(1000):
        # print(1-cosine(np.random.rand(50), np.random.randn(50)))
        cos.append(1-cosine(np.random.rand(50), np.random.randn(50)))

    print(np.mean(cos), np.std(cos))

if __name__ == "__main__":
    main()