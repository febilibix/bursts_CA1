import numpy as np 
from neuron import PyramidalCells
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv


def get_events_bursts(n_pyr, spike_count, burst_count, dt):
        
    events = [[] for _ in range(n_pyr)] 
    bursts = [[] for _ in range(n_pyr)]

    for i in range(spike_count.shape[0]):

        t_new = i * dt
        [events[l[0]].append(t_new) for l in np.argwhere(spike_count[i, :])]
        [bursts[l[0]].append(t_new) for l in np.argwhere(burst_count[i, :])]

    return events, bursts


def create_inputs_constant(n_neurons, n_patterns):
    return np.random.choice([0, 1], (n_neurons, n_patterns), p = [.5, .5])
   

def plot_raster(events, bursts, delta_t, mean_event_rate, mean_burst_rate, tn, t):

    fig, axs = plt.subplots(4,1, figsize = (12,8), dpi = 800, sharex=True)

    fig.suptitle(f"")

    # axs[0].set_title("CA3 output")
    # axs[0].plot(t, I_CA3, lw = 0.5)
    # axs[0].set_ylabel("Input")

    # axs[0].set_title("Top Down learning signal")
    # axs[0].plot(t, top_down, lw = 0.5)
    # axs[0].set_ylabel("Input")

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

    # axs[4].set_title("Cosine distance")
    # axs[4].plot(np.linspace(0, tn, cos_dist.shape[0]), cos_dist, label=[f'pattern{i}' for i in range(cos_dist.shape[1])])
    # axs[4].legend(loc = 'upper left')

    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/raster_plot_test.png')



def run_simulation(n_pyr, n_int_a, n_int_b, n_CA3, n_presentations, lr, n_patterns, return_fr = False, weights = {'inh':1}):

    n_cells = {'pyramidal' : n_pyr, 'inter_a' : n_int_a, 'inter_b' : n_int_b, 'CA3' : n_CA3}

    # Setting the simulation time parameters 
    t_per_pattern = 100
    dt = 0.01
    
    pyramidal = PyramidalCells(n_cells, weights = weights, learning_rate = lr, dt = dt)
    
    patterns = create_inputs_constant(n_CA3, n_patterns)
    top_down = create_inputs_constant(n_pyr, n_patterns)
    
    pyramidal.learn_patterns(patterns, top_down, n_presentations=n_presentations, t_per_pattern=t_per_pattern)
    pyramidal.pattern_retrieval(patterns, top_down, t_per_pattern=t_per_pattern)

    print(pyramidal.cosine_distances)

    if return_fr:
        return pyramidal.firing_rate_means
    
    # plot_results(delta_t, event_count, burst_count, tn, pyramidal.t_values, neuron_type, n_pyr, dt)

    return np.mean(pyramidal.cosine_distances)


def plot_results(delta_t, event_count, burst_count, tn, t_a, neuron_type, n_pyr, dt):

    events, bursts = get_events_bursts(n_pyr, event_count, burst_count, dt)
    
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

    plot_raster(events, bursts, delta_t, mer, mbr, tn, t_a)



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


def run_for_different_number_of_patterns(n_pyr, n_int_a, n_int_b, n_CA3, eta, N_patterns, inh = 1):

    print(N_patterns)
    N_iter = [int(1/i) for i in N_patterns]
    N_iter = [1 if i == 0 else i for i in N_iter]
    n_presentations = 10

    mean_cos = np.zeros((2, len(N_patterns)))
    weights = {'inh': inh}

    with open(f'results/cos_distances_inh_{round(inh,1)}.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(["N_patterns", "N_iter", "Mean_cos_mean", "Mean_cos_std"])
    
    for j, n_patterns in enumerate(N_patterns):
        cos = np.zeros((N_iter[j]))

        for i in range(N_iter[j]):
            
            tn = n_presentations * n_patterns * 100
            cos_dist = run_simulation(n_pyr, n_int_a, n_int_b, n_CA3, n_presentations, lr=eta, n_patterns=n_patterns, weights=weights)

            cos[i] = cos_dist

        mean_cos[:, j] = np.mean(cos), np.std(cos)

        with open(f'results/cos_distances_inh_{inh}.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([n_patterns, N_iter[j], np.mean(cos), np.std(cos)])

    mean_cos_mean = mean_cos[0, :]
    mean_cos_std = mean_cos[1, :]

    plt.figure()
    plt.plot(N_patterns, mean_cos_mean, label='Mean Cosine Distance')
    plt.fill_between(N_patterns, mean_cos_mean - mean_cos_std, mean_cos_mean + mean_cos_std, alpha=0.3, label='Std Dev')
    plt.title("Cosine distances as function of number of patterns")
    plt.xlabel("Number of patterns")
    plt.ylabel("Cosine distance")
    plt.legend()
    plt.savefig('plots/cosine_distance_low.png')




def test_fr_inhibition_strength(inh_strengths, n_pyr, n_int_a, n_int_b, n_CA3, eta):
    inh_strengths = np.arange(0, 2.1, 0.1)

    fr_means_over_inh = []

    for inh in inh_strengths:
        weights = {'inh': inh}
        n_patterns = 1
        n_presentations = 10

        fr_means = run_simulation(n_pyr, n_int_a, n_int_b, n_CA3, n_presentations, lr=eta, n_patterns=n_patterns, return_fr=True, weights = weights)

        fr_means_over_inh.append(fr_means[-1])
        # plt.figure()
        # plt.plot(range(len(fr_means)), fr_means)
        # plt.xlabel("Time/Epochs")
        # plt.ylabel("Mean firing rate/ Mean Activity")
        # plt.savefig('plots/firing_rate_means.png')

    plt.figure()
    plt.plot(inh_strengths, fr_means_over_inh)
    plt.xlabel("Inhibition strength")
    plt.ylabel("Mean firing rate")
    plt.title("Mean firing rate as function of inhibition strength")
    plt.savefig('plots/firing_rate_inh.png')



def main():

    n_CA3 = 30
    n_pyr = 50
    n_int_a = 5
    n_int_b = 5

    delta_t = 10
    eta = .001

    N_patterns = [1,2,3,4,5,10,20,50, 100,200, 500, # 1000
                ]
    
    # run_for_different_number_of_patterns(n_pyr, n_int_a, n_int_b, n_CA3, eta, N_patterns)
    inh_strengths = np.arange(0.5, 2.1, 0.5)
    # test_fr_inhibition_strength(inh_strengths, n_pyr, n_int_a, n_int_b, n_CA3, eta)

    for inh in inh_strengths:
        run_for_different_number_of_patterns(n_pyr, n_int_a, n_int_b, n_CA3, eta, N_patterns, inh = inh)


    ## TODO: Look at average activity of CA1 neurons depending on network size; Percentae of active neurons, see how that scales with network size
    ## See (1) how inhibition strength affects learning, as in reproduce curves of cosine distance as function of number of patterns for different inhibition 
    # (2) See how average activity goes over time (epochs of learning and retrieval)
    # (3) once this is done, if there is stable state see how this scales with inhibition strength as well 

if __name__ == "__main__":
    main()