from neuron import PyramidalCells
import numpy as np 
import matplotlib.pyplot as plt


def simulate_converged_weight_matrix(n_rows, n_cols):
    std_dev = 20  # Standard deviation for the Gaussian distributions

    # Create an empty matrix
    matrix = np.zeros((n_rows, n_cols))
    means = np.linspace(0, n_cols, n_rows, endpoint=False)

    # Generate Gaussian distributions with different means for each row
    for i in range(n_rows):
        matrix[i, :] = 0.05 * np.exp(-0.5 * ((np.arange(n_cols) - means[i]) / std_dev) ** 2)

    
    # Normalize the matrix
    matrix = matrix / (np.sum(matrix, axis=1, keepdims=True)) 
    matrix = np.where(matrix < 0, 0, matrix)

    # Print the matrix
    return matrix


def get_firing_rates(pyramidal, event_count):

    firing_rates = np.zeros((event_count.shape[1], 128))
    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)

    return firing_rates


def plot_activity(tn, all_n_active, ip_weights, pi_weights, sim_weight):        
    t_plot = np.linspace(0, tn, len(all_n_active))
    t_weights = np.linspace(0, tn, len(ip_weights))

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    title = 'Active Cells after Convergence' if sim_weight else 'Active Cells before Convergence'
    fig.suptitle(title)
    axs[0].plot(t_plot, all_n_active, label = 'number of active cells')
    axs[0].set_ylabel('Number of active cells')

    color = 'tab:blue'
    axs[1].plot(t_weights, ip_weights, color = color)
    axs[1].set_ylabel('pyr->inh weight', color=color)
    axs[1].tick_params(axis='y', labelcolor=color)

    ax2 = axs[1].twinx()
    color = 'tab:red'
    ax2.plot(t_weights, pi_weights, color = color)
    ax2.set_ylabel('inh->pyr weight', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xlabel('Time (s)')  
    plt.tight_layout()

    if sim_weight:
        plt.savefig('plots/inhibition/pyramidal_n_active_sim.png')
    else:
        plt.savefig('plots/inhibition/pyramidal_n_active.png')     

    plt.close() 


def test_inhibition():

    lr = 5e-1
    dt = 0.001
    n_cells = {'pyramidal' : 200, 'inter_a' : 20, 'inter_b' : 20, 'CA3' : 120}
    tn = 400 
    len_track = 100
    t_epoch = 10

    w_CA3_sim = simulate_converged_weight_matrix(n_cells['pyramidal'], n_cells['CA3'])

    for sim_weight in [True, False]:

        pyramidal = PyramidalCells(n_cells, learning_rate=lr, dt=dt)
        if sim_weight: 
            pyramidal.W_CA3 = w_CA3_sim

        plt.figure()
        plt.imshow(pyramidal.W_CA3, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar()
        plt.xlabel('CA3 cells')
        plt.ylabel('Pyramidal cells')
        plt.title('Schaffer collateral weight matrix')
        plt.savefig(f'plots/inhibition/{"sim" if sim_weight else "random"}_w_ca3.png')

        pyramidal.inh_plasticity = False
        W_ip_orig = pyramidal.W_ip_b 
        W_pi_orig = pyramidal.W_pi_b
        pyramidal.W_ip_b = 0*pyramidal.W_ip_b 
        pyramidal.W_pi_b = .5*pyramidal.W_pi_b
        
        n_inh_weights = 20

        all_n_active = []
        ip_weights = []
        pi_weights = []

        for i in range(2*n_inh_weights):

            t_run = np.arange(0, tn//(2*n_inh_weights), dt)
            x_run = 50*np.ones_like(t_run)

            spike_count, burst_count = pyramidal.learn_place_cells(t_run, x_run, t_epoch, top_down=False, len_track=len_track, plasiticty=False)
            sr, br = [get_firing_rates(pyramidal, x) for x in (spike_count, burst_count)]

            n_active = np.sum(sr > 0, axis = 0)
            all_n_active.extend(list(n_active))

            ip_weights.extend(list(pyramidal.W_ip_b[0,0]*np.ones_like(t_run)))
            pi_weights.extend(list(pyramidal.W_pi_b[0,0]*np.ones_like(t_run)))

            if i < n_inh_weights:
                pyramidal.W_ip_b = pyramidal.W_ip_b + .5*W_ip_orig
            elif i == n_inh_weights:
                pyramidal.W_ip_b = .5*W_ip_orig
                pyramidal.W_pi_b = 0*pyramidal.W_pi_b
            if i >= n_inh_weights:
                pyramidal.W_pi_b = pyramidal.W_pi_b + .5*W_pi_orig

        plot_activity(tn, all_n_active, ip_weights, pi_weights, sim_weight)


if __name__ == '__main__':
    test_inhibition()