import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from neuron import PyramidalCells
from scipy.stats import pearsonr
import csv


def simulate_run(len_track = 200, n_runs = 20, av_running_speed = 20, dt = 0.01, tn = 1000):
    bins = np.arange(0., len_track)

    fps = 1/dt

    # n_runs = 1000

    # running_speed_a = np.random.chisquare(av_running_speed, size=n_runs) # running speed in the two directions
    # running_speed_b = np.random.chisquare(av_running_speed, size=n_runs) 
    running_speed_a = running_speed_b = np.ones(n_runs) * av_running_speed

    # stopping_time_a = np.random.chisquare(3, size=n_runs) # the time the mouse will spend at the two ends of the track
    # stopping_time_b = np.random.chisquare(3, size=n_runs)
    stopping_time_a = stopping_time_b = np.ones(n_runs) * 0

    x = np.array([])
    i = 0
    while True:
        stop1 = np.ones((int(stopping_time_a[i]*fps),)) * 0.
        run_length = len(bins) * fps / running_speed_a[i]
        run1 = np.linspace(0., float(len(bins)-1), int(run_length))
        stop2 = np.ones((int(stopping_time_b[i]*fps),)) * (len(bins)-1.)
        run_length = len(bins) * fps / running_speed_b[i]
        run2 = np.linspace(len(bins)-1., 0., int(run_length))
        x = np.concatenate((x, stop1, run1, stop2, run2))
        if len(x) >= tn*fps:
            break
        i += 1

    x = x[:int(tn*fps)]
    t = np.arange(len(x))/fps

    return t, x
    
import matplotlib.pyplot as plt


def custom_plot(ax=None, x_data=None, y_data=None, title="My Plot", xlabel="X-Axis", ylabel="Y-Axis", label = None):
    # If no Axes object is provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    
    # Example plotting: Line plot of x_data vs. y_data
    ax.plot(x_data, y_data, label = label)
    
    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label is not None:
        ax.legend()
    ax.grid(True)
    
    # Return the Axes object for further customization
    return ax




def plot_track_CA3(t, x, activity):
    fig, axs = plt.subplots(2,1, figsize = (10,8), dpi = 200, sharex=True)

    extent = [ t.min(), t.max(), 0, activity.shape[0],]   

    fig.suptitle(f"")

    axs[0].set_title("Mouse trajectory")
    axs[0].plot(t, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (cm)')

    axs[1].set_title("CA3 activity")
    # fig.colorbar(axs[1].imshow(activity, aspect='auto', extent=extent), ax=axs[1])
    axs[1].imshow(activity, aspect='auto', extent=extent)
    axs[1].set_ylabel("Neuron")
    axs[1].set_xlabel("Time (s)")

    
    plt.savefig('plots/simulated_activity.png')
    plt.close()


def correlate_firing_rates(fr_old, fr_new):
    cor_row = np.zeros((fr_old.shape[0]))
    for i in range(fr_old.shape[0]):
        cor_row[i] = pearsonr(fr_old[i, :], fr_new[i, :])[0]
        
    cor_col = np.zeros((fr_old.shape[1]))
    for i in range(fr_old.shape[1]):
        cor_col[i] = pearsonr(fr_old[:, i], fr_new[:, i])[0]
 
    return np.nanmean(cor_row), np.nanmean(cor_col)


def plot_firing_rates(fig, ax, firing_rates, m_EC, out):

    sort_TD = np.argsort(m_EC)
    sorted_fr = firing_rates[np.ix_(sort_TD, np.arange(firing_rates.shape[1]))]

    ###### FOR NOW SINCE RETRIEVAL IS ONLY ONE WAY FORWARD AND ONE WAY BACK, I CAN JUST SPLIT IN MIDDLE AND MEAN. 
    ###### IDEALLY I WOULD SOMEHOW PUT IT IN BINS ACCORDING TO POSITION AND THEN MEAN OVER THAT

    half = sorted_fr.shape[1] // 2
    mean_firing_rates = (sorted_fr[:, :half] + sorted_fr[:, :-half-1:-1]) / 2

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Firing rates of {'CA1' if not out.startswith('CA3') else 'CA3'} neurons")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")

    return fig, ax



def get_firing_rates(pyramidal, event_count, delta_t = 10):
    # event_count = pyramidal.spike_count
   
    step_size = int(delta_t // pyramidal.dt)
    firing_rates = np.zeros((event_count.shape[1], len(event_count) // step_size))
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / delta_t

    return firing_rates


def plot_weights(fig, ax, W, m_CA3, m_EC):
    sort_CA3 = np.argsort(m_CA3)
    sort_EC = np.argsort(m_EC)

    # fig, ax = plt.subplots(figsize = (10,8), dpi = 200)
    sorted_W = W[np.ix_(sort_EC, sort_CA3)]
    fig.colorbar(ax.imshow(sorted_W/np.sum(sorted_W)*50, origin='lower', aspect = 'auto'), ax=ax) # 50 n_pyr
    ax.set_title("CA3 weights")
    ax.set_xlabel("CA3 neuron")
    ax.set_ylabel("CA1 neuron")
    return fig, ax


def plot_correlations(A, cors_col, cors_row):

    cors = {"columns": cors_col, "rows": cors_row}
    fig, axs = plt.subplots(1, 2, figsize = (12,6), dpi = 400, sharey=True)
    axs[0].set_ylabel('Average correlation')

    for i, (cor_name, cor) in enumerate(cors.items()):

        mean_cor = np.nanmean(cor, axis = 2)
        std_cor = np.nanstd(cor, axis = 2)
        print(cor_name)
        print('mean', mean_cor, 'std', std_cor)

        lower, upper = mean_cor - std_cor, mean_cor + std_cor

        axs[i].plot(A, mean_cor[:, 0], label = 'CA1')
        axs[i].plot(A, mean_cor[:, 1], label = 'CA3')
        axs[i].fill_between(A, lower[:, 0], upper[:, 0], alpha=0.3)
        axs[i].fill_between(A, lower[:, 1], upper[:, 1], alpha=0.3)
        axs[i].set_title(f'Average correlation over {cor_name}')
        axs[i].set_xlabel('Similarity between environments')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('plots/correlation_unf_fam.png')
    plt.close()


def test_correlations_fam_novel(len_track, tn, av_running_speed, dt, n_cells, lr, t_epoch):
    max_runs = int(tn/20)

    tn_retrieval = 1000
    max_runs_rt = int(tn_retrieval/20)

    A = np.round(np.arange(0,1.1,0.1),1)
    N_simulations = 10
    cors_col_all, cors_row_all = np.zeros((len(A), 2, N_simulations)), np.zeros((len(A), 2, N_simulations))

    t_run, x_run = simulate_run(len_track, max_runs, av_running_speed, dt, tn)
    t_run2, x_run2 = simulate_run(len_track, max_runs_rt, av_running_speed, dt, tn_retrieval)

    with open('correlations.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(["simulation", "a", "cor_row_CA1", "cor_col_CA1", "cor_row_CA3", "cor_col_CA3"])
    
    for sim in range(N_simulations):
        pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr)
        pyramidal.learn_place_cells(t_run, x_run, t_epoch)

        # plot_weights(pyramidal.W_CA3, pyramidal.m_CA3, pyramidal.m_EC) 
        # plot_track_CA3(t_run, x_run, pyramidal.CA3_act)

        plot_weights(pyramidal.W_CA3, pyramidal.m_CA3, pyramidal.m_EC)

        ## Retrieval familiar environment
        event_count = pyramidal.retrieve_place_cells(t_run2, x_run2, new_env=False)
        fr_old = get_firing_rates(pyramidal, event_count, delta_t = 10)

        m_CA3_old = pyramidal.m_CA3
        CA3_act_old = pyramidal.CA3_act.copy()
        plot_firing_rates(pyramidal.CA3_act, m_CA3_old, out = 'CA3_old_env')
        plot_firing_rates(fr_old, pyramidal.m_EC, out = 'old_env')

        cors_col = np.zeros((len(A), 2))
        cors_row = np.zeros((len(A), 2))
        for i, a in enumerate(A):
            print("a =", a)
            
            ## Retrieval new environment
            event_count = pyramidal.retrieve_place_cells(t_run2, x_run2, new_env=True, a = a)
            fr_new = get_firing_rates(pyramidal, event_count, delta_t = 10)

            plot_firing_rates(pyramidal.CA3_act, m_CA3_old, out = 'CA3_new_env')
            plot_firing_rates(fr_new, pyramidal.m_EC, out = 'new_env')

            print('Correlation CA1')
            r_row, r_col = correlate_firing_rates(fr_old, fr_new)
            print(r_row, r_col)

            print('Correlation CA3')
            r_row_CA3, r_col_CA3 = correlate_firing_rates(CA3_act_old, pyramidal.CA3_act)
            print(r_row_CA3, r_col_CA3)

            cors_col[i, :] = r_col, r_col_CA3
            cors_row[i, :] = r_row, r_row_CA3

            with open('correlations.csv', mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([sim, a, r_row, r_col, r_row_CA3, r_col_CA3])

        cors_col_all[:, :, sim] = cors_col
        cors_row_all[:, :, sim] = cors_row
    
    plot_correlations(A, cors_col_all, cors_row_all)


def main():
    np.random.seed(0)
    t_epoch = 50

    lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    v_ths = [-35, -40, -45, -50, -55]
    taus = [1, 5, 10, 15, 20, 25]
    run_for_speed(av_running_speed = 0.2, lrs = lrs, v_ths = v_ths, taus = taus, t_epoch = t_epoch, out_folder = 'param_tuning')

    # t_epoch = 2
    # speed = 5
    # run_for_speed(av_running_speed = speed, lrs = [0.001], v_ths = [-35], taus = [10], t_epoch = t_epoch, out_folder = 'adapt_speed')

    
def run_for_speed(av_running_speed = 0.2, lrs = [0.001], v_ths = [-35], taus = [10], t_epoch = 50, out_folder = 'param_tuning'):
    # Setting the simulation time parameters 
    n_cells = {'pyramidal' : 50, 'inter_a' : 5, 'inter_b' : 5, 'CA3' : 30}
    len_track = 100. 
    tn = 2000

    dt = 0.01
    max_runs = int(2*tn/(len_track/av_running_speed))

    t_run, x_run = simulate_run(len_track, max_runs, av_running_speed, dt, tn)
    tn_retrieval = len_track/av_running_speed*2
    max_runs_rt = int(2*tn_retrieval/(len_track/av_running_speed))
    t_run2, x_run2 = simulate_run(len_track, max_runs_rt, av_running_speed, dt, tn_retrieval)


    t_run, x_run = (t_run, t_run2), (x_run, x_run2)

    for lr in lrs:
        for v_th in v_ths:
            for tau in taus:
                print(f"lr = {lr}, v_th = {v_th}, tau = {tau}")
                param_tuning(lr, v_th, tau, t_run, x_run, t_epoch, tn, n_cells, out_folder)


def param_tuning(lr, v_th, tau, t_run, x_run, t_epoch, tn, n_cells, out_folder = 'param_tuning'):

    pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr)
    pyramidal.pa = {"E_L": -65, "R": 10, "v_th": v_th, "tau": tau} 
    pyramidal.learn_place_cells(t_run[0], x_run[0], t_epoch)

    
    fig = plt.figure(figsize=(10, 8))

    # Define a 2x2 grid with custom subplot spans
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.25, 0.25, 0.5])  # Adjust heights as needed

    # Top row: two full-width subplots
    ax1 = fig.add_subplot(gs[1, :])  # Top-left (spans both columns)
    ax2 = fig.add_subplot(gs[0, :])  # Top-right (also spans both columns)
    ax3 = fig.add_subplot(gs[2, 0])  # Bottom-left
    ax4 = fig.add_subplot(gs[2, 1])

    axs = [ax1, ax2, ax3, ax4]
    
    axs[0] = custom_plot(ax=axs[0], x_data=t_run[0], y_data=x_run[0], title="", xlabel="Time (s)", ylabel="Mouse trajectory")
    axs[1] = custom_plot(ax=axs[1], x_data=np.arange(0, tn, t_epoch), y_data=pyramidal.burst_rate, title="", xlabel="Time (s)", ylabel="Burst rate")
    fig, axs[2] = plot_weights(fig, axs[2], pyramidal.W_CA3, pyramidal.m_CA3, pyramidal.m_EC)
    event_count = pyramidal.retrieve_place_cells(t_run[1], x_run[1], new_env=False)
    fr = get_firing_rates(pyramidal, event_count, delta_t = 1)
    fig, axs[3] = plot_firing_rates(fig, axs[3], fr, pyramidal.m_EC, 'CA1_test')
    plt.tight_layout()
    plt.savefig(f'plots/{out_folder}/lr_{lr}_v_th_{v_th}_tau_{tau}.png')
    
    plt.close()
    # plot_track_CA3(t_run, x_run, pyramidal.CA3_act)




if __name__ ==  "__main__":
    main()

