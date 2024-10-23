import numpy as np 
import matplotlib.pyplot as plt
from neuron import PyramidalCells
from scipy.stats import pearsonr


def simulate_run(len_track = 200, n_runs = 20, av_running_speed = 20, dt = 0.01, tn = 1000):
    bins = np.arange(0., len_track)

    fps = 1/dt

    n_runs = 1000

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
 
    return np.mean(cor_row), np.mean(cor_col)


def plot_firing_rates(firing_rates, m_EC, out):

    sort_TD = np.argsort(m_EC)
    sorted_fr = firing_rates[np.ix_(sort_TD, np.arange(firing_rates.shape[1]))]

    ###### FOR NOW SINCE RETRIEVAL IS ONLY ONE WAY FORWARD AND ONE WAY BACK, I CAN JUST SPLIT IN MIDDLE AND MEAN. 
    ###### IDEALLY I WOULD SOMEHOW PUT IT IN BINS ACCORDING TO POSITION AND THEN MEAN OVER THAT

    half = sorted_fr.shape[1] // 2
    mean_firing_rates = (sorted_fr[:, :half] + sorted_fr[:, :-half-1:-1]) / 2

    fig, ax = plt.subplots(figsize = (10,8), dpi = 200)
    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title("Firing rates of CA1 neurons")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")
    plt.savefig(f'plots/firing_rates_{out}.png')
    plt.close()



def get_firing_rates(pyramidal, event_count, delta_t = 10):
    # event_count = pyramidal.spike_count
   
    step_size = int(delta_t // pyramidal.dt)
    firing_rates = np.zeros((event_count.shape[1], len(event_count) // step_size))
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / delta_t

    return firing_rates


def plot_weights(W, m_CA3, m_EC):
    sort_CA3 = np.argsort(m_CA3)
    sort_EC = np.argsort(m_EC)

    fig, ax = plt.subplots(figsize = (10,8), dpi = 200)
    sorted_W = W[np.ix_(sort_EC, sort_CA3)]
    fig.colorbar(ax.imshow(sorted_W/np.sum(sorted_W)*50, origin='lower', aspect = 'auto'), ax=ax) # 50 n_pyr
    ax.set_title("CA3 weights")
    ax.set_xlabel("CA3 neuron")
    ax.set_ylabel("CA1 neuron")
    plt.savefig('plots/CA3_weights.png')
    plt.close()


def main():

    # Setting the simulation time parameters 
    n_cells = {'pyramidal' : 50, 'inter_a' : 5, 'inter_b' : 5, 'CA3' : 30}
    len_track = 100. 
    tn = 1000
    av_running_speed = .2 # 0.2
    dt = 0.01
    max_runs = int(tn/20)
    t_epoch = 50
    lr = .001 # 0.001
    

    pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr)
    
    t_run, x_run = simulate_run(len_track, max_runs, av_running_speed, dt, tn)
    
    pyramidal.learn_place_cells(t_run, x_run, t_epoch, dt)
    plot_weights(pyramidal.W_CA3, pyramidal.m_CA3, pyramidal.m_EC)
    
    plot_track_CA3(t_run, x_run, pyramidal.CA3_act)

    tn_retrieval = 1000
    max_runs_rt = int(tn_retrieval/20)

    t_run2, x_run2 = simulate_run(len_track, max_runs_rt, av_running_speed, dt, tn_retrieval)
    
    event_count = pyramidal.retrieve_place_cells(t_run2, x_run2, dt, new_env=False)
    fr_old = get_firing_rates(pyramidal, event_count, delta_t = 10)

    print(pyramidal.CA3_act)
    print(pyramidal.CA3_act.shape)
    print(pyramidal.m_CA3)
    plot_firing_rates(pyramidal.CA3_act, pyramidal.m_CA3, out = 'CA3_test')
    quit()

    plot_firing_rates(fr_old, pyramidal.m_EC, out = 'old_env')

    event_count = pyramidal.retrieve_place_cells(t_run2, x_run2, dt, new_env=True)
    fr_new = get_firing_rates(pyramidal, event_count, delta_t = 10)

    plot_firing_rates(fr_new, pyramidal.m_EC, out = 'new_env')

    r_row, r_col = correlate_firing_rates(fr_old, fr_new)
    print(r_row, r_col)


if __name__ ==  "__main__":
    main()

