import numpy as np 
import matplotlib.pyplot as plt
from neuron import PyramidalCells
from test_plasticity import plot_results


def simulate_run(len_track = 200, n_runs = 20, av_running_speed = 20, dt = 0.01, tn = 1000):
    bins = np.arange(0., len_track)

    fps = 1/dt

    n_runs = 1000
    print(n_runs)

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
        print(tn)

    x = x[:int(tn*fps)]
    t = np.arange(len(x))/fps

    return t, x


def simulate_activity(t, x, len_track = 100, n_cells = 30, tn = 1000, dt = 0.01, m = 8):
    sigma_pf = len_track/8
    m_cells = np.arange(0, len_track, len_track/n_cells)
    np.random.shuffle(m_cells)
   
    activity = np.zeros((n_cells, int(tn/dt)))
    for i in range(int(tn/dt)):
        activity[:, i] = np.exp(-0.5 * ((m_cells - x[i])**2) / sigma_pf**2)

    active_cells = np.random.choice([0, 1], size=(n_cells,), p=[0, 1]) # TODO: CHANGE THIS TO A PROBABILITY
    activity = m * activity * active_cells[:, np.newaxis]

    return activity, m_cells * active_cells
    

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


def run_simulation(len_track, av_running_speed, tn, n_cells, lr, n_plast_steps = 100):

    # Setting the simulation time parameters 
    dt = 0.01
    max_runs = int(tn/20)
    t_epoch = 50
    tn = 4000

    pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr)
    
    t_run, x_run = simulate_run(len_track, max_runs, av_running_speed, dt, tn)
    
    pyramidal.learn_place_cells(t_run, x_run, t_epoch, dt)
    plot_weights(pyramidal.W_CA3, pyramidal.m_CA3, pyramidal.m_EC)
    
    plot_track_CA3(t_run, x_run, pyramidal.CA3_act)

    quit()
    pyramidal.pattern_retrieval(patterns, top_down, t_per_pattern=t_per_pattern, dt=dt)

    params_basal  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
    params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5 }
    params_inter  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

    n_pyr, n_CA3, n_int_a, n_int_b = n_cells['pyramidal'], n_cells['CA3'], n_cells['inter_a'], n_cells['inter_b']

    m_a, m_b = 2*6.5, 8*n_pyr*2

    # Setting the simulation time parameters 
    t0 = 0
    tn = tn
    dt = 0.01
    

    
    print(max_runs)

   

    CA3_act, m_CA3 = simulate_activity(t, x, len_track = len_track, n_cells = n_CA3, tn = tn, dt = dt, m = m_b)
    I_CA3 = lambda t: CA3_act[:, int(t/dt)]
    EC_inp, m_EC = simulate_activity(t, x, len_track = len_track, n_cells = n_pyr, tn = tn, dt = dt, m = m_a)
    I_EC = lambda t: EC_inp[:, int(t/dt)]

    plot_track_CA3(t, x, CA3_act)

    # TODO: I want to hardcode all these parameters into the neuron class:
    W_ip_a = 2000*np.ones((n_int_a, n_pyr))/np.sqrt(n_pyr)
    W_ip_b = 1000*np.ones((n_int_b, n_pyr))/np.sqrt(n_pyr)
    W_pi_a = 1000*np.ones((n_pyr, n_int_a))/np.sqrt(n_pyr)
    W_pi_b = 1000*np.ones((n_pyr, n_int_b))/np.sqrt(n_pyr)

    weights = {'pi_a' : W_pi_a, 'pi_b' : W_pi_b, 'ip_a' : W_ip_a, 'ip_b' : W_ip_b, 
               'pp_a' : np.zeros((n_int_a, n_int_a)), 'pp_b' : np.zeros((n_int_b, n_int_b)),
               'CA3' : None}
    
    v0 = {
        "basal":  np.random.normal(params_basal['E_L'], 2*m_a/1000, n_cells['pyramidal']),
        "apical": np.random.normal(params_apical['E_L'], 2*m_a/1000, n_cells['pyramidal']),
        "inter_a":  params_inter['E_L'],
        "inter_b":  params_inter['E_L']
        }
    
    pyramidal = PyramidalCells(  
                    params_basal=params_basal,
                    input_basal=I_CA3,
                    params_apical=params_apical,
                    input_apical=I_EC,
                    params_inter=params_inter,   
                    n_cells=n_cells,
                    weights=weights,
                    plasticity=True,
                    learning_rate=lr
                    )

    _, _, _, event_count, burst_count, events, bursts = pyramidal.run_simulation(
                v0, t0, tn, dt, n_patterns = int(n_plast_steps/20))
    
    cos_dist = pyramidal.cosine_distances
    neuron_type = np.ones(n_pyr)
    delta_t = 10

    plot_weights(pyramidal.W_CA3, m_CA3, m_EC)

    if True:
        plot_results(events, bursts, delta_t, event_count, burst_count, tn, t, CA3_act, EC_inp.T, neuron_type, n_pyr, dt, cos_dist)


def plot_weights(W, m_CA3, m_EC):
    # print(m_CA3, m_EC, m_CA3.shape, m_EC.shape, W.shape)
    sort_CA3 = np.argsort(m_CA3)
    sort_EC = np.argsort(m_EC)
    print(sort_EC.shape, sort_CA3.shape)

    fig, ax = plt.subplots(figsize = (10,8), dpi = 200)
    sorted_W = W[np.ix_(sort_EC, sort_CA3)]
    fig.colorbar(ax.imshow(sorted_W/np.sum(sorted_W)*50, origin='lower', aspect = 'auto'), ax=ax) # 50 n_pyr
    ax.set_title("CA3 weights")
    ax.set_xlabel("CA3 neuron")
    ax.set_ylabel("CA1 neuron")
    plt.savefig('plots/CA3_weights.png')
    plt.close()


def main():
    np.random.seed(42)
    

    len_track = 100. # 100
    tn = 1000
    av_running_speed = .2 # 0.2
    lr = .001 # 0.001
    n_plast_steps = 1000 # 100

    n_cells = {'pyramidal' : 50, 'inter_a' : 5, 'inter_b' : 5, 'CA3' : 30}

    run_simulation(len_track, av_running_speed, tn, n_cells, lr, n_plast_steps)

    # TODO: plot activity of CA1 neurons over space 
    # repeat that for unseeen enviornment, by reshuffling CA3 spatial centres and no top down input
    # Some measure of similarity between the two environments
    # Do the same in CA3, and we want similarity in CA1 to be higher than in CA3, do it column wise and row wise 

    # print(t.shape, x.shape)
    # 
    # plt.figure()
    # plt.plot(t, x)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Position (cm)')
    # plt.savefig('plots/simulated_run.png')
    # plt.close()


if __name__ ==  "__main__":
    main()

