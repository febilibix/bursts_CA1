import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuron import PyramidalCells


ENVIRONMENTS_RUNS = {
    # "F1": {'new_env': False, 'top_down': False, 'a': 0},
    "F2": {'new_env': False, 'top_down': False},
    "N1": {'new_env': True,  'top_down': False},
    # "F3": {'new_env': False, 'top_down': True },
    "N2": {'new_env': True,  'top_down': True },
    }


def simulate_run(len_track = 200, av_running_speed = 20, dt = 0.01, tn = 1000):
    bins = np.arange(0., len_track)
    fps = 1/dt
    n_runs = int(2*tn/(len_track/av_running_speed))

    running_speed_a = running_speed_b = np.ones(n_runs) * av_running_speed
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


def get_firing_rates(pyramidal, event_count):

    firing_rates = np.zeros((event_count.shape[1], 1024))
    step_size = len(event_count)//firing_rates.shape[1]
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*pyramidal.dt)

    return firing_rates


def plot_firing_rates(fig, ax, firing_rates, m_EC, out):

    sort_TD = np.argsort(m_EC)
    sorted_fr = firing_rates[np.ix_(sort_TD, np.arange(firing_rates.shape[1]))]

    ###### IDEALLY I WOULD SOMEHOW PUT IT IN BINS ACCORDING TO POSITION AND THEN MEAN OVER THAT

    lap = sorted_fr.shape[1] // 16
    mean_firing_rates = sum([sorted_fr[:, i*lap:(i+1)*lap] for i in range(16)]) / 16

    half = mean_firing_rates.shape[1] // 2
    mean_firing_rates = (sorted_fr[:, :half] + sorted_fr[:, :-half-1:-1]) / 2

    extent = [0, 100, 0, mean_firing_rates.shape[0]]
    im = ax.imshow(mean_firing_rates, aspect='auto', extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{out}")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Neuron")

    return fig, ax



def run_simulation(alpha, C):
    lr = 0.0001
    t_epoch = 0.5
    speed = 20
    len_track = 100. 
    dt = 0.001
    tn = len_track/speed*32
    a = 0.6 # 1 - similarity between environments
    n_cells = {'pyramidal' : 50, 'inter_a' : 5, 'inter_b' : 5, 'CA3' : 30}

    tau_b = 0.125
    tau = 2*tau_b
    I_a = 8

    pyramidal = PyramidalCells(n_cells, weights = dict(), learning_rate = lr, dt = dt)

    t_run, x_run = simulate_run(len_track, speed, dt, tn)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    
    ### F1 
    pyramidal.learn_place_cells(t_run, x_run, t_epoch)

    for i, (out, params) in enumerate(ENVIRONMENTS_RUNS.items()):
        print(out)
        event_count = pyramidal.retrieve_place_cells(t_run, x_run, **params, a = 0.6, t_per_epoch=t_epoch)
        fr = get_firing_rates(pyramidal, event_count)
        fig, axs[i] = plot_firing_rates(fig, axs[i], fr, pyramidal.m_EC, out)

    plt.tight_layout()
    plt.savefig(f"plots/full_experiment/firing_rates_CA1_alpha_{alpha}_C_{C}.png")
    plt.close()
    print("DONE")

def main():
    np.random.seed(1903)
    alphas = np.linspace(0.1, 0.9, 9)
    Cs = np.linspace(0.1, 0.9, 9)

    alphas = [0.3]
    Cs = [0.2, 0.3, 0.4]

    for alpha in alphas:
        for C in Cs:
            # if alpha == 0.1 and C <= 0.3:
            #     continue
            alpha, C = round(alpha, 2), round(C, 2)
            run_simulation(alpha, C)
    

if __name__ == '__main__':
    main()