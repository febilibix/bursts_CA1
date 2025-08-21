import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, zscore
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter
from final_docs.helpers import DT
from run_experiment_2d import run_simulation
import multiprocessing as mp
import os
from itertools import product


def cor_act_maps(act_map, out1, out2, pop_vec = False):
    act_map1, act_map2 = act_map[out1], act_map[out2]

    if pop_vec:
        act_map1, act_map2 = act_map1.T, act_map2.T
        act_map1 = zscore(act_map1, axis=0)
        act_map2 = zscore(act_map2, axis=0)

        # act_map1 = act_map1/np.sum(act_map1, axis=0)[np.newaxis, :]
        # act_map2 = act_map2/np.sum(act_map2, axis=0)[np.newaxis, :]
        # if np.any(np.isnan(act_map1_new)) or np.any(np.isnan(act_map1_new)):
        #     print("NaN values found in act_map1 or act_map2")
        #     print(out1, out2)
        #     print(act_map1, act_map2)
        #     quit()
        #     print(act_map1, act_map2)
        #     print(np.any(np.isnan(act_map1)), np.any(np.isnan(act_map2)))
        #     print(np.all(np.isnan(act_map1)), np.all(np.isnan(act_map2)))
        # 
        #     cor = np.zeros(act_map1.shape[0])
        # 
        #     for i in range(act_map1.shape[0]):
        #         cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]
        #     
        #     print(cor, np.mean(cor), np.std(cor))
        #     quit()
        
        

    cor = np.zeros(act_map1.shape[0])

    for i in range(act_map1.shape[0]):
        valid_idx = ~np.isnan(act_map1[i, :]) & ~np.isnan(act_map2[i, :])
        if np.any(valid_idx):
            cor[i] = pearsonr(act_map1[i, valid_idx], act_map2[i, valid_idx])[0]
        else:
            cor[i] = np.nan  # Handle case where all values are NaN
        # cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]
       
    return cor


def compute_cross_correlogram(act_maps, out1, out2):

    reshape_val = int(np.sqrt(act_maps[out1].shape[1]))
    act_map1 = act_maps[out1].reshape(act_maps[out1].shape[0], reshape_val, reshape_val)
    act_map2 = act_maps[out2].reshape(act_maps[out2].shape[0], reshape_val, reshape_val)
    
    n_cells = act_map1.shape[0]
    cross_corrs = []

    for i in range(n_cells):
        map1 = act_map1[i] - np.mean(act_map1[i])
        map2 = act_map2[i] - np.mean(act_map2[i])
        
        cross_corr = correlate2d(map1, map2, mode='full', boundary='fill')
        cross_corrs.append(cross_corr)
    
    avg_cross_corr = np.mean(cross_corrs, axis=0)
    return avg_cross_corr


def create_raincloud_plot(act_maps_exp, act_maps_cont, out1, out2, ax):
    # Example synthetic data

    cor_exp = cor_act_maps(act_maps_exp, out1, out2)
    cor_cont = cor_act_maps(act_maps_cont, out1, out2)

    data = {
        'Condition': ['exp'] * len(cor_exp) + ['cont'] * len(cor_cont),
        'Spatial correlation': np.concatenate((cor_exp, cor_cont))
    }
    
    df = pd.DataFrame(data)

    # plt.figure(figsize=(6, 6))

    palette = ['green', 'gray']

    # Raincloud plot with violin, box, and strip
    for i, condition in enumerate(['exp', 'cont']):
        df_use = df[df['Condition'] == condition]
        sns.violinplot(data=df_use, y='Spatial correlation', inner=None, linewidth=2, color=palette[i], split = True, alpha = 0.3, linecolor=palette[i], ax=ax)
    
    df['x'] = ' '
    sns.boxplot(data=df, x = 'x', hue='Condition', y='Spatial correlation', whis=[0, 100], width=0.8, gap = 0.5, palette=palette, fill = False, ax=ax)
    sns.stripplot(data=df, x= 'x', hue='Condition', y='Spatial correlation', jitter=0.2, palette=palette, alpha=0.1, dodge=True, size=3, ax=ax)

    # Add significance bar
    # x1, x2 = 0, 1
    # y, h, col = 1.1, 0.05, 'black'
    # plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * 0.5, y + h + 0.02, "n.s.", ha='center', va='bottom', color=col)

    # Customize axes
    ax.hlines(0, color='grey', lw=0.5, xmin=-1, xmax=1.5)
    ax.set_ylabel("Spatial correlation")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_ylim(-1.1, 1.3)
    ax.set_xlim(-0.5, 1.5)
    sns.despine(ax=ax)
    ax.legend([])
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/act_maps_exp_cont_{out1}_{out2}.png', dpi=300)
    # plt.close()



def plot_cross_correlogram(act_maps, out1, out2, ax, title=None, cmap='jet', sigma=1.5):

    correlogram = compute_cross_correlogram(act_maps, out1, out2)

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(correlogram, sigma=sigma)

    # Plot
    # fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(smoothed, cmap=cmap, origin='lower', interpolation='bilinear')

    # Crosshairs at center
    center_y, center_x = np.array(smoothed.shape) // 2
    ax.axhline(center_y, color='black', linewidth=2)
    ax.axvline(center_x, color='black', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=10)

    # ax.axis('off')
    # plt.tight_layout()
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/cross_cor_exp_test.png', dpi=300)


def plot_pv_corr_distributions(act_maps_exp, act_maps_cont, out1, out2, ax):

    color_exp, color_cont = 'green', 'gray'
    pv_corr_exp = cor_act_maps(act_maps_exp, out1, out2, pop_vec=True)
    pv_corr_cont = cor_act_maps(act_maps_cont, out1, out2, pop_vec=True)

    # if out1 == 'F2' and out2 == 'N1':
    #     print(pv_corr_exp, pv_corr_cont)
    #     quit()

    # fig, ax = plt.subplots(figsize=(3, 4))

    # Plot KDEs (smoothed histograms)
    sns.kdeplot(pv_corr_exp, fill=True, color=color_exp, alpha=0.6, linewidth=1.5, ax=ax)
    sns.kdeplot(pv_corr_cont, fill=True, color=color_cont, alpha=0.6, linewidth=1.5, ax=ax)

    # Plot medians as dashed lines
    ax.axvline(np.median(pv_corr_exp), color=color_exp, linestyle='--', linewidth=1.5)
    ax.axvline(np.median(pv_corr_cont), color=color_cont, linestyle='--', linewidth=1.5)

    # Labeling
    ax.set_xlabel('PV corr. coeff.')
    ax.set_ylabel('Frequency')
    # ax.set_xlim(-.2, 1)
    # .set_ylim(0, 45)
    sns.despine()
    # plt.tight_layout()
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/pv_corr_distributions_test.png', dpi=300)



def plot_delta_burst_barplot(burst_props, ax):

    burst_props_exp, burst_props_cont = burst_props

    means2 = [] # , means1 = [], []
    sems2 = [] #, sems1 = [], []

    for i, (out1, out2) in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):


        # diff_exp = get_diff_burst_prob(burst_props, out1, out2)

        # means1.append(np.mean(diff_exp))
        # sems1.append(np.std(diff_exp)) #/np.sqrt(len(diff_exp))) # TODO: OR SEM ??? : sems1.append(diff.std()/np.sqrt(len(diff)))

        diff_cont = get_diff_burst_prob(burst_props_cont, out1, out2)
        means2.append(np.mean(diff_cont))
        sems2.append(np.std(diff_cont)/np.sqrt(len(diff_cont))) # TODO: OR SEM ??? : sems2.append(diff.std()/np.sqrt(len(diff)))


    n = len(means2)
    x = np.arange(n)
    width = 0.25

    # Plot bars
    # ax.errorbar(x - width/2, means1, yerr=sems1, fmt='o', color='green', label='Group 1', capsize=5)
    ax.errorbar(x + width/2, means2, yerr=sems2, fmt='o', color='black', label='Group 2', capsize=5)

    # Horizontal reference line
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)

    # Labels and ticks
    ax.set_ylabel(r'$\Delta$ Burst prob.')
    # ax.set_ylim(-0.15, 0.13)
    ax.set_xticks(x)
    
    # Custom tick labels with rotation and multicolor
    custom_labels = [
        r'F1/F2$^{\it{LASER}}$', 
        r'F2$_{\it{LASER}}$ / $\it{N1}_{\it{LASER}}$', 
        r'$\it{N1}_{\it{LASER}}$ / $\it{N2}$'
    ]
    ax.set_xticklabels(custom_labels, rotation=45, ha='right')


def extract_active_cells(act_maps_exp, act_maps_cont):

    act_idxs, avg_acts = [], []

    for act_maps in [act_maps_exp, act_maps_cont]:
        all_trials = np.concatenate(list(act_maps.values()), axis=1)
        avg_act = np.mean(all_trials, axis=1)
        avg_acts.append(avg_act)
        # active_cells = np.where(avg_act > np.percentile(avg_act, 90))[0]
        act_idxs.append(np.argsort(-avg_act))

    rank_a = {idx: rank for rank, idx in enumerate(act_idxs[0])}
    rank_b = {idx: rank for rank, idx in enumerate(act_idxs[1])}

    # Step 3: Combine ranks (sum or average)
    all_indices = np.union1d(act_idxs[0], act_idxs[1])  # Unique indices
    combined_ranks = [
        (idx, rank_a.get(idx, len(avg_acts[0])) + rank_b.get(idx, len(avg_acts[1])))  # Default penalizes missing
        for idx in all_indices
    ]

    # Step 4: Sort by combined rank (lower = better) and pick top 40
    combined_ranks_sorted = sorted(combined_ranks, key=lambda x: x[1])
    active_cells = [idx for idx, rank in combined_ranks_sorted[:40]]

    for act_maps in [act_maps_exp, act_maps_cont]:
        for key in act_maps.keys():
            act_maps[key] = act_maps[key][active_cells, :]

    return act_maps_exp, act_maps_cont


def smooth_act_maps(act_maps_exp, act_maps_cont):
    
    for act_maps in [act_maps_exp, act_maps_cont]:
        for key in act_maps.keys():
            reshape_val = int(np.sqrt(act_maps[key].shape[1]))
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], reshape_val, reshape_val)
            act_maps[key] = gaussian_filter(act_maps[key], sigma=1.5)
            act_maps[key] = act_maps[key].reshape(act_maps[key].shape[0], -1)
    
    return act_maps_exp, act_maps_cont


def plot_all(act_maps_exp, act_maps_cont, burst_props, alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, beta, inh_lr, tau_inh):

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'w_pi_b = {W_pi_b}, W_ip_b = {W_ip_b}, tau_inh = {tau_inh}', fontsize=16, fontweight='bold')

    # act_maps_exp, act_maps_cont = extract_active_cells(act_maps_exp, act_maps_cont) 

    # act_maps_exp, act_maps_cont = smooth_act_maps(act_maps_exp, act_maps_cont)

    for i, (out1, out2) in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):
        axs = axes[i, :]
        axs[0].set_title(f'{out1} vs {out2}', loc='left', fontsize=14, fontweight='bold')

        axs[0].set_ylabel('opsin', fontsize=12)
        axs[1].set_ylabel('control', fontsize=12)
        
        plot_cross_correlogram(act_maps_exp, out1, out2, ax=axs[0])
        plot_cross_correlogram(act_maps_cont, out1, out2, ax=axs[1])
        create_raincloud_plot(act_maps_exp, act_maps_cont, out1, out2, ax=axs[2])
        plot_pv_corr_distributions(act_maps_exp, act_maps_cont, out1, out2, ax=axs[3])

    # create_raincloud_plot(ec_act_maps_exp,ec_act_maps_cont, 'F2', 'N1', ax=axes[0,4])
    # plot_pv_corr_distributions(ec_act_maps_exp, ec_act_maps_cont, 'F2', 'N1', ax=axes[1,4])

    print(burst_props)
    # plot_delta_burst_barplot(burst_props, axes[2,4])

    plt.tight_layout()
    print('here')
    plt.savefig(f'plots/full_experiment/2d_case/final_plots/v104_all_in_one_alpha_{alpha}_a_{a}_lr_{lr}_ma_{ma_pc}_mb_{mb_pc}_W_pi_a_{W_pi_a}_W_ip_a_{W_ip_a}_W_pi_b_{W_pi_b}_W_ip_b_{W_ip_b}_tau_a_{tau_a}_beta_{beta}_inhlr_{inh_lr}_tau_inh_{tau_inh}.png', dpi=300)
    plt.close('all')

    # if alpha == 0.1:
    plot_single_maps(act_maps_exp, act_maps_cont)



def get_firing_rates(dt, event_count, x_run, n_bins=1024, n_dim=1):
    ## TODO: Instead of having 1024 hardcoded here i might want to make it dependend on length of simulation

    firing_rates = np.zeros((event_count.shape[1], n_bins))
    x_run_reshaped = np.zeros((n_dim, n_bins))
    step_size = len(event_count)//n_bins
    x_run = x_run.reshape((n_dim, -1)) 
    
    for i in range(firing_rates.shape[1]):
        firing_rates[:, i] = np.sum(event_count[i * step_size:(i + 1) * step_size, :], axis = 0) / (step_size*dt)
        x_run_reshaped[:, i] = np.mean(x_run[:, i * step_size:(i + 1) * step_size], axis=1)

    return firing_rates, x_run_reshaped



def get_activation_map_2d(firing_rates, len_edge, x_run_reshaped, n_bins = 225):

    bins = np.arange(n_bins)
    n_cell = np.arange(firing_rates.shape[0])
    out_collector = {k : [] for k in product(n_cell, bins)}
    out = np.zeros((firing_rates.shape[0], n_bins))
    n_edge = int(np.sqrt(n_bins))
    position_bins = np.mgrid[.5:(len_edge-.5):n_edge*1j, .5:(len_edge-.5):n_edge*1j] 
    position_bins = np.vstack((position_bins[0].flatten(), position_bins[1].flatten())).T

    for idx, pos in enumerate(x_run_reshaped.T):
        bin_idx = np.argmin(np.linalg.norm(position_bins - pos, axis = 1)) 

        for i in range(firing_rates.shape[0]):
            out_collector[(i, bin_idx)].append(firing_rates[i, idx])

    for k, v in out_collector.items():
        out[k] = np.mean(v)

    return out, position_bins



def plot_single_maps(act_maps_exp, act_maps_cont):
    for i in range(act_maps_exp['F1'].shape[0]):
        fig = plt.figure(figsize=(12, 6))

        # Left 2x2 block (Experimental)
        left_gs = plt.GridSpec(2, 2, left=0.05, right=0.45, wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(left_gs[0, 0])
        ax2 = fig.add_subplot(left_gs[0, 1])
        ax3 = fig.add_subplot(left_gs[1, 0])
        ax4 = fig.add_subplot(left_gs[1, 1])

        # Right 2x2 block (Control)
        right_gs = plt.GridSpec(2, 2, left=0.55, right=0.95, wspace=0.3, hspace=0.3)
        ax5 = fig.add_subplot(right_gs[0, 0])
        ax6 = fig.add_subplot(right_gs[0, 1])
        ax7 = fig.add_subplot(right_gs[1, 0])
        ax8 = fig.add_subplot(right_gs[1, 1])

        axes1 = [ax1, ax2, ax3, ax4]  # Left block axes
        axes2 = [ax5, ax6, ax7, ax8]  # Right block axes

        # Add left/right titles
        fig.text(0.25, 0.95, "Experimental Group", ha='center', fontsize=14, weight='bold')
        fig.text(0.75, 0.95, "Control Group", ha='center', fontsize=14, weight='bold')

        for j, act_maps in enumerate([act_maps_exp, act_maps_cont]):
            axes = axes1 if j == 0 else axes2
            for k, out in enumerate(['F2', 'N1', 'F3', 'N2']):
                ax = axes[k]
                act_map = act_maps[out][i, :].reshape(int(np.sqrt(act_maps[out].shape[1])), 
                                             int(np.sqrt(act_maps[out].shape[1])))
                
                im = ax.imshow(act_map, cmap='jet', origin='lower')
                
                # Add colorbar (with adjusted padding)
                cbar_ax = ax.inset_axes([1.05, 0, 0.05, 1])
                fig.colorbar(im, cax=cbar_ax)
                ax.set_title(out, fontsize=12)
                ax.axis('off')


        # Manually adjust layout to avoid colorbar overlap
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)
        plt.savefig(f'plots/full_experiment/2d_case/final_plots/example_cells/cell_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()



def get_diff_burst_prob(burst_props, out1, out2):
    # burst_prop1 = burst_count[out1].sum(axis = 0)/event_count[out1].sum(axis = 0)
    # burst_prop2 = burst_count[out2].sum(axis = 0)/event_count[out2].sum(axis = 0)
    burst_prop1, burst_prop2 = burst_props[out1], burst_props[out2]

    # TODO: How to handle nan's?
    return np.where(burst_prop1 + burst_prop2 == 0, 0, (burst_prop2 - burst_prop1)/(burst_prop1 + burst_prop2))


def run_single_experiment(params):
    alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, beta, inh_lr, tau_inh = params

    # run_simulation(alpha, a, lr, ma_pc, mb_pc, W_ip_a, W_ip_b, tau_a, inh_lr)

    # for condition in ['exp', 'cont']:
    #     act_maps, burst_props = {}, {}
    #     for out in ['F1', 'F2', 'N1', 'F3', 'N2']:
    #         if not os.path.exists(f'data/2d_test/wpib_{W_pi_b}_wipb_{W_ip_b}_tauinh_{tau_inh}/{condition}_{out}.pkl'):
    #             return
    
    outputs = {}
    for condition in ['exp', 'cont']:
        act_maps, burst_props = {}, {}
        for out in ['F1', 'F2', 'N1', 'F3', 'N2']:


            with open(f'final_docs/simulations/data/2d_test/lr_{lr}_ma_{ma_pc}_mb_{mb_pc}_inhlr_{inh_lr}_alpha_{alpha}/{condition}_{out}.pkl', 'rb') as f:
                act_map, burst_prop, m_ECs, event_count, x_run = pickle.load(f)
                # fr, x_run_reshaped = get_firing_rates_gaussian(DT, event_count, x_run, sigma_s=.5, n_bins = 1024, n_dim=2)
            # fr, x_run_reshaped = get_firing_rates(DT, event_count, x_run, n_bins=2**14, n_dim=2)
            # act_map, _ = get_activation_map_2d(fr, 40, x_run_reshaped)
            # with open(f'final_docs/simulations/data/2d_test/lr_{lr}_ma_{ma_pc}_mb_{mb_pc}/{condition}_{out}.pkl', 'rb') as f:
            #     act_map, burst_prop, _ = pickle.load(f)
            act_maps[out] = act_map
            burst_props[out] = burst_prop
        outputs[condition] = (act_maps, burst_props)

    act_maps_exp, burst_props_exp = outputs['exp']
    act_maps_cont, burst_props_cont = outputs['cont']

    plot_all(act_maps_exp, act_maps_cont, (burst_props_exp, burst_props_cont), alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, beta, inh_lr, tau_inh)

    del outputs
    del act_maps_exp, burst_props_exp
    del act_maps_cont, burst_props_cont

    import gc
    gc.collect()


if __name__ == '__main__':

    #### TODO: I need to clean this whole functionality today! 
    ########## I guess what needs to be done is: 
    ########## 1. Run each condition
    ########## 2. Save the outputs in a dictionary - delete outputs there and then 
    ########## 3. Plot the results in a single function (which i guess i can call straight away for now but that will then go in the notebook)
    alphas = [0.5, 0.75, 1.5, 5, 10, 25] # TODO: I think this one is not tuned yet
    aas = [0.3]
    lrs = [40, 50, 60, 70, 80]   
    ma_pcs = [240, 360, 500] # , 400, 800, 1200]# [5000] 
    mb_pcs = [32, 48, 64] # , 64, 128, 256, 512]# [2000]
    W_ip_a = [1000]
    W_pi_a = [500] 
    W_ip_b = [4000]
    W_pi_b = [30] 
    tau_a = [1.0]
    inh_lrs = [10, 25, 50, 75, 100] ## TODO: MAybe even less?! probably need to tune again, last time i was running with weird normalization
    betas = [0.1] ## TODO: This one could still be tuned.  probably need to tune again, last time i was running with weird normalization
    tau_inh = [0.1]

    param_combinations = list(product(alphas, aas, lrs, ma_pcs, mb_pcs, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, betas, inh_lrs, tau_inh))
    os.makedirs('plots/full_experiment/2d_case/final_plots', exist_ok=True)

    # run_single_experiment(param_combinations[0])
    # quit()

    with mp.Pool(processes=mp.cpu_count()//4) as pool:
        pool.map(run_single_experiment, param_combinations)

    # with open(f'plots/full_experiment/2d_case/outputs.pkl', 'rb') as f:
    #     outputs = pickle.load(f)
    # act_maps_exp, burst_count_exp, event_count_exp, ec_act_maps_exp = outputs['exp']
    # act_maps_cont, burst_count_cont, event_count_cont, ec_act_maps_cont = outputs['cont']
    # plot_all(act_maps_exp, act_maps_cont, ec_act_maps_exp, ec_act_maps_cont,
    #         (burst_count_exp, burst_count_cont), (event_count_exp, event_count_cont),
    #          alphas[-1], aas[-1], lrs[-1], ma_pcs[-1], W_ip_a[-1], W_ip_b[-1], tau_a[-1])
