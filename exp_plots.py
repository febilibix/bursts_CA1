import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter
from run_experiment import run_simulation



def cor_act_maps(act_map, out1, out2, pop_vec = False):
    act_map1, act_map2 = act_map[out1], act_map[out2]

    if pop_vec:
        act_map1, act_map2 = act_map1.T, act_map2.T

    cor = np.zeros(act_map1.shape[0])

    for i in range(act_map1.shape[0]):
        cor[i] = pearsonr(act_map1[i, :], act_map2[i, :])[0]
       
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 45)
    sns.despine()
    # plt.tight_layout()
    # plt.savefig(f'plots/full_experiment/2d_case/final_plots/pv_corr_distributions_test.png', dpi=300)



def plot_delta_burst_barplot(burst_counts, event_counts, ax):

    burst_count_exp, burst_count_cont = burst_counts
    event_count_exp, event_count_cont = event_counts

    means1, means2 = [], []
    sems1, sems2 = [], []

    for i, (out1, out2) in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):


        diff_exp = get_diff_burst_prob(burst_count_exp, event_count_exp, out1, out2)

        means1.append(np.mean(diff_exp))
        sems1.append(np.std(diff_exp)) #/np.sqrt(len(diff_exp))) # TODO: OR SEM ??? : sems1.append(diff.std()/np.sqrt(len(diff)))

        diff_cont = get_diff_burst_prob(burst_count_cont, event_count_cont, out1, out2)
        means2.append(np.mean(diff_cont))
        sems2.append(np.std(diff_cont)) #/np.sqrt(len(diff_cont))) # TODO: OR SEM ??? : sems2.append(diff.std()/np.sqrt(len(diff)))


    n = len(means1)
    x = np.arange(n)
    width = 0.25

    # Plot bars
    ax.errorbar(x - width/2, means1, yerr=sems1, fmt='o', color='green', label='Group 1', capsize=5)
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


def plot_all(act_maps_exp, act_maps_cont, burst_counts, event_counts, alpha, a, lr):

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'alpha = {alpha}, a = {a}, lr = {lr}', fontsize=16, fontweight='bold')

    for i, (out1, out2) in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):
        axs = axes[i, :]
        axs[0].set_title(f'{out1} vs {out2}', loc='left', fontsize=14, fontweight='bold')

        axs[0].set_ylabel('opsin', fontsize=12)
        axs[1].set_ylabel('control', fontsize=12)
        
        plot_cross_correlogram(act_maps_exp, out1, out2, ax=axs[0])
        plot_cross_correlogram(act_maps_cont, out1, out2, ax=axs[1])
        create_raincloud_plot(act_maps_exp, act_maps_cont, out1, out2, ax=axs[2])
        plot_pv_corr_distributions(act_maps_exp, act_maps_cont, out1, out2, ax=axs[3])

    plot_delta_burst_barplot(burst_counts, event_counts, axes[2,4])

    plt.tight_layout()
    plt.savefig(f'plots/full_experiment/2d_case/final_plots/all_in_one.png', dpi=300)


def get_diff_burst_prob(burst_count, event_count, out1, out2):
    burst_prop1 = burst_count[out1].sum(axis = 0)/event_count[out1].sum(axis = 0)
    burst_prop2 = burst_count[out2].sum(axis = 0)/event_count[out2].sum(axis = 0)

    # TODO: How to handle nan's?
    return np.where(burst_prop1 + burst_prop2 == 0, 0, (burst_prop2 - burst_prop1)/(burst_prop1 + burst_prop2))

alphas = [0.001, 0.005, 0.01, 0.05]
aas = [0.1, 0.2, 0.3, 0.4]
lrs = [1, 5, 10, 20]

for alpha in alphas:
    for a in aas:
        for lr in lrs:

            run_simulation(alpha, a, lr)

            with open(f'plots/full_experiment/2d_case/act_maps_exp.pkl', 'rb') as f:
                act_maps_exp, pos_bins_exp, burst_count_exp, event_count_exp = pickle.load(f)

            with open(f'plots/full_experiment/2d_case/act_maps_cont.pkl', 'rb') as f:
                act_maps_cont, pos_bins_cont, burst_count_cont, event_count_cont = pickle.load(f)

            plot_all(act_maps_exp, act_maps_cont, (burst_count_exp, burst_count_cont), (event_count_exp, event_count_cont), alpha, a, lr)


