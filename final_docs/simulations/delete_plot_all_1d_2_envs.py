import sys 
sys.path.append('../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
sys.path.append('../../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
from helpers import *
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.gridspec as gridspec



def plot_all_conditions(all_act_maps, axs):
    
    # fig, axs = plt.subplots(1, 4, figsize=(10, 5), sharey=True)
    # axs = axs.flatten()
    activation_maps = all_act_maps[condition]
    # Find global vmin/vmax for colorbar scaling
    all_maps = [activation_maps[out] for out in ['F2', 'N1', 'F3', 'N2']]
    vmin = min(np.min(m) for m in all_maps)
    vmax = max(np.max(m) for m in all_maps)
    ims = []
    for i, out in enumerate(['F2', 'N1', 'F3', 'N2']):
        im = plot_firing_rates(axs[i], activation_maps[out], out, vmin=vmin, vmax=vmax)
        ims.append(im)
        if i > 0:
            axs[i].set_ylabel(None)
    # plt.tight_layout(rect=[0, 0, 0.92, 1])
    # fig.suptitle(condition)
    # # Add a single colorbar for all subplots
    # cbar = fig.colorbar(ims[0], ax=axs, orientation='vertical', fraction=0.025, pad=0.04)
    # cbar.set_label('Firing rate')
    # # plt.savefig(f"plots/full_experiment/activation_maps/{condition}_act_map.png", dpi=300)
    # plt.show()
    # plt.close()

path_name = 'data/1d_2envs_test2/'   

dirs = os.listdir(path_name)

for dir_name in dirs:
    # if dir_name != 'test':
    #     continue

    out_file = f"{path_name}/{dir_name}/plot_all_conditions.png"
    if not os.path.exists(f"{path_name}/{dir_name}/act_maps_and_pvs_inh_no_inh.pkl"):
        print(f"Skipping {out_file}, already exists.")
        continue

    with open(f"{path_name}/{dir_name}/act_maps_and_pvs_inh_normal.pkl", 'rb') as f:
        # with open(f"simulations/data/1d_2envs/param_tuning/v2_act_maps_and_pvs_inh_{inh_plast}_lr_15_ma_pc_160_mb_pc_24_w_ip_b_1000_w_pi_b_200_w_pi_a_200_w_ip_a_5000.pkl", 'rb') as f:
        a_T = pickle.load(f)

    try:
        with open(f"{path_name}/{dir_name}/act_maps_and_pvs_inh_no_inh.pkl", 'rb') as f:
            a_F = pickle.load(f)
    except EOFError:
        continue

    cont1_T = np.nanmedian(a_T['sp_per_condition']['control'][1])
    cont2_T, exp2_T = a_T['sp_per_condition']['control'][2], a_T['sp_per_condition']['exp'][2]
    # cont2 = act_maps_and_pvs_True['pvs_per_condition']['control'][2]
    cont1_F = np.nanmedian(a_F['sp_per_condition']['control'][1])
    exp2_F = np.nanmedian(a_F['sp_per_condition']['exp'][2])

    # if cont1_T >= .2 or cont1_F >= .2 or exp2_F >= .2 or np.nanmedian(cont2_T) < .95 or wasserstein_distance(cont2_T, exp2_T) > .3:
    if cont1_T >= .2 or cont1_F >= .2 or exp2_F >= .2 or np.nanmedian(cont2_T) < .96 or wasserstein_distance(cont2_T, exp2_T) > .4:
        print('bad parameter combination, skipping')
        continue

    fig = plt.figure(figsize=(20, 10))

    # Outer gridspec with 3 rows (your 3 figure rows)
    outer = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # --- Nested gridspec for top row ---
    # 1 row, 18 columns
    width_ratios = [0.8]*18
    width_ratios[8] = 0.3  # narrow the 9th column (index 8)
    width_ratios[9] = 0.3  # narrow the 10th column (index 9)

    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 18,
        subplot_spec=outer[0],
        width_ratios=width_ratios,
        wspace=0.05  # small horizontal spacing between columns
    )

    axes_row0 = []
    for i in range(18):
        if i == 0:
            ax = fig.add_subplot(gs_top[0, i])
            axes_row0.append(ax)
        elif i != 8 and i != 9:
            ax = fig.add_subplot(gs_top[0, i], sharey=axes_row0[0])
            plt.setp(ax.get_yticklabels(), visible=False)
            axes_row0.append(ax)

    # --- Nested gridspec for middle row ---
    # 1 row, 18 columns (6 subplots spanning 3 columns each)
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 18,
        subplot_spec=outer[1],
        wspace=0.3  # larger spacing horizontally here
    )

    axes_row1 = []
    for i in range(6):
        start = i * 3
        if i == 0:
            ax = fig.add_subplot(gs_mid[0, start:start+3])
        else:
            ax = fig.add_subplot(gs_mid[0, start:start+3], sharey=axes_row1[0])
            plt.setp(ax.get_yticklabels(), visible=False)
        axes_row1.append(ax)

    # --- Nested gridspec for bottom row ---
    # 1 row, 18 columns (6 subplots spanning 3 columns each), shared y
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 18,
        subplot_spec=outer[2],
        wspace=0.05  # tighter spacing again, if you like
    )

    axes_row2 = []
    for i in range(6):
        start = i * 3
        if i == 0:
            ax = fig.add_subplot(gs_bot[0, start:start+3])
        else:
            ax = fig.add_subplot(gs_bot[0, start:start+3], sharey=axes_row2[0])
            plt.setp(ax.get_yticklabels(), visible=False)
        axes_row2.append(ax)

    mdn_of_int = np.nan

    all_ps = {}

    for inh_plast in ['normal', 'no_inh']:
    # for inh_plast in [15]: # [1, 10, 15, 20, 25, 100]

        with open(f"{path_name}/{dir_name}/act_maps_and_pvs_inh_{inh_plast}.pkl", 'rb') as f:
        # with open(f"simulations/data/1d_2envs/param_tuning/v2_act_maps_and_pvs_inh_{inh_plast}_lr_15_ma_pc_160_mb_pc_24_w_ip_b_1000_w_pi_b_200_w_pi_a_200_w_ip_a_5000.pkl", 'rb') as f:
            act_maps_and_pvs = pickle.load(f)

        pvs_per_condition = act_maps_and_pvs['pvs_per_condition']
        all_act_maps = act_maps_and_pvs['all_act_maps']
        sp_per_condition = act_maps_and_pvs['sp_per_condition']

        all_ps[inh_plast] = []
        
        for j, measure in enumerate([pvs_per_condition, sp_per_condition]):

            for i, comp in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):
                i_use = i if inh_plast == 'normal' else i + 3
                ## TODO: Here I might want to adapt such that for the spatial correlation i use the raincloud plot which i am also using for 2D
                data1, data2 = measure['exp'][i], measure['control'][i]
                stat, p = ks_2samp(data1, data2)
                all_ps[inh_plast].append(p)
                data1_nonan, data2_nonan = np.where(~np.isnan(data1), data1, 0), np.where(~np.isnan(data2), data2, 0)
                was_d = wasserstein_distance(data1_nonan, data2_nonan)
                if j == 0:
                    plot_pv_corr_distributions((data1, data2), comp[0], comp[1], axes_row1[i_use], p=(p,))
                    axes_row1[i_use].set_title(f"{comp[0]} vs {comp[1]}, d = {was_d:.3f}")
                    axes_row1[i_use].set_xlabel(f"")
                else:
                    create_raincloud_plot((data1, data2), comp[0], comp[1], axes_row2[i_use], p=(p,))
                    axes_row2[i_use].set_title(f"{comp[0]} vs {comp[1]}, d = {was_d:.3f}")


                
                print(f"KS test for {comp[0]} vs {comp[1]}: stat={stat}, p={p}, wasserstein distance={was_d}")

        all_ps[inh_plast] = np.array(all_ps[inh_plast])

        for condition in all_act_maps.keys():
            if inh_plast == 'normal' and condition == 'exp':
                axs = axes_row0[:4] 
            elif inh_plast == 'normal' and condition == 'control':
                axs = axes_row0[4:8]
            elif inh_plast == 'no_inh' and condition == 'exp':
                axs = axes_row0[8:12]
            elif inh_plast == 'no_inh' and condition == 'control':
                axs = axes_row0[12:16]
        
            plot_all_conditions(all_act_maps, axs = axs)

    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.savefig(out_file)
    os.makedirs('/'.join(out_file.replace(path_name, f'{path_name[:-1]}_verygood').split('/')[:-1]), exist_ok=True)
    # axes_row2[1].set_title(f"{comp[0]} vs {comp[1]}, Mdn = {mdn_of_int:.3f}")
    plt.savefig(out_file.replace(path_name, f'{path_name[:-1]}_verygood'))
    # 
    for inh_plast in ['normal', 'no_inh']:
        if sum(all_ps[inh_plast] >= 0.05) >= 1:
            os.makedirs('/'.join(out_file.replace(path_name, f'{path_name[:-1]}_perfect').split('/')[:-1]), exist_ok=True)
            plt.savefig(out_file.replace(path_name, f'{path_name[:-1]}_perfect'))

    plt.close(fig)
    