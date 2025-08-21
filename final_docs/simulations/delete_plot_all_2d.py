import sys 
sys.path.append('../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
sys.path.append('../../') # TODO: This is a temporary fix, i will need to properly structure the code and remove this line
from helpers import *
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.gridspec as gridspec
from tqdm import tqdm




def plot_one_file(path, out_path):
    for condition in ['exp', 'cont']:
        act_maps, burst_props = {}, {}
        for out in ['F1', 'F2', 'N1', 'F3', 'N2']:
            try:
                with open(path + '/' + condition + '_' + out + '.pkl', 'rb') as f:
                    act_map, burst_prop = pickle.load(f)
            except EOFError:
                return
            # fr, x_run_reshaped = get_firing_rates_gaussian(DT, event_count, x_run, sigma_s=.5, n_bins = 1024, n_dim=2)
            # fr, x_run_reshaped = get_firing_rates(DT, event_count, x_run, n_bins=2**14, n_dim=2)
            # act_map, _ = get_activation_map_2d(fr, 40, x_run_reshaped)
            act_maps[out] = act_map
            burst_props[out] = burst_prop

        outputs[condition] = (act_maps, burst_props)

    act_maps_exp, burst_props_exp = outputs['exp']
    act_maps_cont, burst_props_cont = outputs['cont']


    act_maps_exp, act_maps_cont = smooth_act_maps(act_maps_exp, act_maps_cont, sigma=1)


    sp_cont_1 = cor_act_maps_2d(act_maps_cont['F2'], act_maps_cont['N1'], which='spatial')
    sp_exp_1 = cor_act_maps_2d(act_maps_exp['F2'], act_maps_exp['N1'], which='spatial')
    sp_exp_2 = cor_act_maps_2d(act_maps_exp['N1'], act_maps_exp['N2'], which='spatial')

    if np.all(np.isnan(sp_cont_1)) or np.all(np.isnan(sp_exp_2)):
        print('Skipping, no data')
        return
    
    if not (np.nanmedian(sp_cont_1) < 0.05 and np.nanmedian(sp_exp_1) > 0.5 and np.nanmedian(sp_exp_2) > 0.3):
        # print('bad params, skipping')
        # print('Mdn cont 1: ', np.nanmedian(sp_cont_1), ', ', 'Mdn exp 2: ', np.nanmedian(sp_exp_2))
        return
 

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    flag1, flag2 = False, False
    # act_maps_exp, act_maps_cont = extract_active_cells(act_maps_exp, act_maps_cont) 
    
    for i, (out1, out2) in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):
        axs = axes[i, :]
        axs[0].set_title(f'{out1} vs {out2}', loc='left', fontsize=14, fontweight='bold')

        axs[0].set_ylabel('opsin', fontsize=12)
        axs[1].set_ylabel('control', fontsize=12)

        print(act_maps_exp[out1].shape, act_maps_exp[out2].shape)
        # print(m_ECs[0].shape, m_ECs[1])

        plot_cross_correlogram(act_maps_exp, out1, out2, ax=axs[0])
        plot_cross_correlogram(act_maps_cont, out1, out2, ax=axs[1])
        sp_exp = cor_act_maps_2d(act_maps_exp[out1], act_maps_exp[out2], which='spatial')
        sp_cont = cor_act_maps_2d(act_maps_cont[out1], act_maps_cont[out2], which='spatial')
        # if (out1, out2) == ('N1', 'N2'):
        #     print(sp_exp - sp_cont)
        create_raincloud_plot((sp_exp, sp_cont), out1, out2, ax=axs[2])
        pv_exp = cor_act_maps_2d(act_maps_exp[out1], act_maps_exp[out2], which='pv')
        pv_cont = cor_act_maps_2d(act_maps_cont[out1], act_maps_cont[out2], which='pv')
        plot_pv_corr_distributions((pv_exp, pv_cont), out1, out2, ax=axs[3])

    # create_raincloud_plot(ec_act_maps_exp,ec_act_maps_cont, 'F2', 'N1', ax=axes[0,4])
    # plot_pv_corr_distributions(ec_act_maps_exp, ec_act_maps_cont, 'F2', 'N1', ax=axes[1,4])

    # print(burst_props)
    # plot_delta_burst_barplot(burst_props, axes[2,4])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    os.makedirs(os.path.dirname(out_path.replace('plots', 'plots_verygood')), exist_ok=True)
    plt.savefig(out_path.replace('plots', 'plots_verygood'), dpi=300)
    # plt.show()

    # if alpha == 0.1:
    # plot_single_maps(act_maps_exp, act_maps_cont, [])


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

path_name = 'data/2d_test5/'

dirs = os.listdir(path_name)

for dir_name in tqdm(dirs):
    if 'plots' in dir_name:
        continue
    # if dir_name != 'test':
    #     continue

    out_file = f"{path_name}/plots/{dir_name}/plot_all_conditions.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    all_outs = ['cont_' + i for i in ['F1', 'F2', 'N1', 'F3', 'N2']] + ['exp_' + i for i in ['F1', 'F2', 'N1', 'F3', 'N2']]
    exists = [os.path.exists(f"{path_name}/{dir_name}/{out}.pkl") for out in all_outs]


    if not np.all(exists):
        print(f"Skipping {out_file}, not enough files")
        continue

    ## TODO: In this part i will somehow implement automatic parameter detection

    # with open(f"data/2d_test/{dir_name}/act_maps_and_pvs_inh_normal.pkl", 'rb') as f:
    #     # with open(f"simulations/data/1d_2envs/param_tuning/v2_act_maps_and_pvs_inh_{inh_plast}_lr_15_ma_pc_160_mb_pc_24_w_ip_b_1000_w_pi_b_200_w_pi_a_200_w_ip_a_5000.pkl", 'rb') as f:
    #     act_map, m_EC = pickle.load(f)
# 
# 
    # if cont1_T >= .2 or cont1_F >= .2 or exp2_F >= .2 or np.nanmedian(cont2_T) < .95 or wasserstein_distance(cont2_T, exp2_T) > .3:
    #     print('bad parameter combination, skipping')
    #     continue

    outputs = {}
    file_name = f"{path_name}/{dir_name}"

    plot_one_file(file_name, out_file)




    # plt.savefig(out_file)
    # os.makedirs('/'.join(out_file.replace('param_tuning7', 'param_tuning7_reallygood').split('/')[:-1]), exist_ok=True)
    # # axes_row2[1].set_title(f"{comp[0]} vs {comp[1]}, Mdn = {mdn_of_int:.3f}")
    # plt.savefig(out_file.replace('param_tuning7', 'param_tuning7_reallygood'))
# 
    # for inh_plast in ['normal', 'no_inh']:
    #     if sum(all_ps[inh_plast] >= 0.05) >= 1:
    #         os.makedirs('/'.join(out_file.replace('param_tuning7', 'param_tuning7_perfect').split('/')[:-1]), exist_ok=True)
    #         plt.savefig(out_file.replace('param_tuning7', 'param_tuning7_perfect'))
# 
    # plt.close(fig)
    