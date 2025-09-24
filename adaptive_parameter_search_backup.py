#!/usr/bin/env python3
"""
Adaptive Parameter Search for CA1 Hippocampal Model
This script runs parameter searches, evaluates results, and adapts the parameter grid
to find optimal combinations that produce the desired correlation patterns.
"""

import sys
sys.path.append('final_docs/')
from final_docs.simulations.simulate_2d import run_simulation
from final_docs.helpers import cor_act_maps_2d
import numpy as np
import pickle
import os
import multiprocessing as mp
from itertools import product
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import time
import json

class AdaptiveParameterSearch:
    def __init__(self, max_iterations=10, results_dir='adaptive_search_results'):
        """
        Initialize the adaptive parameter search.
        
        Args:
            max_iterations: Maximum number of search iterations
            results_dir: Directory to store results and plots
        """
        self.max_iterations = max_iterations
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Target criteria for good results
        self.target_criteria = {
            'F1_vs_F2': {
                'both_conditions_mean': [0.8, 1.0],
                'wasserstein_distance': [0.0, 0.1]
            },
            'F2_vs_N1': {
                'experimental_mean': [0.7, 1.0],
                'control_mean': [-0.05, 0.05],
                'wasserstein_distance': [0.7, float('inf')]
            },
            'N1_vs_N2': {
                'both_conditions_mean': [0.7, 1.0],
                'wasserstein_distance': [0.0, 0.2]
            }
        }
        
        # Initial parameter ranges (targeting 100-128 combinations)
        self.param_ranges = {
            'alphas': [0.1, 0.7, 1.5],     # 3 values
            'lrs': [5, 25],                 # 2 values  
            'ma_pcs': [180, 360, 750],      # 3 values
            'mb_pcs': [50, 100],             # 2 values
            'inh_lrs': [25, 125, 375],      # 3 values
            'W_ip_a': [1000],
            'W_pi_a': [500],
            'W_ip_b': [4000],
            'W_pi_b': [30],
            'tau_a': [1.0],
            'betas': [0.1],
            'tau_inh': [0.1]
        }
        
        # Fixed parameter (not tunable)
        self.fixed_params = {
            'aas': 0.3
        }
        
        # Store results from all iterations
        self.all_results = []
        self.best_results = []
        
    def run_simulation_batch(self, param_combinations, iteration):
        """Run a batch of simulations with given parameters."""
        print(f"\n=== Running iteration {iteration} with {len(param_combinations)} parameter combinations ===")
        
        # Create output directory for this iteration
        iter_dir = os.path.join(self.results_dir, f'iteration_{iteration}')
        os.makedirs(iter_dir, exist_ok=True)
        
        # Run simulations in parallel
        with mp.Pool(processes=48) as pool:  # Use 48 CPUs for efficient processing
            results = list(tqdm(
                pool.imap(self._run_single_simulation, param_combinations),
                total=len(param_combinations),
                desc=f"Iteration {iteration}"   
            ))
        
        # Filter out failed simulations
        valid_results = [r for r in results if r is not None]
        print(f"Completed {len(valid_results)} out of {len(param_combinations)} simulations")
        
        return valid_results, iter_dir
    
    def _run_single_simulation(self, params):
        """Run a single simulation with given parameters."""
        try:
            # Unpack the 12 tunable parameters
            alpha, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh = params
            
            # Use fixed value for aas
            a = self.fixed_params['aas']
            
            # Create unique directory name for this parameter combination
            param_name = f'alpha_{alpha}_lr_{lr}_mapc_{ma_pc}_mbpc_{mb_pc}_lrinh_{inh_lr}'
            sim_dir = os.path.join(self.results_dir, 'simulations', param_name)
            os.makedirs(sim_dir, exist_ok=True)
            
            # First, we need to create the trajectory data that the simulation expects
            # The simulation script expects data/2d_test2/run_F1.pkl, etc.
            self._create_trajectory_data()
            
            # Change to simulation directory and run simulation
            original_cwd = os.getcwd()
            os.chdir(sim_dir)
            
            try:
                # Run the simulation
                run_simulation(alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, 
                              W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh)
            finally:
                # Always return to original directory
                os.chdir(original_cwd)
            
            # Return parameter info for later analysis
            return {
                'params': params,
                'param_name': param_name,
                'sim_dir': sim_dir
            }
            
        except Exception as e:
            print(f"Simulation failed for params {params}: {e}")
            return None
    
    def _create_trajectory_data(self):
        """Create the trajectory data files that the simulation script expects."""
        try:
            # Create the data directory structure
            data_dir = 'data/2d_test2'
            os.makedirs(data_dir, exist_ok=True)
            
            # Import the simulation function
            from final_docs.helpers import simulate_2d_run
            
            # Generate trajectory data for each environment
            environments = ['F1', 'F2', 'N1', 'F3', 'N2']
            seed = 101
            
            for env in environments:
                # Check if file already exists
                if os.path.exists(f'{data_dir}/run_{env}.pkl'):
                    continue
                
                # Generate trajectory
                t_run, x_run = simulate_2d_run(len_edge=50, av_running_speed=20, dt=0.001, tn=250, seed=seed)
                seed += 1
                
                # Save trajectory data
                with open(f'{data_dir}/run_{env}.pkl', 'wb') as f:
                    pickle.dump((t_run, x_run), f)
                    
        except Exception as e:
            print(f"Warning: Could not create trajectory data: {e}")
            # Continue anyway, the simulation might fail but we'll handle it
    
    def evaluate_results(self, results, iter_dir):
        """Evaluate simulation results and compute correlations."""
        print(f"\n=== Evaluating results for iteration ===")
        
        evaluated_results = []
        
        for result in tqdm(results, desc="Evaluating results"):
            try:
                # Load simulation results
                sim_dir = result['sim_dir']
                param_name = result['param_name']
                
                # Check if all required files exist
                # NOTE: Due to double-nested directory structure in run_simulation,
                # the actual files are in sim_dir/param_name/ not sim_dir/
                actual_sim_dir = os.path.join(sim_dir, param_name)
                
                required_files = []
                for condition in ['exp', 'cont']:
                    for out in ['F1', 'F2', 'N1', 'F3', 'N2']:
                        required_files.append(f'{actual_sim_dir}/{condition}_{out}.pkl')
                
                if not all(os.path.exists(f) for f in required_files):
                    continue
                
                # Load activation maps
                act_maps = {'exp': {}, 'cont': {}}
                for condition in ['exp', 'cont']:
                    for out in ['F1', 'F2', 'N1', 'F3', 'N2']:
                        with open(f'{actual_sim_dir}/{condition}_{out}.pkl', 'rb') as f:
                            act_map, _ = pickle.load(f)
                        act_maps[condition][out] = act_map
                
                # Compute correlations and evaluate
                evaluation = self._evaluate_single_result(act_maps, result['params'])
                if evaluation:
                    result['evaluation'] = evaluation
                    evaluated_results.append(result)
                    
                    # If this is a very good result, create a plot
                    if self._is_very_good_result(evaluation):
                        self._create_result_plot(act_maps, evaluation, sim_dir, param_name)
                        self.best_results.append(result)
                
            except Exception as e:
                print(f"Evaluation failed for {result['param_name']}: {e}")
                continue
        
        return evaluated_results
    
    def _evaluate_single_result(self, act_maps, params):
        """Evaluate a single simulation result."""
        try:
            # Compute spatial and PV correlations for all comparisons
            correlations = {}
            
            for out1, out2 in [('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]:
                # Spatial correlations
                sp_exp = cor_act_maps_2d(act_maps['exp'][out1], act_maps['exp'][out2], which='spatial')
                sp_cont = cor_act_maps_2d(act_maps['cont'][out1], act_maps['cont'][out2], which='spatial')
                
                # PV correlations
                pv_exp = cor_act_maps_2d(act_maps['exp'][out1], act_maps['exp'][out2], which='pv')
                pv_cont = cor_act_maps_2d(act_maps['cont'][out1], act_maps['cont'][out2], which='pv')
                
                # Compute statistics
                sp_exp_mean = np.nanmean(sp_exp)
                sp_cont_mean = np.nanmean(sp_cont)
                pv_exp_mean = np.nanmean(pv_exp)
                pv_cont_mean = np.nanmean(pv_cont)
                
                # Compute Wasserstein distances
                sp_wasserstein = wasserstein_distance(
                    sp_exp[~np.isnan(sp_exp)], 
                    sp_cont[~np.isnan(sp_cont)]
                )
                pv_wasserstein = wasserstein_distance(
                    pv_exp[~np.isnan(pv_exp)], 
                    pv_cont[~np.isnan(pv_cont)]
                )
                
                correlations[f'{out1}_vs_{out2}'] = {
                    'spatial': {
                        'exp_mean': sp_exp_mean,
                        'cont_mean': sp_cont_mean,
                        'wasserstein': sp_wasserstein
                    },
                    'pv': {
                        'exp_mean': pv_exp_mean,
                        'cont_mean': pv_cont_mean,
                        'wasserstein': pv_wasserstein
                    }
                }
            
            return correlations
            
        except Exception as e:
            print(f"Correlation computation failed: {e}")
            return None
    
    def _is_very_good_result(self, evaluation):
        """Check if a result meets the very good criteria."""
        try:
            # F1 vs F2: both conditions should have high correlation
            f1f2_sp = evaluation['F1_vs_F2']['spatial']
            f1f2_pv = evaluation['F1_vs_F2']['pv']
            
            # F2 vs N1: experimental should be high, control should be low
            f2n1_sp = evaluation['F2_vs_N1']['spatial']
            f2n1_pv = evaluation['F2_vs_N1']['pv']
            
            # N1 vs N2: both conditions should have high correlation
            n1n2_sp = evaluation['N1_vs_N2']['spatial']
            n1n2_pv = evaluation['N1_vs_N2']['pv']
            
            # Check criteria
            f1f2_good = (0.8 <= f1f2_sp['exp_mean'] <= 1.0 and 
                         0.8 <= f1f2_sp['cont_mean'] <= 1.0 and
                         0.8 <= f1f2_pv['exp_mean'] <= 1.0 and
                         0.8 <= f1f2_pv['cont_mean'] <= 1.0)
            
            f2n1_good = (0.7 <= f2n1_sp['exp_mean'] <= 1.0 and
                         -0.05 <= f2n1_sp['cont_mean'] <= 0.05 and
                         0.7 <= f2n1_pv['exp_mean'] <= 1.0 and
                         -0.05 <= f2n1_pv['cont_mean'] <= 0.05)
            
            n1n2_good = (0.7 <= n1n2_sp['exp_mean'] <= 1.0 and
                         0.7 <= n1n2_sp['cont_mean'] <= 1.0 and
                         0.7 <= n1n2_pv['exp_mean'] <= 1.0 and
                         0.7 <= n1n2_pv['cont_mean'] <= 1.0)
            
            return f1f2_good and f2n1_good and n1n2_good
            
        except Exception as e:
            print(f"Error checking very good result: {e}")
            return False
    
    def _create_result_plot(self, act_maps, evaluation, sim_dir, param_name):
        """Create a plot for a very good result."""
        try:
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            for i, (out1, out2) in enumerate([('F1', 'F2'), ('F2', 'N1'), ('N1', 'N2')]):
                axs = axes[i, :]
                axs[0].set_title(f'{out1} vs {out2}', loc='left', fontsize=14, fontweight='bold')
                axs[0].set_ylabel('opsin', fontsize=12)
                axs[1].set_ylabel('control', fontsize=12)
                
                # Plot activation maps
                self._plot_activation_map(act_maps['exp'][out1], axs[0], f'{out1} (exp)')
                self._plot_activation_map(act_maps['exp'][out2], axs[1], f'{out2} (exp)')
                
                # Plot correlation distributions
                comp_key = f'{out1}_vs_{out2}'
                if comp_key in evaluation:
                    sp_exp = cor_act_maps_2d(act_maps['exp'][out1], act_maps['exp'][out2], which='spatial')
                    sp_cont = cor_act_maps_2d(act_maps['cont'][out1], act_maps['cont'][out2], which='spatial')
                    pv_exp = cor_act_maps_2d(act_maps['exp'][out1], act_maps['exp'][out2], which='pv')
                    pv_cont = cor_act_maps_2d(act_maps['cont'][out1], act_maps['cont'][out2], which='pv')
                    
                    # Spatial correlation plot
                    self._plot_correlation_distribution(sp_exp, sp_cont, axs[2], 'Spatial Correlation')
                    
                    # PV correlation plot
                    self._plot_correlation_distribution(pv_exp, pv_cont, axs[3], 'PV Correlation')
            
            plt.tight_layout()
            plot_path = os.path.join(sim_dir, f'{param_name}_very_good_result.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Created plot for very good result: {plot_path}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def _plot_activation_map(self, act_map, ax, title):
        """Plot a single activation map."""
        im = ax.imshow(act_map, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Spatial bin')
        ax.set_ylabel('Neuron')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_correlation_distribution(self, exp_data, cont_data, ax, title):
        """Plot correlation distribution comparison."""
        exp_data = exp_data[~np.isnan(exp_data)]
        cont_data = cont_data[~np.isnan(cont_data)]
        
        if len(exp_data) > 0 and len(cont_data) > 0:
            ax.hist(exp_data, alpha=0.7, label='Experimental', bins=20, density=True)
            ax.hist(cont_data, alpha=0.7, label='Control', bins=20, density=True)
            ax.axvline(np.mean(exp_data), color='blue', linestyle='--', alpha=0.8)
            ax.axvline(np.mean(cont_data), color='orange', linestyle='--', alpha=0.8)
            ax.legend()
            ax.set_xlabel('Correlation')
            ax.set_ylabel('Density')
            ax.set_title(title)
    
    def adapt_parameter_grid(self, evaluated_results):
        """Adapt the parameter grid based on evaluation results."""
        print(f"\n=== Adapting parameter grid based on {len(evaluated_results)} results ===")
        
        if not evaluated_results:
            print("No results to adapt from, keeping current parameter ranges")
            return
        
        # Analyze which parameters work well
        good_params = []
        for result in evaluated_results:
            if 'evaluation' in result:
                score = self._compute_result_score(result['evaluation'])
                good_params.append((result['params'], score))
        
        # Sort by score
        good_params.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 20% of parameters
        top_count = max(1, len(good_params) // 5)
        top_params = good_params[:top_count]
        
        print(f"Top {top_count} parameter combinations:")
        for i, (params, score) in enumerate(top_params[:5]):  # Show top 5
            print(f"  {i+1}. Score: {score:.3f}, Params: {params}")
        
        # Adapt parameter ranges based on good results
        self._adapt_ranges_from_good_params(top_params)
        
        # Save adaptation results
        self._save_adaptation_results(top_params)
    
    def _prune_parameter_grid(self):
        """Prune parameter grid to keep size manageable (< 128 combinations)."""
        print(f"\n✂️  Pruning parameter grid to stay under 128 combinations...")
        
        current_size = self._calculate_grid_size()
        print(f"   Current grid size: {current_size} combinations")
        
        if current_size <= 128:
            print(f"   ✅ Grid size already manageable ({current_size} <= 128)")
            return
        
        print(f"   🔧 Grid too large ({current_size} > 128), pruning...")
        
        # Strategy: Keep the most promising parameter ranges and reduce others
        # Priority order: alphas, lrs, ma_pcs, mb_pcs, inh_lrs (most important)
        priority_params = ['alphas', 'lrs', 'ma_pcs', 'mb_pcs', 'inh_lrs']
        
        # First, try to reduce non-priority parameters
        for param in ['W_ip_a', 'W_pi_a', 'W_ip_b', 'W_pi_b', 'tau_a', 'betas', 'tau_inh']:
            if param in self.param_ranges and len(self.param_ranges[param]) > 1:
                # Reduce to single best value
                self.param_ranges[param] = [self.param_ranges[param][0]]
                new_size = self._calculate_grid_size()
                print(f"   Reduced {param} to single value: {new_size} combinations")
                
                if new_size <= 128:
                    print(f"   ✅ Grid size now manageable ({new_size} <= 128)")
                    return
        
        # If still too large, reduce priority parameters
        for param in priority_params:
            if param in self.param_ranges and len(self.param_ranges[param]) > 2:
                current_values = self.param_ranges[param]
                # Keep best performing values and reduce to 2-3 values
                if len(current_values) > 3:
                    # Keep first, middle, and last values (edge + center)
                    if len(current_values) % 2 == 0:
                        mid = len(current_values) // 2
                        self.param_ranges[param] = [current_values[0], current_values[mid], current_values[-1]]
                    else:
                        mid = len(current_values) // 2
                        self.param_ranges[param] = [current_values[0], current_values[mid], current_values[-1]]
                    
                    new_size = self._calculate_grid_size()
                    print(f"   Reduced {param} from {len(current_values)} to {len(self.param_ranges[param])} values: {new_size} combinations")
                    
                    if new_size <= 128:
                        print(f"   ✅ Grid size now manageable ({new_size} <= 128)")
                        return
        
        # Final check - if still too large, force reduction
        final_size = self._calculate_grid_size()
        if final_size > 128:
            print(f"   ⚠️  Grid still too large ({final_size} > 128), forcing reduction...")
            # Keep only 2 values for each parameter to ensure manageable size
            for param in priority_params:
                if param in self.param_ranges and len(self.param_ranges[param]) > 2:
                    current_values = self.param_ranges[param]
                    self.param_ranges[param] = [current_values[0], current_values[-1]]  # Keep edges
            
            final_size = self._calculate_grid_size()
            print(f"   🔧 Forced reduction complete: {final_size} combinations")
    
    def _ensure_new_combinations(self):
        """Ensure we have new parameter combinations for the next iteration."""
        print(f"\n🆕 Ensuring new parameter combinations for next iteration...")
        
        # Track which parameter combinations we've already explored
        if not hasattr(self, '_explored_combinations'):
            self._explored_combinations = set()
        
        # Generate current combinations
        from itertools import product
        current_combinations = list(product(
            self.param_ranges['alphas'],
            self.param_ranges['lrs'],
            self.param_ranges['ma_pcs'],
            self.param_ranges['mb_pcs'],
            self.param_ranges['W_pi_a'],
            self.param_ranges['W_ip_a'],
            self.param_ranges['W_ip_b'],
            self.param_ranges['W_pi_b'],
            self.param_ranges['tau_a'],
            self.param_ranges['inh_lrs'],
            self.param_ranges['betas'],
            self.param_ranges['tau_inh']
        ))
        
        # Check how many are new
        new_combinations = [combo for combo in current_combinations if combo not in self._explored_combinations]
        explored_count = len(self._explored_combinations)
        new_count = len(new_combinations)
        total_count = len(current_combinations)
        
        print(f"   📊 Combination analysis:")
        print(f"      Previously explored: {explored_count}")
        print(f"      New combinations: {new_count}")
        print(f"      Total combinations: {total_count}")
        
        if new_count == 0:
            print(f"   ⚠️  No new combinations! Expanding parameter space...")
            self._expand_parameter_space()
        elif new_count < total_count * 0.3:  # Less than 30% new
            print(f"   ⚠️  Too few new combinations ({new_count}/{total_count}), expanding...")
            self._expand_parameter_space()
        else:
            print(f"   ✅ Sufficient new combinations ({new_count}/{total_count})")
        
        # Update explored combinations
        self._explored_combinations.update(current_combinations)
    
    def _expand_parameter_space(self):
        """Expand parameter space to ensure new combinations."""
        print(f"   🔍 Expanding parameter space...")
        
        # Strategy: Add new values around the best performing ranges
        for param in ['alphas', 'lrs', 'ma_pcs', 'mb_pcs', 'inh_lrs']:
            if param in self.param_ranges:
                current_values = self.param_ranges[param]
                if len(current_values) >= 2:
                    # Add values between existing ones
                    new_values = []
                    for i in range(len(current_values) - 1):
                        val1, val2 = current_values[i], current_values[i + 1]
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            mid_val = (val1 + val2) / 2
                            new_values.append(mid_val)
                    
                    # Add new values to the range
                    self.param_ranges[param].extend(new_values)
                    self.param_ranges[param].sort()
                    
                    print(f"      Added {len(new_values)} new values to {param}")
        
        # Ensure we don't exceed 128 combinations after expansion
        new_size = self._calculate_grid_size()
        if new_size > 128:
            print(f"   ⚠️  Expansion made grid too large ({new_size} > 128), pruning...")
            self._prune_parameter_grid()
    
    def _display_grid_evolution(self, iteration):
        """Display the evolution of grid size and parameter ranges."""
        if not hasattr(self, '_grid_history'):
            self._grid_history = []
        
        current_size = self._calculate_grid_size()
        grid_info = {
            'iteration': iteration,
            'grid_size': current_size,
            'param_ranges': {k: len(v) for k, v in self.param_ranges.items()}
        }
        self._grid_history.append(grid_info)
        
        print(f"\n📊 Grid Evolution (Iteration {iteration}):")
        print(f"   Grid size: {current_size} combinations")
        print(f"   Parameter ranges:")
        for param, count in grid_info['param_ranges'].items():
            print(f"     {param}: {count} values")
        
        if len(self._grid_history) > 1:
            prev_size = self._grid_history[-2]['grid_size']
            change = current_size - prev_size
            if change > 0:
                print(f"   📈 Grid size increased by {change} combinations")
            elif change < 0:
                print(f"   📉 Grid size decreased by {abs(change)} combinations")
            else:
                print(f"   ➡️  Grid size unchanged")
    
    def _compute_result_score(self, evaluation):
        """Compute a score for a result based on how close it is to target criteria."""
        try:
            score = 0.0
            
            # F1 vs F2 score
            f1f2 = evaluation['F1_vs_F2']
            f1f2_score = 0
            if (0.8 <= f1f2['spatial']['exp_mean'] <= 1.0 and 
                0.8 <= f1f2['spatial']['cont_mean'] <= 1.0):
                f1f2_score += 1
            if f1f2['spatial']['wasserstein'] < 0.1:
                f1f2_score += 1
            score += f1f2_score / 2
            
            # F2 vs N1 score
            f2n1 = evaluation['F2_vs_N1']
            f2n1_score = 0
            if 0.7 <= f2n1['spatial']['exp_mean'] <= 1.0:
                f2n1_score += 1
            if -0.05 <= f2n1['spatial']['cont_mean'] <= 0.05:
                f2n1_score += 1
            if f2n1['spatial']['wasserstein'] > 0.7:
                f2n1_score += 1
            score += f2n1_score / 3
            
            # N1 vs N2 score
            n1n2 = evaluation['N1_vs_N2']
            n1n2_score = 0
            if (0.7 <= n1n2['spatial']['exp_mean'] <= 1.0 and 
                0.7 <= n1n2['spatial']['cont_mean'] <= 1.0):
                n1n2_score += 1
            if n1n2['spatial']['wasserstein'] < 0.2:
                n1n2_score += 1
            score += n1n2_score / 2
            
            return score
            
        except Exception as e:
            print(f"Error computing score: {e}")
            return 0.0
    
    def _adapt_ranges_from_good_params(self, top_params):
        """Adapt parameter ranges based on good parameter combinations."""
        if not top_params:
            return
        
        # Extract parameter values from top results
        param_values = {name: [] for name in self.param_ranges.keys()}
        for params, _ in top_params:
            for i, (name, _) in enumerate(self.param_ranges.items()):
                param_values[name].append(params[i])
        
        # Analyze if edge parameters are performing better
        edge_performance = self._analyze_edge_vs_center_performance(top_params)
        
        # Adapt ranges based on performance analysis
        for name, values in param_values.items():
            if len(values) > 0:
                current_range = self.param_ranges[name]
                if len(current_range) > 1:  # Only adapt if there are multiple values
                    
                    if edge_performance.get(name, False):
                        # Edge parameters are better - ENLARGE the grid
                        self._enlarge_parameter_range(name, current_range, values)
                    else:
                        # Center parameters are better - FOCUS the grid
                        self._focus_parameter_range(name, current_range, values)
        
        # After adaptation, ensure grid size stays manageable
        self._prune_parameter_grid()
        
        # Ensure we have new parameter combinations for next iteration
        self._ensure_new_combinations()
    
    def _analyze_edge_vs_center_performance(self, top_params):
        """Analyze whether edge or center parameters perform better."""
        edge_performance = {}
        
        for name in ['alphas', 'lrs', 'ma_pcs', 'mb_pcs', 'inh_lrs']:
            if name not in self.param_ranges or len(self.param_ranges[name]) <= 1:
                continue
                
            current_range = self.param_ranges[name]
            edge_values = [current_range[0], current_range[-1]]  # First and last values
            center_values = current_range[1:-1] if len(current_range) > 2 else current_range
            
            # Count how many top results come from edge vs center
            edge_count = 0
            center_count = 0
            
            for params, score in top_params:
                param_idx = list(self.param_ranges.keys()).index(name)
                param_value = params[param_idx]
                
                if param_value in edge_values:
                    edge_count += 1
                elif param_value in center_values:
                    center_count += 1
            
            # If edge parameters dominate, mark for enlargement
            edge_performance[name] = edge_count > center_count and edge_count >= 2
        
        return edge_performance
    
    def _enlarge_parameter_range(self, name, current_range, good_values):
        """Enlarge parameter range when edge parameters perform better."""
        if len(current_range) < 2:
            return
            
        min_val, max_val = min(current_range), max(current_range)
        range_size = max_val - min_val
        
        # Enlarge the range significantly
        enlarged_min = max(0, min_val - range_size * 0.5)
        enlarged_max = max_val + range_size * 0.5
        
        # Create new grid with more points, extending beyond current range
        new_range = []
        
        # Add points extending beyond current range
        if enlarged_min < min_val:
            new_range.extend([enlarged_min, enlarged_min + (min_val - enlarged_min) * 0.5])
        
        # Keep current range points
        new_range.extend(current_range)
        
        # Add points extending beyond current range
        if enlarged_max > max_val:
            new_range.extend([max_val + (enlarged_max - max_val) * 0.5, enlarged_max])
        
        # Ensure we don't exceed reasonable bounds
        if name == 'alphas':
            new_range = [max(0.01, v) for v in new_range if v > 0]
        elif name == 'lrs':
            new_range = [max(0.1, v) for v in new_range if v > 0]
        elif name == 'ma_pcs':
            new_range = [max(50, v) for v in new_range if v > 0]
        elif name == 'mb_pcs':
            new_range = [max(10, v) for v in new_range if v > 0]
        elif name == 'inh_lrs':
            new_range = [max(1, v) for v in new_range if v > 0]
        
        self.param_ranges[name] = sorted(list(set(new_range)))
        print(f"🔍 Enlarged {name}: {len(current_range)} -> {len(self.param_ranges[name])} values (edge parameters performing better)")
    
    def _focus_parameter_range(self, name, current_range, good_values):
        """Focus parameter range when center parameters perform better."""
        if len(current_range) < 2:
            return
            
        # Find the range that contains most good values
        min_val, max_val = min(good_values), max(good_values)
        range_size = max_val - min_val
        
        # Focus range around promising region
        focused_min = max(0, min_val - range_size * 0.2)
        focused_max = max_val + range_size * 0.2
        
        # Create new grid with more points in the promising region
        new_range = []
        
        # Add points around the promising region
        for i in range(5):
            new_range.append(focused_min + i * (focused_max - focused_min) / 4)
        
        # Add some points outside to maintain exploration
        if focused_min > 0:
            new_range.insert(0, focused_min * 0.5)
        new_range.append(focused_max * 1.5)
        
        # Ensure we don't exceed reasonable bounds
        if name == 'alphas':
            new_range = [max(0.01, v) for v in new_range if v > 0]
        elif name == 'lrs':
            new_range = [max(0.1, v) for v in new_range if v > 0]
        elif name == 'ma_pcs':
            new_range = [max(50, v) for v in new_range if v > 0]
        elif name == 'mb_pcs':
            new_range = [max(10, v) for v in new_range if v > 0]
        elif name == 'inh_lrs':
            new_range = [max(1, v) for v in new_range if v > 0]
        
        self.param_ranges[name] = sorted(list(set(new_range)))
        print(f"🎯 Focused {name}: {len(current_range)} -> {len(self.param_ranges[name])} values (center parameters performing better)")
    
    def _save_adaptation_results(self, top_params):
        """Save adaptation results to file."""
        try:
            adaptation_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'top_params': [(params.tolist() if hasattr(params, 'tolist') else params, score) 
                               for params, score in top_params],
                'new_param_ranges': self.param_ranges
            }
            
            adaptation_file = os.path.join(self.results_dir, 'adaptation_history.json')
            
            # Load existing history if it exists
            if os.path.exists(adaptation_file):
                with open(adaptation_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(adaptation_data)
            
            with open(adaptation_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"Error saving adaptation results: {e}")
    
    def run_adaptive_search(self):
        """Run the complete adaptive parameter search."""
        print("=== Starting Adaptive Parameter Search ===")
        print(f"Target criteria: {self.target_criteria}")
        print(f"Initial parameter ranges: {self.param_ranges}")
        print(f"Grid size: {self._calculate_grid_size()} combinations")
        print("\n🔄 Adaptation Strategy:")
        print("   - If edge parameters perform better → ENLARGE grid")
        print("   - If center parameters perform better → FOCUS grid")
        print("   - Always maintain exploration vs exploitation balance")
        
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # For iteration 1, use initial ranges. For subsequent iterations, adapt first
            if iteration > 0:
                # Adapt parameter grid based on previous iteration results
                # Get results from the previous iteration (assuming same number of combinations)
                prev_iter_results = self.all_results[-108:]  # 108 is our grid size
                self.adapt_parameter_grid(prev_iter_results)
                print(f"🔄 Adapted parameter ranges: {self.param_ranges}")
            
            # Generate parameter combinations using current (possibly adapted) ranges
            param_combinations = list(product(
                self.param_ranges['alphas'],
                self.param_ranges['lrs'],
                self.param_ranges['ma_pcs'],
                self.param_ranges['mb_pcs'],
                self.param_ranges['W_pi_a'],
                self.param_ranges['W_ip_a'],
                self.param_ranges['W_pi_b'],
                self.param_ranges['W_ip_b'],
                self.param_ranges['tau_a'],
                self.param_ranges['inh_lrs'],
                self.param_ranges['betas'],
                self.param_ranges['tau_inh']
            ))
            
            print(f"Generated {len(param_combinations)} parameter combinations")
            print(f"Grid size: {self._calculate_grid_size()} combinations")
            
            # Display grid size evolution
            self._display_grid_evolution(iteration + 1)
            
            # Run simulations
            try:
                results, iter_dir = self.run_simulation_batch(param_combinations, iteration + 1)
                print(f"✅ Simulations completed: {len(results)} successful")
                
                # Evaluate results
                evaluated_results = self.evaluate_results(results, iter_dir)
            
            # Store results
            self.all_results.extend(evaluated_results)
            
            # Check if we found very good results
            if self.best_results:
                print(f"\n🎉 Found {len(self.best_results)} very good results!")
                print("These results meet all the target criteria.")
                
                # Save best results
                best_results_file = os.path.join(self.results_dir, 'best_results.json')
                best_data = []
                for result in self.best_results:
                    best_data.append({
                        'params': result['params'].tolist() if hasattr(result['params'], 'tolist') else result['params'],
                        'param_name': result['param_name'],
                        'evaluation': result['evaluation']
                    })
                
                with open(best_results_file, 'w') as f:
                    json.dump(best_data, f, indent=2)
                
                print(f"Best results saved to: {best_results_file}")
                
                # Auto-continue for screen session
                print(f"🔄 Continuing to next iteration automatically...")
            
            # Save iteration summary
            self._save_iteration_summary(iteration + 1, evaluated_results, iter_dir)
                
                print(f"✅ Iteration {iteration + 1} completed successfully!")
                
            except Exception as e:
                print(f"❌ Error in iteration {iteration + 1}: {e}")
                import traceback
                traceback.print_exc()
                
                # Save error information
                error_file = os.path.join(self.results_dir, f'iteration_{iteration + 1}_error.log')
                with open(error_file, 'w') as f:
                    f.write(f"Error in iteration {iteration + 1}: {e}\n")
                    f.write(traceback.format_exc())
                
                print(f"Error details saved to: {error_file}")
                
                # Try to continue with next iteration
                continue
        
        print(f"\n{'='*60}")
        print("ADAPTIVE SEARCH COMPLETED")
        print(f"{'='*60}")
        print(f"Total results: {len(self.all_results)}")
        print(f"Very good results: {len(self.best_results)}")
        
        if self.best_results:
            print("\n🎯 Very good results found:")
            for i, result in enumerate(self.best_results):
                print(f"  {i+1}. {result['param_name']}")
                print(f"     Score: {self._compute_result_score(result['evaluation']):.3f}")
        
        # Save final summary
        self._save_final_summary()
    
    def _load_existing_results(self):
        """Load existing results if available."""
        try:
            best_results_file = os.path.join(self.results_dir, 'best_results.json')
            if os.path.exists(best_results_file):
                with open(best_results_file, 'r') as f:
                    best_data = json.load(f)
                self.best_results = best_data
                print(f"✅ Loaded {len(self.best_results)} existing best results")
            
            # Try to load adaptation history
            adaptation_file = os.path.join(self.results_dir, 'adaptation_history.json')
            if os.path.exists(adaptation_file):
                with open(adaptation_file, 'r') as f:
                    history = json.load(f)
                if 'adapted_param_ranges' in history:
                    self.param_ranges = history['adapted_param_ranges']
                    print(f"✅ Loaded adapted parameter ranges from previous run")
                    
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}")
    
    def _save_all_results(self):
        """Save all results to disk."""
        try:
            results_file = os.path.join(self.results_dir, 'all_results.json')
            results_data = []
            for result in self.all_results:
                results_data.append({
                    'params': result['params'],
                    'param_name': result['param_name'],
                    'sim_dir': result['sim_dir'],
                    'evaluation': result.get('evaluation', None)
                })
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Could not save all results: {e}")
    
    def _save_adapted_ranges(self, iteration):
        """Save adapted parameter ranges."""
        try:
            adaptation_file = os.path.join(self.results_dir, 'adaptation_history.json')
            history = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'iteration': iteration,
                'total_results': len(self.all_results),
                'very_good_results': len(self.best_results),
                'adapted_param_ranges': self.param_ranges
            }
            
            with open(adaptation_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            print(f"✅ Adapted parameter ranges saved for iteration {iteration}")
            
        except Exception as e:
            print(f"⚠️  Could not save adapted ranges: {e}")
    
    def _save_iteration_summary(self, iteration, evaluated_results, iter_dir):
        """Save summary of current iteration."""
        try:
            summary = {
                'iteration': iteration,
                'total_results': len(evaluated_results),
                'very_good_results': len([r for r in evaluated_results if 'evaluation' in r and self._is_very_good_result(r['evaluation'])]),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'param_ranges': self.param_ranges
            }
            
            summary_file = os.path.join(iter_dir, 'iteration_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            print(f"Error saving iteration summary: {e}")
    
    def _save_final_summary(self):
        """Save final summary of the search."""
        try:
            final_summary = {
                'total_iterations': len(self.all_results),
                'total_results': len(self.all_results),
                'very_good_results': len(self.best_results),
                'final_param_ranges': self.param_ranges,
                'best_results': [r['param_name'] for r in self.best_results],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            final_file = os.path.join(self.results_dir, 'final_summary.json')
            with open(final_file, 'w') as f:
                json.dump(final_summary, f, indent=2)
                
            print(f"\nFinal summary saved to: {final_file}")
            
        except Exception as e:
            print(f"Error saving final summary: {e}")
    
    def _calculate_grid_size(self):
        """Calculate the total number of parameter combinations."""
        try:
            from itertools import product
            combinations = list(product(
                self.param_ranges['alphas'],
                self.param_ranges['lrs'],
                self.param_ranges['ma_pcs'],
                self.param_ranges['mb_pcs'],
                self.param_ranges['W_pi_a'],
                self.param_ranges['W_ip_a'],
                self.param_ranges['W_ip_b'],
                self.param_ranges['W_pi_b'],
                self.param_ranges['tau_a'],
                self.param_ranges['inh_lrs'],
                self.param_ranges['betas'],
                self.param_ranges['tau_inh']
            ))
            return len(combinations)
        except Exception as e:
            return "Error calculating"


if __name__ == "__main__":
    # Create and run the adaptive search
    searcher = AdaptiveParameterSearch(max_iterations=10)
    searcher.run_adaptive_search() 