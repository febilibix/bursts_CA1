#!/usr/bin/env python3
"""
Iterative Adaptive Parameter Search for CA1 Hippocampal Model
Runs 10 iterations with parameter adaptation based on results.
"""

import os
import sys
import json
import time
import multiprocessing as mp
from itertools import product
import pickle
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('final_docs/')

from final_docs.simulations.simulate_2d import run_simulation
from final_docs.helpers import cor_act_maps_2d, SEED
from scipy.stats import wasserstein_distance

class IterativeAdaptiveSearch:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.iteration = 0
        self.adaptation_history = []
        self.best_results = []
        
        # Load initial parameters (adapted from recovery)
        self.param_ranges = self._load_initial_parameters()
        
        # Fixed parameters
        self.fixed_params = {'a': 0.3}
        
        print(f"🌱 Initialized with SEED: {SEED}")
        print(f"🎯 Target iterations: {self.max_iterations}")
    
    def _load_initial_parameters(self):
        """Load the adapted parameters from recovery."""
        adaptation_file = 'adaptive_search_results/adaptation_history.json'
        if os.path.exists(adaptation_file):
            with open(adaptation_file, 'r') as f:
                history = json.load(f)
            if 'adapted_param_ranges' in history:
                print("✅ Loaded adapted parameters from recovery:")
                for param, values in history['adapted_param_ranges'].items():
                    print(f"   {param}: {values}")
                return history['adapted_param_ranges']
        
        print("⚠️  No adapted parameters found, using defaults")
        return {
            'alphas': [0.1, 0.7, 1.5],
            'lrs': [5, 25],
            'ma_pcs': [180, 360, 750],
            'mb_pcs': [50, 100],
            'inh_lrs': [25, 125, 375],
            'W_ip_a': [1000],
            'W_pi_a': [500],
            'W_ip_b': [4000],
            'W_pi_b': [30],
            'tau_a': [1.0],
            'betas': [0.1],
            'tau_inh': [0.1]
        }
    
    def _calculate_grid_size(self):
        """Calculate the total number of parameter combinations."""
        total = 1
        for param, values in self.param_ranges.items():
            total *= len(values)
        return total
    
    def _prune_parameter_grid(self, target_size=128):
        """Prune parameter grid to keep size manageable."""
        print(f"\n✂️  Pruning parameter grid to stay under {target_size} combinations...")
        
        current_size = self._calculate_grid_size()
        print(f"   Current grid size: {current_size} combinations")
        
        if current_size <= target_size:
            print(f"   ✅ Grid size already manageable ({current_size} <= {target_size})")
            return
        
        print(f"   🔧 Grid too large ({current_size} > {target_size}), pruning...")
        
        # Priority parameters (most important)
        priority_params = ['alphas', 'lrs', 'ma_pcs', 'mb_pcs', 'inh_lrs']
        
        # First reduce non-priority parameters
        for param in ['W_ip_a', 'W_pi_a', 'W_ip_b', 'W_pi_b', 'tau_a', 'betas', 'tau_inh']:
            if param in self.param_ranges and len(self.param_ranges[param]) > 1:
                self.param_ranges[param] = [self.param_ranges[param][0]]
                new_size = self._calculate_grid_size()
                print(f"   Reduced {param} to single value: {new_size} combinations")
                
                if new_size <= target_size:
                    print(f"   ✅ Grid size now manageable ({new_size} <= {target_size})")
                    return
        
        # If still too large, reduce priority parameters
        for param in priority_params:
            if param in self.param_ranges and len(self.param_ranges[param]) > 2:
                current_values = self.param_ranges[param]
                if len(current_values) > 3:
                    # Keep first, middle, and last values
                    mid = len(current_values) // 2
                    self.param_ranges[param] = [current_values[0], current_values[mid], current_values[-1]]
                    
                    new_size = self._calculate_grid_size()
                    print(f"   Reduced {param} from {len(current_values)} to {len(self.param_ranges[param])} values: {new_size} combinations")
                    
                    if new_size <= target_size:
                        print(f"   ✅ Grid size now manageable ({new_size} <= {target_size})")
                        return
        
        # Final reduction if needed
        final_size = self._calculate_grid_size()
        if final_size > target_size:
            print(f"   ⚠️  Grid still too large ({final_size} > {target_size}), forcing reduction...")
            for param in priority_params:
                if param in self.param_ranges and len(self.param_ranges[param]) > 2:
                    current_values = self.param_ranges[param]
                    self.param_ranges[param] = [current_values[0], current_values[-1]]
            
            final_size = self._calculate_grid_size()
            print(f"   🔧 Forced reduction complete: {final_size} combinations")
    
    def _create_trajectory_data(self):
        """Create trajectory data files if they don't exist."""
        print("🔄 Creating trajectory data...")
        
        # Check if trajectory directory exists
        traj_dir = 'data/2d_test2'
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir, exist_ok=True)
        
        # Generate trajectories for each environment
        from final_docs.simulations.simulate_2d import simulate_2d_run, ENVIRONMENTS_RUNS
        
        seed = SEED
        for k in ENVIRONMENTS_RUNS.keys():
            traj_file = f'{traj_dir}/run_{k}.pkl'
            if not os.path.exists(traj_file):
                print(f"   Generating trajectory for {k}...")
                t_run, x_run = simulate_2d_run(100, 0.1, 0.01, 100, seed=seed)
                with open(traj_file, 'wb') as f:
                    pickle.dump((t_run, x_run), f)
                seed += 1
            else:
                print(f"   ✅ Trajectory for {k} already exists")
    
    def _run_single_simulation(self, params):
        """Run a single simulation with given parameters."""
        try:
            # Unpack parameters
            alpha, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh = params
            a = self.fixed_params['a']
            
            # Create unique directory name
            param_name = f'alpha_{alpha}_lr_{lr}_mapc_{ma_pc}_mbpc_{mb_pc}_lrinh_{inh_lr}'
            sim_dir = os.path.join('adaptive_search_results', 'simulations', param_name)
            os.makedirs(sim_dir, exist_ok=True)
            
            # Change to simulation directory and run simulation
            original_cwd = os.getcwd()
            os.chdir(sim_dir)
            
            try:
                run_simulation(alpha, a, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, 
                              W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh)
            finally:
                os.chdir(original_cwd)
            
            return {
                'params': params,
                'param_name': param_name,
                'sim_dir': sim_dir
            }
            
        except Exception as e:
            print(f"Simulation failed for params {params}: {e}")
            return None
    
    def _evaluate_results(self, results):
        """Evaluate simulation results using a flexible scoring system."""
        print(f"\n🔍 Evaluating {len(results)} simulation results...")
        
        scored_solutions = []
        
        for result in results:
            if result is None:
                continue
            
            # Load simulation results
            sim_dir = result['sim_dir']
            param_name = result['param_name']
            # NOTE: Due to double-nested directory structure in run_simulation,
            # the actual files are in sim_dir/param_name/ not sim_dir/
            actual_sim_dir = os.path.join(sim_dir, param_name)
            
            # Check if all required files exist
            required_files = []
            for condition in ['exp', 'cont']:
                for out in ['F1', 'F2', 'N1', 'F3', 'N2']:
                    required_files.append(f'{actual_sim_dir}/{condition}_{out}.pkl')
            
            if not all(os.path.exists(f) for f in required_files):
                continue
            
            try:
                # Load activation maps
                act_maps = {'exp': {}, 'cont': {}}
                for condition in ['exp', 'cont']:
                    for out in ['F1', 'F2', 'N1', 'F3', 'N2']:
                        with open(f'{actual_sim_dir}/{condition}_{out}.pkl', 'rb') as f:
                            act_map, _ = pickle.load(f)
                        act_maps[condition][out] = act_map
                
                # Calculate correlations
                corr_exp = cor_act_maps_2d(act_maps['exp']['F1'], act_maps['exp']['F2'])
                corr_cont = cor_act_maps_2d(act_maps['cont']['F1'], act_maps['cont']['F2'])
                
                corr_exp_n1n2 = cor_act_maps_2d(act_maps['exp']['N1'], act_maps['exp']['N2'])
                corr_cont_n1n2 = cor_act_maps_2d(act_maps['cont']['N1'], act_maps['cont']['N2'])
                
                corr_exp_f2n1 = cor_act_maps_2d(act_maps['exp']['F2'], act_maps['exp']['N1'])
                corr_cont_f2n1 = cor_act_maps_2d(act_maps['cont']['F2'], act_maps['cont']['N1'])
                
                # Calculate Wasserstein distances
                wass_f1f2 = wasserstein_distance(corr_exp, corr_cont)
                wass_n1n2 = wasserstein_distance(corr_exp_n1n2, corr_cont_n1n2)
                wass_f2n1 = wasserstein_distance(corr_exp_f2n1, corr_cont_f2n1)
                
                # Calculate individual scores (0 = perfect, higher = worse)
                f1f2_score = self._calculate_f1f2_score(np.mean(corr_exp), np.mean(corr_cont), wass_f1f2)
                n1n2_score = self._calculate_n1n2_score(np.mean(corr_exp_n1n2), np.mean(corr_cont_n1n2), wass_n1n2)
                f2n1_score = self._calculate_f2n1_score(np.mean(corr_exp_f2n1), np.mean(corr_cont_f2n1), wass_f2n1)
                
                # Overall score (lower is better)
                total_score = f1f2_score + n1n2_score + f2n1_score
                
                # Store solution with scores
                scored_solutions.append({
                    'param_name': param_name,
                    'params': result['params'],
                    'total_score': total_score,
                    'individual_scores': {
                        'f1f2': f1f2_score,
                        'n1n2': n1n2_score,
                        'f2n1': f2n1_score
                    },
                    'metrics': {
                        'f1f2_exp_mean': np.mean(corr_exp),
                        'f1f2_cont_mean': np.mean(corr_cont),
                        'f1f2_wass': wass_f1f2,
                        'n1n2_exp_mean': np.mean(corr_exp_n1n2),
                        'n1n2_cont_mean': np.mean(corr_cont_n1n2),
                        'n1n2_wass': wass_n1n2,
                        'f2n1_exp_mean': np.mean(corr_exp_f2n1),
                        'f2n1_cont_mean': np.mean(corr_cont_f2n1),
                        'f2n1_wass': wass_f2n1
                    }
                })
                
                # Print excellent solutions (very low scores)
                if total_score < 0.5:
                    print(f"🎉 EXCELLENT SOLUTION: {param_name} (score: {total_score:.3f})")
                    print(f"   F1 vs F2: exp={np.mean(corr_exp):.3f}, cont={np.mean(corr_cont):.3f}, wass={wass_f1f2:.3f} (score: {f1f2_score:.3f})")
                    print(f"   N1 vs N2: exp={np.mean(corr_exp_n1n2):.3f}, cont={np.mean(corr_cont_n1n2):.3f}, wass={wass_n1n2:.3f} (score: {n1n2_score:.3f})")
                    print(f"   F2 vs N1: exp={np.mean(corr_exp_f2n1):.3f}, cont={np.mean(corr_cont_f2n1):.3f}, wass={wass_f2n1:.3f} (score: {f2n1_score:.3f})")
                
            except Exception as e:
                print(f"Error evaluating {param_name}: {e}")
                continue
        
        # Sort by total score (best first)
        scored_solutions.sort(key=lambda x: x['total_score'])
        
        # Keep top solutions for adaptation (but save all for analysis)
        top_solutions = scored_solutions[:min(20, len(scored_solutions))]  # Top 20 for adaptation
        
        print(f"✅ Evaluated {len(scored_solutions)} solutions")
        print(f"📊 Score range: {scored_solutions[0]['total_score']:.3f} (best) to {scored_solutions[-1]['total_score']:.3f} (worst)")
        print(f"🎯 Using top {len(top_solutions)} solutions for adaptation")
        
        return top_solutions, scored_solutions
    
    def _calculate_f1f2_score(self, exp_mean, cont_mean, wass):
        """Calculate F1 vs F2 score (0 = perfect, higher = worse)."""
        score = 0.0
        
        # Mean correlation scores (target: exp [0.8, 1.0], cont [0.8, 1.0])
        if exp_mean < 0.8:
            score += (0.8 - exp_mean) * 2  # Penalty for being below 0.8
        elif exp_mean > 1.0:
            score += (exp_mean - 1.0) * 2  # Penalty for being above 1.0
        
        if cont_mean < 0.8:
            score += (0.8 - cont_mean) * 2
        elif cont_mean > 1.0:
            score += (cont_mean - 1.0) * 2
        
        # Wasserstein distance score (target: < 0.1)
        if wass >= 0.1:
            score += (wass - 0.1) * 5  # Strong penalty for high Wasserstein
        
        return score
    
    def _calculate_n1n2_score(self, exp_mean, cont_mean, wass):
        """Calculate N1 vs N2 score (0 = perfect, higher = worse)."""
        score = 0.0
        
        # Mean correlation scores (target: exp [0.7, 1.0], cont [0.7, 1.0])
        if exp_mean < 0.7:
            score += (0.7 - exp_mean) * 2
        elif exp_mean > 1.0:
            score += (exp_mean - 1.0) * 2
        
        if cont_mean < 0.7:
            score += (0.7 - cont_mean) * 2
        elif cont_mean > 1.0:
            score += (cont_mean - 1.0) * 2
        
        # Wasserstein distance score (target: < 0.2)
        if wass >= 0.2:
            score += (wass - 0.2) * 3  # Moderate penalty for high Wasserstein
        
        return score
    
    def _calculate_f2n1_score(self, exp_mean, cont_mean, wass):
        """Calculate F2 vs N1 score (0 = perfect, higher = worse)."""
        score = 0.0
        
        # Mean correlation scores (target: exp [0.7, 1.0], cont [-0.05, 0.05])
        if exp_mean < 0.7:
            score += (0.7 - exp_mean) * 2
        elif exp_mean > 1.0:
            score += (exp_mean - 1.0) * 2
        
        if cont_mean < -0.05:
            score += abs(cont_mean + 0.05) * 3  # Strong penalty for negative control
        elif cont_mean > 0.05:
            score += (cont_mean - 0.05) * 3  # Strong penalty for positive control
        
        # Wasserstein distance score (target: > 0.7)
        if wass <= 0.7:
            score += (0.7 - wass) * 4  # Strong penalty for low Wasserstein
        
        return score
    
    def _adapt_parameter_grid(self, good_solutions):
        """Adapt parameter grid based on good solutions."""
        if not good_solutions:
            print("⚠️  No good solutions to adapt from, keeping current ranges")
            return
        
        print(f"\n🔄 Adapting parameter grid based on {len(good_solutions)} good solutions...")
        
        # Analyze which parameters are most successful
        param_success = {}
        for solution in good_solutions:
            params = solution['params']
            for i, (param_name, value) in enumerate(self.param_ranges.items()):
                if param_name not in param_success:
                    param_success[param_name] = []
                param_success[param_name].append(params[i])
        
        # Adapt ranges based on successful values
        for param_name, successful_values in param_success.items():
            if len(successful_values) > 0:
                current_range = self.param_ranges[param_name]
                successful_values = list(set(successful_values))  # Remove duplicates
                
                if len(successful_values) == 1:
                    # Single successful value - narrow around it
                    value = successful_values[0]
                    if param_name in ['alphas', 'lrs', 'ma_pcs', 'mb_pcs', 'inh_lrs']:
                        # For important parameters, create a small range around the value
                        if value > 0:
                            self.param_ranges[param_name] = [value * 0.8, value, value * 1.2]
                        else:
                            self.param_ranges[param_name] = [value * 1.2, value, value * 0.8]
                    else:
                        # For less important parameters, keep single value
                        self.param_ranges[param_name] = [value]
                
                elif len(successful_values) > 1:
                    # Multiple successful values - expand range to include them
                    min_val = min(successful_values)
                    max_val = max(successful_values)
                    
                    # Add some padding around the range
                    if param_name in ['alphas', 'lrs', 'ma_pcs', 'mb_pcs', 'inh_lrs']:
                        padding = (max_val - min_val) * 0.2
                        self.param_ranges[param_name] = [
                            max(0, min_val - padding),
                            min_val,
                            max_val,
                            max_val + padding
                        ]
                    else:
                        self.param_ranges[param_name] = [min_val, max_val]
        
        # Ensure grid size is manageable
        self._prune_parameter_grid(target_size=128)
        
        print("✅ Parameter grid adapted")
        print(f"   New grid size: {self._calculate_grid_size()} combinations")
    
    def _save_iteration_results(self, all_solutions):
        """Save results from current iteration."""
        iteration_dir = f'adaptive_search_results/iteration_{self.iteration}'
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save iteration summary
        summary = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'param_ranges': self.param_ranges,
            'grid_size': self._calculate_grid_size(),
            'good_solutions_count': len(self.best_results),
            'total_solutions_evaluated': len(all_solutions)
        }
        
        with open(f'{iteration_dir}/iteration_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save all solutions with scores
        with open(f'{iteration_dir}/all_solutions_scored.json', 'w') as f:
            json.dump(all_solutions, f, indent=2)
        
        # Save good solutions separately
        if self.best_results:
            with open(f'{iteration_dir}/good_solutions.json', 'w') as f:
                json.dump(self.best_results, f, indent=2)
    
    def run(self):
        """Run the iterative adaptive search."""
        print("🚀 ITERATIVE ADAPTIVE PARAMETER SEARCH")
        print("=" * 60)
        
        # Create trajectory data once
        self._create_trajectory_data()
        
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            print(f"\n🔄 ITERATION {iteration}/{self.max_iterations}")
            print("=" * 40)
            
            # Ensure grid size is manageable
            self._prune_parameter_grid(target_size=128)
            
            # Generate parameter combinations
            param_combinations = list(product(
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
            
            print(f"🎯 Running {len(param_combinations)} parameter combinations")
            print(f"   Grid size: {self._calculate_grid_size()} combinations")
            
            # Run simulations
            print(f"🔄 Running simulations...")
            start_time = time.time()
            
            with mp.Pool(processes=48) as pool:
                results = list(pool.imap(self._run_single_simulation, param_combinations))
            
            sim_time = time.time() - start_time
            print(f"✅ Simulations completed in {sim_time:.1f} seconds")
            
            # Filter successful results
            successful_results = [r for r in results if r is not None]
            print(f"✅ Completed {len(successful_results)} out of {len(param_combinations)} simulations")
            
            # Evaluate results
            good_solutions, all_solutions = self._evaluate_results(successful_results)
            
            # Update best results
            self.best_results.extend(good_solutions)
            
            # Save iteration results
            self._save_iteration_results(all_solutions)
            
            # Adapt parameter grid for next iteration
            if iteration < self.max_iterations:
                self._adapt_parameter_grid(good_solutions)
                
                # Save adaptation history
                self.adaptation_history.append({
                    'iteration': iteration,
                    'param_ranges': self.param_ranges.copy(),
                    'grid_size': self._calculate_grid_size(),
                    'good_solutions_count': len(good_solutions)
                })
                
                with open('adaptive_search_results/adaptation_history.json', 'w') as f:
                    json.dump({
                        'adapted_param_ranges': self.param_ranges,
                        'adaptation_history': self.adaptation_history
                    }, f, indent=2)
            
            print(f"✅ Iteration {iteration} completed")
        
        # Final summary
        print(f"\n🎉 ADAPTIVE SEARCH COMPLETED!")
        print("=" * 60)
        print(f"Total iterations: {self.max_iterations}")
        print(f"Total good solutions found: {len(self.best_results)}")
        print(f"Final grid size: {self._calculate_grid_size()} combinations")
        
        # Save final results
        with open('adaptive_search_results/final_summary.json', 'w') as f:
            json.dump({
                'total_iterations': self.max_iterations,
                'total_good_solutions': len(self.best_results),
                'final_param_ranges': self.param_ranges,
                'final_grid_size': self._calculate_grid_size(),
                'all_good_solutions': self.best_results,
                'scoring_system': {
                    'description': 'Lower scores are better. Scores are calculated based on distance from target ranges.',
                    'f1f2_target': 'exp/cont mean cor [0.8, 1.0], wass < 0.1',
                    'n1n2_target': 'exp/cont mean cor [0.7, 1.0], wass < 0.2',
                    'f2n1_target': 'exp mean cor [0.7, 1.0], cont mean cor [-0.05, 0.05], wass > 0.7'
                }
            }, f, indent=2)
        
        print(f"📊 Final results saved to: adaptive_search_results/final_summary.json")

def main():
    """Main function to run the iterative adaptive search."""
    search = IterativeAdaptiveSearch(max_iterations=10)
    search.run()

if __name__ == "__main__":
    main() 