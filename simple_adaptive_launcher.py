#!/usr/bin/env python3
"""
Simple adaptive launcher that uses the adapted parameters from recovery.
This will run a basic adaptive search with the enlarged parameter grid.
"""

import os
import sys
import json
import time
import multiprocessing as mp
from itertools import product
import pickle
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('final_docs/')

from final_docs.simulations.simulate_2d import run_simulation
from final_docs.helpers import cor_act_maps_2d
from scipy.stats import wasserstein_distance

def load_adapted_parameters():
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

def calculate_grid_size(param_ranges):
    """Calculate the total number of parameter combinations."""
    total = 1
    for param, values in param_ranges.items():
        total *= len(values)
    return total

def prune_parameters(param_ranges, target_size=128):
    """Prune parameters to keep grid size manageable."""
    print(f"\n✂️  Pruning parameters to stay under {target_size} combinations...")
    
    current_size = calculate_grid_size(param_ranges)
    print(f"   Current grid size: {current_size} combinations")
    
    if current_size <= target_size:
        print(f"   ✅ Grid size already manageable ({current_size} <= {target_size})")
        return param_ranges
    
    print(f"   🔧 Grid too large ({current_size} > {target_size}), pruning...")
    
    # Priority parameters (most important)
    priority_params = ['alphas', 'lrs', 'ma_pcs', 'mb_pcs', 'inh_lrs']
    
    # First reduce non-priority parameters
    for param in ['W_ip_a', 'W_pi_a', 'W_ip_b', 'W_pi_b', 'tau_a', 'betas', 'tau_inh']:
        if param in param_ranges and len(param_ranges[param]) > 1:
            param_ranges[param] = [param_ranges[param][0]]
            new_size = calculate_grid_size(param_ranges)
            print(f"   Reduced {param} to single value: {new_size} combinations")
            
            if new_size <= target_size:
                print(f"   ✅ Grid size now manageable ({new_size} <= {target_size})")
                return param_ranges
    
    # If still too large, reduce priority parameters
    for param in priority_params:
        if param in param_ranges and len(param_ranges[param]) > 2:
            current_values = param_ranges[param]
            if len(current_values) > 3:
                # Keep first, middle, and last values
                mid = len(current_values) // 2
                param_ranges[param] = [current_values[0], current_values[mid], current_values[-1]]
                
                new_size = calculate_grid_size(param_ranges)
                print(f"   Reduced {param} from {len(current_values)} to {len(param_ranges[param])} values: {new_size} combinations")
                
                if new_size <= target_size:
                    print(f"   ✅ Grid size now manageable ({new_size} <= {target_size})")
                    return param_ranges
    
    # Final reduction if needed
    final_size = calculate_grid_size(param_ranges)
    if final_size > target_size:
        print(f"   ⚠️  Grid still too large ({final_size} > {target_size}), forcing reduction...")
        for param in priority_params:
            if param in param_ranges and len(param_ranges[param]) > 2:
                current_values = param_ranges[param]
                param_ranges[param] = [current_values[0], current_values[-1]]
        
        final_size = calculate_grid_size(param_ranges)
        print(f"   🔧 Forced reduction complete: {final_size} combinations")
    
    return param_ranges

def run_single_simulation(params):
    """Run a single simulation with given parameters."""
    try:
        # Unpack parameters
        alpha, lr, ma_pc, mb_pc, W_pi_a, W_ip_a, W_pi_b, W_ip_b, tau_a, inh_lr, beta, tau_inh = params
        a = 0.3  # Fixed value
        
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

def main():
    """Main function to run the adaptive search."""
    print("🚀 SIMPLE ADAPTIVE PARAMETER SEARCH")
    print("=" * 50)
    
    # Load adapted parameters
    param_ranges = load_adapted_parameters()
    
    # Prune to manageable size
    param_ranges = prune_parameters(param_ranges, target_size=128)
    
    # Generate parameter combinations
    param_combinations = list(product(
        param_ranges['alphas'],
        param_ranges['lrs'],
        param_ranges['ma_pcs'],
        param_ranges['mb_pcs'],
        param_ranges['W_pi_a'],
        param_ranges['W_ip_a'],
        param_ranges['W_ip_b'],
        param_ranges['W_pi_b'],
        param_ranges['tau_a'],
        param_ranges['inh_lrs'],
        param_ranges['betas'],
        param_ranges['tau_inh']
    ))
    
    print(f"\n🎯 Starting search with {len(param_combinations)} parameter combinations")
    print(f"Grid size: {calculate_grid_size(param_ranges)} combinations")
    
    # Run simulations
    print(f"\n🔄 Running simulations...")
    with mp.Pool(processes=48) as pool:
        results = list(pool.imap(run_single_simulation, param_combinations))
    
    # Filter successful results
    successful_results = [r for r in results if r is not None]
    print(f"✅ Completed {len(successful_results)} out of {len(param_combinations)} simulations")
    
    # Save results
    results_file = 'adaptive_search_results/simple_search_results.json'
    with open(results_file, 'w') as f:
        json.dump([r['param_name'] for r in successful_results], f, indent=2)
    
    print(f"📊 Results saved to: {results_file}")
    print("🎉 Search completed!")

if __name__ == "__main__":
    main() 