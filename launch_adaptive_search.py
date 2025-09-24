#!/usr/bin/env python3
"""
Launcher script for the adaptive parameter search with adapted parameters.
This will continue the search from where the recovery left off.
"""

import os
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adaptive_parameter_search import AdaptiveParameterSearch

def main():
    """Launch the adaptive search with adapted parameters."""
    print("🚀 LAUNCHING ADAPTIVE PARAMETER SEARCH WITH ADAPTED PARAMETERS")
    print("=" * 70)
    
    # Create a searcher instance
    searcher = AdaptiveParameterSearch(max_iterations=10)
    
    # Check if we have adapted parameters from recovery
    adaptation_file = os.path.join(searcher.results_dir, 'adaptation_history.json')
    if os.path.exists(adaptation_file):
        try:
            with open(adaptation_file, 'r') as f:
                history = json.load(f)
            
            if 'adapted_param_ranges' in history:
                print("✅ Found adapted parameter ranges from recovery:")
                for param, values in history['adapted_param_ranges'].items():
                    print(f"   {param}: {values}")
                
                # Use the adapted ranges
                searcher.param_ranges = history['adapted_param_ranges']
                print(f"\n🔄 Using adapted parameter ranges (grid size: {searcher._calculate_grid_size()} combinations)")
            else:
                print("⚠️  No adapted parameter ranges found, using defaults")
                
        except Exception as e:
            print(f"⚠️  Could not load adaptation history: {e}")
            print("Using default parameter ranges")
    else:
        print("⚠️  No adaptation history found, using default parameter ranges")
    
    print(f"\n🎯 Starting adaptive search with {searcher.max_iterations} iterations")
    print("=" * 70)
    
    # Start the adaptive search
    try:
        searcher.run_adaptive_search()
    except KeyboardInterrupt:
        print("\n⏹️  Search interrupted by user")
        print("Results have been saved and can be recovered later")
    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        print("Check error logs for details")

if __name__ == "__main__":
    main() 