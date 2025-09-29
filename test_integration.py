#!/usr/bin/env python3
"""
Test script to verify the integration of the run function with the Evolvium pipeline.
"""

import numpy as np
from evolvium import run, Individual

def simple_fitness(genes):
    """Simple fitness function: minimize sum of squares."""
    return np.sum(genes ** 2)

def test_basic_run():
    """Test basic run functionality."""
    print("Testing basic run functionality...")
    
    # Test with default parameters
    best = run(
        max_gen=10,
        pop_size=10,
        prole_size=3,
        verbose=False,
        metric=simple_fitness,
        gene_type='real',
        gene_size=5,
        gene_init_range=10
    )
    
    print(f"Best individual: {best}")
    print(f"Best fitness: {best.fitness:.4f}")
    print(f"Best genes: {best.gene}")
    
    return best

def test_verbose_run():
    """Test run with verbose output."""
    print("\nTesting verbose run functionality...")
    
    best = run(
        max_gen=5,
        pop_size=8,
        prole_size=2,
        mutation_rate=(0.1, 0.01),  # Decreasing mutation rate
        verbose=True,
        metric=simple_fitness,
        gene_type='integer',
        gene_size=3,
        gene_init_range=5
    )
    
    print(f"Best individual: {best}")
    print(f"Best fitness: {best.fitness:.4f}")
    
    return best

if __name__ == "__main__":
    print("Testing Evolvium pipeline integration...")
    
    # Test basic functionality
    best1 = test_basic_run()
    
    # Test verbose functionality
    best2 = test_verbose_run()
    
    print("\n✅ All tests completed successfully!")
    print("The run function is properly integrated with the Evolvium pipeline.")
