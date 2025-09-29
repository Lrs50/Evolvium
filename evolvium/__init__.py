"""
Evolvium: A Genetic Algorithm Framework

This package provides a comprehensive genetic algorithm implementation with caching,
customizable operators, and flexible individual representations.

Main Components:
- Individual: Represents a candidate solution with genetic operations
- run: Main genetic algorithm execution function
- Cache: LRU cache for fitness evaluations
- Utility functions for selection and mutation rate transitions
"""

from evolvium.algorithms.genetic.algorithm import run
from evolvium.algorithms.genetic.individual import Individual
from evolvium.algorithms.genetic.base import default_metric, default_repr
from evolvium.algorithms.genetic.operators import roulette, slow_transition
from evolvium.utils.cache import Cache

__version__ = "0.1.0"
__author__ = "Lucas Reis"

__all__ = [
    "run",
    "Individual", 
    "default_metric",
    "default_repr",
    "roulette",
    "slow_transition",
    "Cache"
]
