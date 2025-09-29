"""
Genetic algorithm module for Evolvium.

This module contains the core genetic algorithm implementation including:
- Individual class for representing candidate solutions
- Main run function for executing the genetic algorithm
- Base functions for default metrics and representations
- Genetic operators for mutation and crossover
"""

from evolvium.algorithms.genetic.algorithm import run
from evolvium.algorithms.genetic.individual import Individual
from evolvium.algorithms.genetic.base import default_metric, default_repr

__all__ = ["run", "Individual", "default_metric", "default_repr"]
