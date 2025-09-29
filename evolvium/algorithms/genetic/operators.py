from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from evolvium.algorithms.genetic.individual import Individual

def slow_transition(
    x: float = 0.01, y: float = 0.1, n: int = 1000, power: float = 3
) -> List[float]:
    """
    Generate a sequence transitioning from x to y in n steps, slowing as it approaches y.

    Args:
        x (float): Start value.
        y (float): End value.
        n (int): Number of steps in the sequence.
        power (float): Controls the slowness of the transition; higher values slow the approach to y.

    Returns:
        List[float]: List of n values transitioning from x to y.
    """
    t = np.linspace(0, 1, n)
    t_slow = t**power
    transition = x + (y - x) * t_slow
    return list(transition)


def roulette(
    fitness_list: List[float], pop: List["Individual"], n: int
) -> List["Individual"]:
    """
    Select individuals from a population using roulette wheel selection based on fitness.

    Args:
        fitness_list (List[float]): Fitness values for each individual in the population.
        pop (List[Individual]): List of individuals in the population.
        n (int): Number of individuals to select.

    Returns:
        List[Individual]: List of n selected individuals (copied) based on their fitness.
    """
    if n <= 0:
        return []

    fitness_arr = np.array(fitness_list, dtype=float)
    info = 1 / fitness_arr
    prob = info / np.sum(info)
    selected_indices = np.random.choice(len(pop), size=n, p=prob, replace=False)
    return [pop[i].copy() for i in selected_indices]
