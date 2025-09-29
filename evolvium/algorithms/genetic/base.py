from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolvium.algorithms.genetic.individual import Individual


def default_metric(candidate: "Individual") -> float:
    """
    Default metric function for evaluating a candidate.

    Args:
        candidate (Individual): The candidate solution to evaluate.

    Returns:
        float: The default fitness value (always 0.1).
    """
    return 0.1


def default_repr(candidate: "Individual") -> str:
    """
    Default string representation for a candidate.

    Args:
        candidate (Individual): The candidate solution to represent as a string.

    Returns:
        str: The string representation of the candidate.
    """
    return str(candidate)
