from evolvium.algorithms.genetic.base import default_metric, default_repr
from evolvium.utils.cache import Cache
import numpy as np

from typing import Any, List


class _IndividualSharedData:
    """
    Internal class to hold SHARED data for all Individual instances.

    Attributes:
        fitness_calls (int): Number of fitness calls for the current generation.
        total_fitness_calls (int): Total number of fitness calls.
        cache (Cache): Cache object for storing fitness values.
    """

    def __init__(self) -> None:
        self.fitness_calls = 0
        self.total_fitness_calls = 0
        self.cache = Cache(size=1000)


SHARED = _IndividualSharedData()

class Individual(object):
    def __init__(self, copy: bool = False, **kwargs: Any) -> None:
        """
        Initialize an Individual with customizable genetic properties.

        Args:
            copy (bool): If True, creates a copy of another individual. Defaults to False.
            **kwargs: Additional attributes for customization, with their defaults:
                - metric (Callable): Fitness evaluation function. Default: default_metric
                - repr (Callable): String representation function. Default: default_repr
                - gene_type (str): Type of genes ('binary', 'integer', 'real'). Default: 'integer'
                - gene_size (int): Number of genes in the chromosome. Default: 10
                - gene_upper_limit (int|float): Upper limit for gene values. Default: 10
                - init_method (str): Initialization method ('random', 'mix', 'zeros'). Default: 'random'
                - mutation_type (str): Mutation strategy. Default: 'random range'
                - mutation_range (tuple): Range for mutation values (low, high). Default: (-1, 1)
                - crossover_type (str): Crossover strategy (random mix, split, avg). Default: 'split'
                - chromosome_size (int): Size of each chromosome segment. Default: 1

        Returns:
            None
        """
        self.kwargs = kwargs
        self._gene = None
        self._fitness = None

        self.metric = kwargs.get("metric", default_metric)
        self.repr = kwargs.get("repr", default_repr)

        if not copy:
            self.generate()

    def generate(self) -> None:
        """
        Generate a new genome based on initialization parameters.

        Args:
            None

        Returns:
            None

        Notes:
            The genome's characteristics (type, size, and values) are determined by:
                - gene_type: 'binary', 'integer', or 'real'
                - gene_size: number of genes
                - init_method: initialization method
                - gene_init_range: range for gene values
        """
        kwargs = self.kwargs
        gene_type = kwargs.get("gene_type", "integer")
        gene_size = kwargs.get("gene_size", 10)
        init_method = kwargs.get("init_method", "random")
        gene_init_range = kwargs.get("gene_init_range", 10)

        if init_method == "mix":
            init_method = "random" if np.random.rand() < 0.5 else "zeros"

        if init_method == "random":
            if gene_type == "binary":
                self._gene = np.random.randint(2, size=gene_size)
            elif gene_type == "integer":
                self._gene = np.random.randint(gene_init_range, size=gene_size)
            elif gene_type == "real":
                self._gene = np.random.rand(gene_size) * gene_init_range
            else:
                raise ValueError(f"{gene_type} is not a valid gene type")
        elif init_method == "zeros":
            self._gene = np.zeros(gene_size)
        else:
            raise ValueError(f"{init_method} is not a valid initialization method")

    @property
    def fitness(self) -> float:
        """
        Evaluate and return the fitness of the individual.

        Uses the metric function and a caching mechanism to avoid redundant calculations.
        Increments global counters for fitness evaluations.

        Args:
            None

        Returns:
            float: The fitness value of the individual.
        """
        global SHARED
        if self._fitness is None:
            SHARED.fitness_calls += 1
            cached_fitness = SHARED.cache[str(self._gene)]
            if cached_fitness:
                self._fitness = cached_fitness
            else:
                SHARED.fitness_calls += 1
                self._fitness = self.metric(self._gene)
                SHARED.cache[str(self._gene)] = self._fitness
        return self._fitness

    @property
    def gene(self) -> Any:
        """
        Get the genome representation of the individual.

        Returns:
            Any: The genome array, or None if not available.
        """
        return self._gene

    def mutate(self, prob: float = 1.0) -> "Individual":
        """
        Mutate the current genome based on mutation parameters.

        Args:
            prob (float): Probability of mutation (0 to 1). Defaults to 1.0.

        Returns:
            Individual: The mutated individual (self).
        """
        if np.random.rand() < prob:
            kwargs = self.kwargs
            mutation_type = kwargs.get("mutation_type", "random range")
            gene_type = kwargs.get("gene_type", "integer")
            gene_upper_limit = kwargs.get("gene_upper_limit", 10)
            mutation_range = kwargs.get("mutation_range", (-1, 1))

            if mutation_type == "random range":
                low, high = mutation_range
                i = np.random.randint(len(self._gene))
                if gene_type == "real":
                    if gene_upper_limit > 0:
                        self._gene[i] += (
                            np.random.rand() * (high - low) + low
                        ) % gene_upper_limit
                    else:
                        self._gene[i] += np.random.rand() * (high - low) + low
                else:
                    self._gene[i] += np.random.randint(low, high + 1)
                    if gene_upper_limit > 0:
                        self._gene[i] %= gene_upper_limit + 1
            else:
                raise ValueError(f"{mutation_type} is not a valid mutation type")

            self._fitness = None
        return self

    def copy(self) -> "Individual":
        """
        Create and return a copy of the current individual, preserving its genome and fitness.

        Args:
            None

        Returns:
            Individual: A new Individual object with the same genome and fitness.
        """
        c = Individual(copy=True, **self.kwargs)
        c._gene = self._gene.copy()
        c._fitness = self._fitness
        return c

    def crossover(self, dad: "Individual") -> "Individual":
        """
        Create a new genome by combining the genes of the current individual and a "dad".

        Args:
            dad (Individual): Another individual whose genes are used for crossover.

        Returns:
            Individual: A new child individual with combined genes.

        Crossover Strategies:
            - "random mix": Randomly selects gene segments from each parent.
            - "split": Divides genes at a random cut point.
            - "avg": Computes the average of corresponding genes (real only).
        """
        kwargs = self.kwargs
        crossover_type = kwargs.get("crossover_type", "split")
        chromosome_size = kwargs.get("chromosome_size", 1)
        gene_type = kwargs.get("gene_type", "integer")
        n_chromosome = len(self._gene) // chromosome_size

        new_gene = []
        if crossover_type == "random mix":
            choice = np.random.randint(2, size=n_chromosome)
            for i, c in enumerate(choice):
                if c:
                    new_gene.extend(
                        self._gene[i * chromosome_size : (i + 1) * chromosome_size]
                    )
                else:
                    new_gene.extend(
                        dad._gene[i * chromosome_size : (i + 1) * chromosome_size]
                    )
        elif crossover_type == "split":
            split_size = np.random.rand()
            cut_point = int(len(self._gene) * split_size)
            new_gene.extend(dad._gene[:cut_point])
            new_gene.extend(self._gene[cut_point:])
        elif crossover_type == "avg":
            if gene_type != "real":
                raise ValueError(
                    f"{gene_type} is not a valid gene type for {crossover_type} crossover!"
                )
            new_gene = [
                (dad._gene[i] + self._gene[i]) / 2 for i in range(len(self._gene))
            ]
        else:
            raise ValueError(f"{crossover_type} is not a valid crossover type")

        child = Individual(**self.kwargs)
        child._gene = new_gene
        return child

    def __repr__(self) -> str:
        """
        Return the string representation of the current individual using the provided representation function.

        Args:
            None

        Returns:
            str: String representation of the individual's genome.
        """
        return self.repr(self._gene)
