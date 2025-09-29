# [Evolvium](https://test.pypi.org/project/evolvium/)

Evolvium is a high-performance genetic algorithm library for Python that provides a simple interface for evolutionary optimization problems. It uses natural selection principles to evolve populations of candidate solutions toward optimal results.

## Installation

```bash
pip install evolvium
```

## genetic.run

```python
genetic.run(max_gen=100, pop_size=30, prole_size=10, mutation_rate=1/30, 
             stop=0.5, verbose=False, **kwargs)
```

Run a genetic algorithm to optimize a fitness function over multiple generations.

### Parameters

**max_gen** : int, default=100
    Maximum number of generations to evolve. The algorithm terminates when this limit is reached or when the fitness threshold is met.

**pop_size** : int, default=30
    Size of the population in each generation. Larger populations provide more genetic diversity but require more computational resources.

**prole_size** : int, default=10
    Number of offspring generated through crossover in each generation. These offspring are added to the population during reproduction.

**mutation_rate** : float or tuple of float, default=1/30
    Probability of mutation for each individual. If a tuple (start, end) is provided, the mutation rate transitions linearly from start to end over generations.

**stop** : float, default=0.5
    Fitness threshold for early termination. The algorithm stops when the best individual's fitness is less than or equal to this value.

**verbose** : bool, default=False
    If True, displays detailed progress information including fitness statistics, population diversity, and generates evolution plots.

### Individual Configuration Parameters

**metric** : callable, default=default_metric
    Fitness function that takes an individual's genes and returns a fitness score. Lower scores indicate better fitness.

**gene_type** : {'integer', 'binary', 'real'}, default='integer'
    Type of genes in the chromosome:
    - 'integer': Integer values within specified bounds
    - 'binary': Binary values (0 or 1)
    - 'real': Floating-point values

**gene_size** : int, default=10
    Number of genes in each individual's chromosome.

**gene_upper_limit** : int or float, default=10
    Upper bound for gene values. Genes are initialized between 0 and this limit.

**init_method** : {'random', 'zeros', 'mix'}, default='random'
    Method for initializing gene values:
    - 'random': Random values within bounds
    - 'zeros': All genes set to zero
    - 'mix': Mixed initialization (used internally for crossover)

**mutation_type** : str, default='random range'
    Strategy for applying mutations. Currently supports 'random range' which adds random values within the mutation_range.

**mutation_range** : tuple of float, default=(-1, 1)
    Range (low, high) for mutation values. Mutations are random values sampled from this range.

**crossover_type** : {'split', 'random mix', 'avg'}, default='split'
    Crossover strategy for combining parent genes:
    - 'split': Split chromosomes at random points
    - 'random mix': Randomly select genes from each parent
    - 'avg': Average gene values from both parents

**chromosome_size** : int, default=1
    Size of each chromosome segment. Used when genes represent multi-dimensional vectors.

**cache_size** : int, default=1000
    Size of the LRU cache for storing fitness evaluations to avoid redundant calculations.

### Returns

**Individual**
    The best individual found across all generations, containing the optimized gene sequence and its fitness score.

### Notes

- Uses roulette wheel selection for parent selection based on fitness
- Implements elitism by preserving the best individuals across generations
- Automatic fitness caching improves performance for expensive fitness functions
- Population diversity is maintained through mutation and crossover operations

## Example

Here's a complete example that evolves a string to match "Hello, World!":

```python
from evolvium.algorithms import genetic

# Define the target string
target = "Hello, World!"

# Define fitness function (lower is better)
def string_fitness(genes):
    candidate = ''.join(chr(gene) for gene in genes)
    return sum(abs(ord(a) - ord(b)) for a, b in zip(candidate, target))

# Run the genetic algorithm
best = genetic.run(
    max_gen=1000,
    pop_size=100,
    prole_size=20,
    mutation_rate=0.1,
    stop=0,  # Stop when perfect match is found
    verbose=True,
    # Individual configuration
    metric=string_fitness,
    gene_type='integer',
    gene_size=len(target),
    gene_upper_limit=127,  # ASCII range
    mutation_range=(-10, 10)
)

# Display results
result_string = ''.join(chr(gene) for gene in best.gene)
print(f"Target:  {target}")
print(f"Result:  {result_string}")
print(f"Fitness: {best.fitness}")
```

This example demonstrates how Evolvium can solve optimization problems where traditional methods might struggle. The genetic algorithm explores the solution space through evolution, gradually improving candidates until an optimal solution is found.
