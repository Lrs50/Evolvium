# Evolution

## Genetic Algorithm

This project implements a genetic algorithm with a caching mechanism for optimizing solutions. It supports customizable parameters such as mutation rate, crossover strategies, and initialization methods. The caching mechanism uses a least-recently-used (LRU) policy to store and retrieve fitness values, improving the performance of repeated evaluations.

## Features

- **Cache**: Implements an LRU cache for storing fitness evaluations to avoid redundant calculations.
- **Customizable Individuals**: Each individual in the population has customizable properties, including gene types, initialization methods, and mutation strategies.
- **Genetic Operations**: Supports standard genetic algorithm operations like mutation, crossover, and fitness evaluation.
- **Roulette Wheel Selection**: Selects individuals for reproduction based on their fitness, with the probability inversely proportional to their fitness.
- **Evolution Tracking**: Tracks the evolution of the population's fitness over generations.
- **Flexible Termination**: Allows early termination when a fitness threshold is met.


## `Ind` Class Documentation

### Overview

The `Ind` class represents an individual in an evolutionary algorithm, typically used for optimization or genetic algorithms. It is highly customizable, allowing users to define various genetic properties and evaluation functions for each individual.

### Parameters

#### `copy` (bool)
Indicates whether the individual is a copy of another individual. If `True`, the individual is copied from an existing one, preserving its genetic material. If `False`, a new individual is initialized with random genetic values.

#### `**kwargs`
Additional attributes can be passed for further customization. These include properties related to the individual’s genetic makeup and behavior during evolutionary processes.

### Customizable Attributes

#### `metric` (function, default: `_metric`)
A function used to evaluate the fitness of the individual. The fitness value is critical for selection and reproduction in evolutionary algorithms. The default function, `_metric`, returns a constant value (`0`), but users can provide a custom function for more complex evaluations.

#### `repr` (function, default: `_repr`)
A function that defines the string representation of the individual. This is useful for inspecting the object and debugging. The default function returns a basic representation of the individual.

#### `gene_type` (str, default: `'integer'`)
The type of genes that make up the individual's chromosome. Possible values are:
- `'binary'`: Genes are binary (0 or 1).
- `'integer'`: Genes are integers.
- `'real'`: Genes are real (floating-point) numbers.

#### `gene_size` (int, default: 10)
The number of genes in the chromosome. This defines the length of the genetic sequence for the individual. Increasing this number allows for more genetic diversity.

#### `gene_upper_limit` (int/float, default: 10)
The upper limit for the gene values. The individual’s genes will have values between the lower limit (usually 0) and this upper limit.

#### `init_method` (str, default: `'random'`)
The method used to initialize the individual's genes. Options include:
- `'random'`: Genes are initialized with random values within the specified limits.
- `'mix'`: Genes are initialized by mixing values from other individuals (e.g., for crossover).
- `'zeros'`: All genes are initialized to zero.

#### `mutation_type` (str, default: `'random range'`)
The type of mutation applied to the individual’s genes. Common options include:
- `'random range'`: Mutates genes by randomly selecting a value within a specified range.

#### `mutation_range` (tuple, default: `(-1, 1)`)
Defines the range within which the mutation values will be chosen. The mutation will modify gene values randomly within this range. For example, a mutation range of `(-1, 1)` means the mutation will be a random value between -1 and 1.

#### `crossover_type` (str, default: `'split'`)
The crossover strategy applied during reproduction. This defines how two individuals combine their genetic material to create offspring. Possible strategies include:
- `'random mix'`: Genes from both parents are mixed randomly.
- `'split'`: The genetic material is split between the two parents, with parts coming from each.
- `'avg'`: Genes are averaged from both parents.

#### `chromossome_size` (int, default: 1)
The size of each chromosome segment. If the chromosome is composed of multiple segments (e.g., if each gene represents a vector), this defines the number of components in each segment.

#### Example Usage

```python
# Create an individual with custom properties
ind = Ind(copy=False, gene_type='real', gene_size=20, gene_upper_limit=5.0, init_method='random', mutation_type='random range', mutation_range=(-2, 2), crossover_type='random mix')

# Evaluate the individual's fitness using a custom metric function
fitness = ind.metric(ind)
```

## `run` Function Documentation

### Overview

The `run` function executes the genetic algorithm over a specified number of generations, evolving a population to optimize for a given fitness function. The function iteratively selects individuals, applies crossover and mutation, and tracks the best individual based on its fitness score.

### Parameters

#### `max_gen` (int, default: 100)
The maximum number of generations the algorithm will run. The algorithm stops either when this number of generations is reached or when an individual meets the fitness threshold (`stop`).

#### `pop_size` (int, default: 30)
The size of the population in each generation. A larger population can increase diversity but also requires more computational resources. 

#### `prole_size` (int, default: 10)
The number of individuals generated by crossover (offspring) in each generation. These offspring replace individuals in the population during the reproduction step.

#### `mutation_rate` (float or tuple, default: `1/30`)
The rate at which mutation occurs in the population. If a tuple is provided, the mutation rate will transition gradually over generations. A higher rate may increase diversity but can also disrupt convergence.

#### `stop` (float, default: 0.5)
The fitness threshold at which the algorithm will stop early. If the best individual's fitness meets or exceeds this threshold, the algorithm terminates before reaching the maximum number of generations.

#### `verbose` (bool, default: False)
If set to `True`, the function will display detailed progress and information about the algorithm's performance, including fitness scores, the number of generations completed, and other relevant metrics.

#### `**kwargs`
Additional keyword arguments passed to the `Ind` class for initializing individuals. These arguments can be used to customize the genetic properties of the individuals, such as gene type, size, and mutation methods.

### Returns

#### `Ind`
The best individual found after running all generations. This individual is returned as the final solution to the optimization problem.

### Evolution Process

1. **Initialization**:
   - A random population of individuals is created based on the parameters passed to the `Ind` class.
   
2. **Selection**:
   - Individuals are selected for reproduction based on their fitness using roulette wheel selection. This method gives a higher chance of selection to individuals with better fitness.

3. **Crossover**:
   - Selected individuals undergo crossover to generate offspring. The `prole_size` determines how many offspring are created per generation.

4. **Mutation**:
   - Mutation is applied to the offspring based on the `mutation_rate`. This introduces small random changes to the genes of the offspring to maintain genetic diversity.

5. **Termination**:
   - The algorithm tracks the best individual throughout the generations. If the best individual’s fitness meets the threshold (`stop`), the algorithm terminates early. Otherwise, it continues until `max_gen` generations are reached.

6. **Optional Visualization**:
   - If desired, the evolution of the population’s fitness over generations can be visualized to understand the convergence behavior.

### Example Usage

```python
# Run the genetic algorithm with custom settings
best_individual = run(max_gen=200, pop_size=50, prole_size=15, mutation_rate=0.05, stop=0.9, verbose=True)

# The best individual after all generations
print(best_individual)
```
