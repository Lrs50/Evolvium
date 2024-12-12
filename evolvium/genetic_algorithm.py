import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from collections import OrderedDict
import time

"""
added no limits on range in gene option
initialization range 

"""

class Cache(object):
    def __init__(self, size=100):
        '''
        Initializes a Cache object with a least-recently-used (LRU) eviction policy.

        Parameters:
        size (int): Maximum number of elements allowed in the cache at any time.
        '''
        self.storage = OrderedDict()
        self.limit = size

    def add(self, key, value):
        '''
        Adds an element to the cache. If the cache exceeds the maximum size, the least-recently-used item is evicted.

        Parameters:
        key: The key used to store and retrieve the value.
        value: The data to be stored in the cache.
        '''
        current_time = time.time()

        self.storage[key] = {
            "value": value,
            "timestamp": current_time
        }

        # Mark the item as the most recently used
        self.storage.move_to_end(key)
        self._enforce_size_constraint()

    def get(self, key):
        '''
        Retrieves the value associated with the given key from the cache.

        Parameters:
        key: The key to retrieve the corresponding value.

        Returns:
        The value associated with the key if it exists, or None if the key is not found.
        '''
        current_time = time.time()

        if key not in self.storage:
            return None

        # Update the access timestamp and move the item to the end
        self.storage[key]['timestamp'] = current_time
        self.storage.move_to_end(key)

        return self.storage[key]["value"]

    def _enforce_size_constraint(self):
        '''
        Ensures the cache does not exceed its size limit by evicting the least-recently-used item if necessary.

        This is an internal method.
        '''
        if len(self.storage) > self.limit:
            # Remove the least-recently-used item (the first item in the OrderedDict)
            self.storage.popitem(last=False)

    def __repr__(self):
        '''
        Provides a string representation of the Cache object for debugging purposes.

        Returns:
        str: A dictionary-like string representation of the cache's contents,
             showing keys, values, and their corresponding timestamps.
        '''
        return str({
            key: {
                "value": meta["value"],
                "timestamp": meta["timestamp"]
            }
            for key, meta in self.storage.items()
        })


cache = Cache(size=1000)
fitness_calls = 0
total_fitness_calls = 0

def _metric(candidate):
    '''
    Default metric function for evaluating a candidate object.

    Returns:
    float: A constant value (0.1) to ensure consistency when no custom metric is provided.
    '''
    return 0.1

def _repr(candidate):
    '''
    Default string representation function for a candidate object.

    Parameters:
    candidate: The object to be represented as a string.

    Returns:
    str: The string representation of the candidate, using Python's `str` function.
    '''
    return str(candidate)


class Ind(object):
    def __init__(self, copy=False, **kwargs):
        '''
        Initializes an individual object with customizable genetic properties.

        Parameters:
        copy (bool): Indicates if the object is a copy of another individual.
        **kwargs: Additional attributes for customization.

        Customizable Attributes:
        metric (function): Evaluation function for fitness. Defaults to `_metric`.
        repr (function): String representation function. Defaults to `_repr`.
        gene_type (str): Type of genes ('binary', 'integer', 'real'). Defaults to 'integer'.
        gene_size (int): Number of genes in the chromosome. Defaults to 10.
        gene_upper_limit (int/float): Upper limit for gene values. Defaults to 10.
        init_method (str): Initialization method ('random', 'mix', 'zeros'). Defaults to 'random'.
        mutation_type (str): Mutation strategy ('random range'). Defaults to 'random range'.
        mutation_range (tuple): Range for mutation values (low, high). Defaults to (-1, 1).
        crossover_type (str): Crossover strategy ('random mix', 'split', 'avg'). Defaults to 'split'.
        chromossome_size (int): Size of each chromosome segment. Defaults to 1.
        '''
        self.kwargs = kwargs
        self._gene = None
        self._fitness = None  

        self.metric = kwargs.get('metric', _metric)
        self.repr = kwargs.get('repr', _repr)

        if not copy:
            self.generate()

    def generate(self):
        '''
        Generates a new genome based on initialization parameters.

        The genome's characteristics (type, size, and values) are determined by:
        - `gene_type`: Defines whether genes are binary, integer, or real.
        - `gene_size`: Specifies the number of genes in the genome.
        - `gene_upper_limit`: Sets the upper limit for gene values.
        - `init_method`: Defines the initialization method.
        '''
        kwargs = self.kwargs
        gene_type = kwargs.get('gene_type', 'integer')
        gene_size = kwargs.get('gene_size', 10)
        init_method = kwargs.get('init_method', 'random')
        gene_init_range = kwargs.get('gene_init_range', 10)

        if init_method == 'mix':
            init_method = "random" if np.random.rand() < 0.5 else "zeros"

        if init_method == "random":
            if gene_type == 'binary':
                self._gene = np.random.randint(2, size=gene_size)
            elif gene_type == 'integer':
                self._gene = np.random.randint(gene_init_range, size=gene_size)
            elif gene_type == 'real':
                self._gene = np.random.rand(gene_size) * gene_init_range
            else:
                raise ValueError(f"{gene_type} is not a valid gene type")
        elif init_method == "zeros":
            self._gene = np.zeros(gene_size)
        else:
            raise ValueError(f"{init_method} is not a valid initialization method")

    @property
    def fitness(self):
        '''
        Evaluates the quality of the solution.

        Uses the metric function and a caching mechanism to avoid redundant calculations.
        Increments global counters for fitness evaluations.
        '''
        if self._fitness is None:
            global total_fitness_calls
            total_fitness_calls += 1
            cached_fitness = cache.get(str(self._gene))
            if cached_fitness:
                self._fitness = cached_fitness
            else:
                global fitness_calls
                fitness_calls += 1
                self._fitness = self.metric(self._gene)
                cache.add(str(self._gene), self._fitness)
        return self._fitness

    @property
    def gene(self):
        '''
        Returns the genome representation.

        If no genome is available, a default message is returned.
        '''
        return self._gene if self._gene is not None else "No genome available"

    def mutate(self, prob=1.0):
        '''
        Mutates the current genome based on the specified mutation parameters.

        Parameters:
        prob (float): Probability of mutation (0 to 1). Defaults to 1.0.
        '''
        if np.random.rand() < prob:
            kwargs = self.kwargs
            mutation_type = kwargs.get('mutation_type', 'random range')
            gene_type = kwargs.get('gene_type', 'integer')
            gene_upper_limit = kwargs.get('gene_upper_limit', 10)
            mutation_range = kwargs.get('mutation_range', (-1, 1))

            if mutation_type == 'random range':
                low, high = mutation_range
                i = np.random.randint(len(self._gene))
                if gene_type == 'real':
                    if gene_upper_limit > 0:
                        self._gene[i] += (np.random.rand() * (high - low) + low)%gene_upper_limit
                    else:
                        self._gene[i] += (np.random.rand() * (high - low) + low)
                else:
                    self._gene[i] += np.random.randint(low, high + 1)
                    if gene_upper_limit>0:
                        self._gene[i] %= (gene_upper_limit + 1)
                    
            else:
                raise ValueError(f"{mutation_type} is not a valid mutation type")

            self._fitness = None
        return self

    def copy(self):
        '''
        Returns a copy of the current individual, preserving its genome and fitness.
        '''
        c = Ind(copy=True, **self.kwargs)
        c._gene = self._gene.copy()
        c._fitness = self._fitness
        return c

    def crossover(self, dad):
        '''
        Creates a new genome by combining the genes of the current individual and a "dad".

        Parameters:
        dad (Ind): Another individual whose genes are used for crossover.

        Crossover Strategies:
        - `random mix`: Randomly selects gene segments from each parent.
        - `split`: Divides genes at a random cut point.
        - `avg`: Computes the average of corresponding genes (real only).
        '''
        kwargs = self.kwargs
        crossover_type = kwargs.get('crossover_type', 'split')
        chromosome_size = kwargs.get('chromosome_size', 1)
        gene_type = kwargs.get('gene_type', 'integer')
        n_chromosome = len(self._gene) // chromosome_size

        new_gene = []
        if crossover_type == "random mix":
            choice = np.random.randint(2, size=n_chromosome)
            for i, c in enumerate(choice):
                if c:
                    new_gene.extend(self._gene[i * chromosome_size:(i + 1) * chromosome_size])
                else:
                    new_gene.extend(dad._gene[i * chromosome_size:(i + 1) * chromosome_size])
        elif crossover_type == "split":
            split_size = np.random.rand()
            cut_point = int(len(self._gene) * split_size)
            new_gene.extend(dad._gene[:cut_point])
            new_gene.extend(self._gene[cut_point:])
        elif crossover_type == 'avg':
            if gene_type != 'real':
                raise ValueError(f'{gene_type} is not a valid gene type for {crossover_type} crossover!')
            new_gene = [(dad._gene[i] + self._gene[i]) / 2 for i in range(len(self._gene))]
        else:
            raise ValueError(f"{crossover_type} is not a valid crossover type")

        child = Ind(**self.kwargs)
        child._gene = new_gene
        return child

    def __repr__(self):
        '''
        Returns the string representation of the current individual using the provided representation function.
        '''
        return self.repr(self._gene)

    
def slow_transition(x=0.01, y=0.1, n=1000, power=3):
    """
    Generates a sequence transitioning from x to y in n steps, slowing as it approaches y.
    
    Parameters:
    x (float): Start value.
    y (float): End value.
    n (int): Number of steps.
    power (float): Controls the slowness of the transition. Higher values mean slower transition near y.
    
    Returns:
    list: A list of n values transitioning from x to y.
    """
    # Create a normalized scale from 0 to 1 with n points
    t = np.linspace(0, 1, n)
    # Apply a power function to slow the transition
    t_slow = t**power
    # Scale and shift to transition from x to y
    transition = x + (y - x) * t_slow
    return list(transition)

def _roulette(fitness_list, pop, n):
    '''
    Implements the roulette wheel selection mechanism to choose individuals for reproduction.

    Parameters:
    fitness_list (list): A list of fitness values corresponding to individuals in the population.
    pop (list): A list of individuals in the population.
    n (int): The number of individuals to select for the next generation.

    Returns:
    list: A list of `n` selected individuals based on their fitness.
    
    Selection Process:
    - Fitness values are inversely transformed into probabilities.
    - The probability for each individual is proportional to the inverse of its fitness.
    - Individuals are selected randomly based on these probabilities.
    - A copy of the selected individuals is returned.
    '''
    
    # If no individuals need to be selected, return an empty list
    if n <= 0:
        return []

    # Convert fitness values to a numpy array for better performance
    fitness_list = np.array(fitness_list, dtype=float)

    # Calculate the inverse of fitness values to get the selection probability
    info = 1 / fitness_list

    # Normalize the probabilities so that they sum to 1
    prob = np.array(info) / np.sum(info)

    # Randomly select 'n' individuals from the population based on their selection probability
    selected = np.random.choice(pop, size=n, p=prob, replace=False).tolist()

    # Return a list of copies of the selected individuals to avoid reference issues
    return [s.copy() for s in selected]


def run(max_gen=100, pop_size=30, prole_size=10, mutation_rate=1/30, stop=0.5, verbose=False, **kwargs):
    '''
    Runs the genetic algorithm for a specified number of generations to evolve a population.

    Parameters:
    max_gen (int): The maximum number of generations to run the algorithm. Default is 100.
    pop_size (int): The size of the population in each generation. Default is 30.
    prole_size (int): The number of individuals that will be generated by crossover (offspring). Default is 10.
    mutation_rate (float or tuple): The rate of mutation. If tuple, a slow transition is applied over generations. Default is 1/30.
    stop (float): The fitness threshold at which the algorithm will stop early. Default is 0.5.
    verbose (bool): Whether to display detailed progress and information about the algorithm. Default is False.
    **kwargs: Additional keyword arguments passed to the `Ind` class for initializing individuals.

    Returns:
    Ind: The best individual found after all generations.
    
    Evolution Process:
    - Initializes a random population.
    - For each generation:
        - Selects individuals for reproduction using roulette wheel selection.
        - Applies crossover and mutation to produce offspring.
        - The best individual is tracked and returned if it meets the stop threshold.
    - Optionally, visualizes the evolution of fitness over generations.
    '''
    
    # Cache size, useful for optimization or preventing recalculating fitness
    cache_size = kwargs.get("cache_size", 1000)
    debug_func = kwargs.get("debug_func", None)


    # Determine mutation rate progression (if using a tuple, gradually change mutation rate)
    if isinstance(mutation_rate, tuple):
        mutation_list = slow_transition(mutation_rate[0], mutation_rate[1], max_gen)
    else:
        mutation_list = np.ones(max_gen) * mutation_rate

    # Initialize variables for fitness tracking
    global fitness_calls, total_fitness_calls
    fitness_list_best = []
    fitness_list_avg = []
    fitness_calls = 0
    total_fitness_calls = 0

    # Initialize population
    pop = []
    pbar = tqdm(list(range(pop_size)))

    for i in pbar:
        # Create individuals for the initial population
        pop.append(Ind(**kwargs))
        pbar.set_description(f"Loading Initial Population | {pop[i]} |Current Fitness = {pop[i].fitness:.2e}")

    # Sort population by fitness (ascending)
    pop.sort(key=lambda x: x.fitness)
    
    # Track the best individual globally
    best_global = pop[0].copy()
    if verbose and debug_func:
        debug_func(best_global.gene)


    pbar = tqdm(list(range(max_gen)))

    for gen in pbar:

        diversity = {str(p) for p in pop}

        # Update mutation rate for this generation
        mutation_rate = mutation_list[gen]

        # Calculate fitness of the current population
        fitness_list = [ind.fitness for ind in pop]

        # Update the best global individual if a better one is found
        if best_global.fitness > pop[0].fitness:
            best_global = pop[0].copy()
            if verbose and debug_func:
                debug_func(best_global.gene)
            # Stop early if the best fitness is below the stop threshold
            if best_global.fitness <= stop:
                break

        # Track the best and average fitness values
        best = best_global.fitness
        avg = np.mean(fitness_list)
        fitness_list_avg.append(avg)
        fitness_list_best.append(best)

        # Display progress bar with additional information (verbose)
        if verbose:
            pbar.set_description(f"AVG = {avg:.2e} | BEST = {best:.2e} | D = {len(diversity)/pop_size:.2f} |{best_global} |Calls {fitness_calls:5d} | {total_fitness_calls-fitness_calls:5d} | Mutation Rate = {mutation_rate:.2%}")
        else:
            pbar.set_description(f"AVG = {avg:.2e} | BEST = {best:.2e}")

        # Select survivors using roulette wheel selection
        survivors = _roulette(fitness_list.copy(), pop.copy(), pop_size - prole_size)

        # Apply mutation to survivors
        new_gen = [s.mutate(mutation_rate) for s in survivors]

        # Generate offspring via crossover and mutation
        for _ in range(prole_size):
            parents = _roulette(fitness_list, pop, 2)
            mum = parents[0]
            dad = parents[1]
            kid = mum.crossover(dad).mutate(mutation_rate)
            new_gen.append(kid)

        # Sort the new generation by fitness and select the top population size
        new_gen.sort(key=lambda x: x.fitness)
        pop = new_gen[:pop_size]

    # Optionally visualize the fitness evolution over generations
    if verbose:
        plt.figure(figsize=(20, 6))
        plt.suptitle("Genetic Algorithm Evolution Graph", fontsize=25)
        plt.plot(fitness_list_best, '--', color="lightsteelblue", label="Best")
        #plt.plot(fitness_list_avg, '--', color="lightcoral", label="Average")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

    return best_global
