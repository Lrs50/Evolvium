import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from collections import OrderedDict
import time

class Cache(object):
    def __init__(self,size):
        self.storage = OrderedDict()
        self.limit = size

    def add(self,key,value):
        
        current_time = time.time()

        self.storage[key] = {
            "value" : value,
            "timestamp": current_time
        }

        self.storage.move_to_end(key)
        self._enforce_size_constraint()

    def get(self,key):
        current_time = time.time()

        if key not in self.storage:
            return None
        
        self.storage[key]['timestamp'] = current_time

        item = self.storage[key]

        self.storage.move_to_end(key)

        return item["value"]
    

    def _enforce_size_constraint(self):
        """Evict the oldest accessed item if the cache exceeds the max size."""
        if len(self.storage) > self.limit:
            # Remove the first item (oldest accessed item)
            self.storage.popitem(last=False)

    def __repr__(self):
        """Display the current cache contents."""
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
    return 0.1

def _repr(candidae):
    return f"Nothing to Show"

class Ind(object):
    def __init__(self,copy=False,**kwargs):

        self.kwargs          = kwargs
        self._gene           = None
        self._fitness        = None  

        self.metric          = kwargs.get('metric') if kwargs.get('metric') else _metric
        self.repr            = kwargs.get('repr')   if kwargs.get('repr')   else _repr

        if not copy:
            self.generate()

    def generate(self):
        kwargs = self.kwargs
        gene_type           = kwargs.get('gene_type')        if kwargs.get('gene_type')        else 'integer'
        gene_size           = kwargs.get('gene_size')        if kwargs.get('gene_size')        else 10
        gene_upper_limit    = kwargs.get('gene_upper_limit') if kwargs.get('gene_upper_limit') else 10
        init_method         = kwargs.get('init_method')      if kwargs.get('init_method')      else "random" 

        if init_method == 'mix':
            init_method = "random" if np.random.rand()<0.5 else "zeros"

        if init_method == "random":
            if gene_type == 'binary':
                self._gene = np.random.randint(2, size=gene_size)

            elif gene_type == 'integer':
                self._gene = np.random.randint(gene_upper_limit, size=gene_size)

            elif gene_type == 'real':
                self._gene = np.random.rand(gene_size)*gene_upper_limit

            else:
                raise(f"{gene_type} is not a Valid Gene Type")
        elif init_method == "zeros":
            self._gene = np.zeros(gene_size)
        else: 
            raise(f"{init_method} is not a Initialization Method")
        
    @property
    def fitness(self):
        if self._fitness == None:
            global total_fitness_calls
            total_fitness_calls+=1
            cached_fitness = cache.get(str(self._gene))
            if cached_fitness:
                self._fitness = cached_fitness
            else:
                global fitness_calls
                fitness_calls+=1
                self._fitness = self.metric(self._gene)
                cache.add(str(self._gene),self._fitness)
            
        return self._fitness

    @property
    def gene(self):
        if self._gene:
            return self._gene
        else:
            return "No genome Available"
    
    def mutate(self,prob = 1.0):

        if np.random.rand() < prob:
            kwargs = self.kwargs
            mutation_type       = kwargs.get('mutation_type')    if kwargs.get('mutation_type')    else 'random range'
            gene_type           = kwargs.get('gene_type')        if kwargs.get('gene_type')        else 'integer'
            gene_upper_limit    = kwargs.get('gene_upper_limit') if kwargs.get('gene_upper_limit') else 10
            mutation_range      = kwargs.get('mutation_range')   if kwargs.get('mutation_range')   else (-1,1) 

            if mutation_type == 'random range':

                low,high = mutation_range
                i = np.random.randint(len(self._gene))

                if gene_type == 'real':
                    self._gene[i] += np.random.rand()*(high - low) + low

                else:
                    self._gene[i] += np.random.randint(low = low,high = high+1)
                    self._gene[i] %= (gene_upper_limit+1)
            else:
                raise(f"{mutation_type} is not a Valid Mutation Type")
            
            self._fitness = None
        
        return self
    
    def copy(self):
        c = Ind(copy=True,**self.kwargs)
        c._gene           = self._gene.copy()
        c._fitness        = self._fitness  

        return c


    def crossover(self,dad):
        kwargs = self.kwargs
        crossover_type  = kwargs.get('crossover_type')      if kwargs.get('crossover_type')  else 'split'
        cromossome_size = kwargs.get('cromossome_size')     if kwargs.get('cromossome_size') else 1
        gene_type       = kwargs.get('gene_type')           if kwargs.get('gene_type')       else 'integer'
        n_cromossomes = len(self._gene)//cromossome_size

        new_gene = []
        if crossover_type == "random mix":
            choice = np.random.randint(2, size=n_cromossomes)
            
            for i,c in enumerate(choice):
                if c:
                    new_gene.extend(self._gene[i*cromossome_size:(i+1)*cromossome_size])
                else:
                    new_gene.extend(self._gene[i*cromossome_size:(i+1)*cromossome_size])
        elif crossover_type == "split":

            split_size = np.random.rand()

            cut_point = int(len(self._gene)*split_size)

            new_gene.extend(dad._gene[:cut_point])
            new_gene.extend(self._gene[cut_point:])

        elif crossover_type == 'avg':
            if gene_type != 'real':
                raise(f'{gene_type} is not a valid gene for {crossover_type} crossover!')
            
            new_gene = []

            for i in range(len(self._gene)):
                new_gene.append((dad._gene[i]+self._gene[i])/2)

        else:
            raise(f"{crossover_type} is not a Valid Crossover Type")

        child = Ind(**self.kwargs)
        child._gene = new_gene

        return child

    def __repr__(self):
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

def _roulette(fitness_list,pop,n):

    if n <= 0:
        return []

    fitness_list = np.array(fitness_list,dtype=float)

    info = 1/(fitness_list+0.0000001)

    prob = np.array(info)/np.sum(info)
    sur = np.random.choice(pop, size=n, p=prob, replace=False).tolist()

    return [s.copy() for s in sur]


def run(max_gen=100,pop_size=30,prole_size=10,mutation_rate=1/30,stop=0.5,verbose=False,**kwargs):

    cache_size = kwargs.get("cache_size") if kwargs.get("cache_size") else 1000
    
    if isinstance(mutation_rate, tuple):
        mutation_list = slow_transition(mutation_rate[0],mutation_rate[1],max_gen)
    else:
        mutation_list = np.ones(max_gen)*mutation_rate

    global fitness_calls
    global total_fitness_calls
    fitness_list_best = []
    fitness_list_avg  = []
    fitness_calls = 0
    total_fitness_calls = 0
    pop = []
    
    pbar = tqdm(list(range(pop_size)))

    for i in pbar:
        pop.append(Ind(**kwargs))
        pbar.set_description(f"Loading Initial Population | Current Fitness = {pop[i].fitness:.2e}")


    pop.sort(key = lambda x: x.fitness)
    best_global = pop[0].copy()
    pbar = tqdm(list(range(max_gen)))

    for gen in pbar:
        
        mutation_rate = mutation_list[gen]
        fitness_list = [ind.fitness for ind in pop]
        if best_global.fitness > pop[0].fitness:
            best_global = pop[0].copy() 
            if best_global.fitness <= stop:
                break
        
        best = best_global.fitness
        avg  = np.mean(fitness_list)

        fitness_list_avg.append(avg)
        fitness_list_best.append(best)
                
        if verbose:
            pbar.set_description(f"AVG = {avg:.2e} | BEST = {best:.2e} | {best_global} |Total Calls {fitness_calls:5d} | {total_fitness_calls-fitness_calls:5d} | Mutation Rate = {mutation_rate:.2%}")
        else:
             pbar.set_description(f"AVG = {avg:.2e} | BEST = {best:.2e}")
             
        survivors = _roulette(fitness_list.copy(), pop.copy(), pop_size - prole_size)

        new_gen = [s.mutate(mutation_rate) for s in survivors]

        for _ in range(prole_size):
            parents = _roulette(fitness_list, pop, 2)
            mum = parents[0]
            dad = parents[1]
            kid = mum.crossover(dad).mutate(mutation_rate)
            new_gen.append(kid)

        new_gen.sort(key=lambda x: x.fitness)
        pop = new_gen[:pop_size]


    if verbose:
        plt.figure(figsize=(20, 6))
        plt.suptitle("Genetic Algorithm Evolution Graph", fontsize=25)

        plt.plot(fitness_list_best,'--',color="lightsteelblue",label="Best")
        plt.plot(fitness_list_avg,'--',color="lightcoral",label="Average")

        plt.xlabel("generation")
        plt.ylabel("Fitness")

        plt.legend()
        plt.show()

    return best_global