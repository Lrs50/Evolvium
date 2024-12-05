import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

fitness_calls = 0

class Ind(object):
    def __init__(self,copy=False,**kwargs):

        self.kwargs          = kwargs
        self._gene           = None
        self._fitness        = None  

        self.metric          = kwargs.get('metric')
        self.repr            = kwargs.get('repr')

        if not copy:
            self.generate()

    def generate(self):
        gene_type           = self.kwargs.get('gene_type')
        gene_size           = self.kwargs.get('gene_size')
        gene_upper_limit    = self.kwargs.get('gene_upper_limit')

        if gene_type == 'binary':
            self._gene = np.random.randint(2, size=gene_size)

        elif gene_type == 'integer':
            self._gene = np.random.randint(gene_upper_limit, size=gene_size)

        elif gene_type == 'real':
            self._gene = np.random.rand(gene_type)*gene_upper_limit

        else:
            raise(f"{gene_type} is not a Valid Gene Type")
        
    @property
    def fitness(self):
        if self._fitness == None:
            global fitness_calls
            fitness_calls+=1
            self._fitness = self.metric(self._gene)
            
        return self._fitness

    @property
    def gene(self):
        if self._gene:
            return self._gene
        else:
            return "No genome Available"
    
    def mutate(self,prob = 1.0):

        if np.random.rand() < prob:

            mutation_type       = self.kwargs.get('mutation_type')
            gene_type           = self.kwargs.get('gene_type')
            gene_upper_limit    = self.kwargs.get('gene_upper_limit')

            if mutation_type == 'random range':

                mutation_range = self.kwargs.get('mutation_range')     
                mutation_range = (-1,1) if not mutation_range else mutation_range
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
        c._gene           = self._gene
        c._fitness        = self._fitness  

        return c


    def crossover(self,dad):
        crossover_type = self.kwargs.get('crossover_type')

        new_gene = None
        if crossover_type == "random mix":
            choice = np.random.randint(2, size=len(self._gene))
            new_gene = []
            
            for i,c in enumerate(choice):
                if c:
                    new_gene.append(self._gene[i])
                else:
                    new_gene.append(dad._gene[i])
        else:
            raise(f"{crossover_type} is not a Valid Crossover Type")

        child = Ind(**self.kwargs)
        child._gene = new_gene

        return child

    def __repr__(self):
        return self.repr(self._gene)
    

def _roulette(fitness_list,pop,n):

    if n <= 0:
        return []

    fitness_list = np.array(fitness_list)

    info = 1/fitness_list

    prob = np.array(info)/np.sum(info)
    sur = np.random.choice(pop, size=n, p=prob, replace=False).tolist()

    return [s.copy() for s in sur]


def run(max_gen=100,pop_size=30,prole_size=10,mutation_rate=1/30,stop=0,verbose=True,**kwargs):

    global fitness_calls
    fitness_list_best = []
    fitness_list_avg  = []
    fitness_calls = 0

    pop = []

    pbar = tqdm(list(range(pop_size)))

    for i in pbar:
        pop.append(Ind(**kwargs))
        pbar.set_description(f"Loading Initial Population | Current Fitness = {pop[i].fitness:.2f}")


    pop.sort(key = lambda x: x.fitness)
    best_global = pop[0].copy()
    pbar = tqdm(list(range(max_gen)))

    for gen in pbar:
    
        fitness_list = [ind.fitness for ind in pop]
        if best_global.fitness > pop[0].fitness:
            best_global = pop[0].copy() 
            if best_global.fitness == stop:
                break
        
        best = best_global.fitness
        avg  = np.mean(fitness_list)

        fitness_list_avg.append(avg)
        fitness_list_best.append(best)
                
        pbar.set_description(f"AVG = {avg:.2e} | BEST = {best:.2e} | {best_global} |Total Fitness Calculations = {fitness_calls:5d}")

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