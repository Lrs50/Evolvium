import numpy as np
from tqdm import tqdm 

fitness_calls = 0

class Ind(object):
    def __init__(self,copy=False,**kwargs):

        self.kwargs          = kwargs
        self._gene           = None
        self.generator       = kwargs.get('generator')
        self.metric          = kwargs.get('metric')
        self._fitness        = None  
        self.custom_mutation = kwargs.get('custom_mutation')
        self.view            = kwargs.get('view')
        if not copy:
            self._gene       = self.generate()

    def generate(self,type=None):
        if self.generator:
            return self.generator()
        
    @property
    def fitness(self):
        if self._fitness == None:
            global fitness_calls
            fitness_calls+=1
            self.get_fitness() 
            
        return self._fitness

    @property
    def gene(self):
        if self._gene:
            return self._gene
        else:
            return "No genome Available"
        
    def get_fitness(self):
        self._fitness = self.metric(self._gene)
        
    
    def mutate(self,prob = 1.0,type = "Random Change"):
        if np.random.rand() < prob:
            if self.custom_mutation:
                self._gene = self.custom_mutation(self._gene)

            self._fitness = None
        
        return self
    
    def copy(self):
        c = Ind(copy=True,**self.kwargs)
        c._gene           = self._gene
        c._fitness        = self._fitness  

        return c


    def crossover(self,dad,type = "Random Mix"):
        new_gene = None
        if type == "Random Mix":
            choice = np.random.randint(2, size=len(self._gene))

            new_gene = []

            for i,c in enumerate(choice):
                if c:
                    new_gene.append(self._gene[i])
                else:
                    new_gene.append(dad._gene[i])


        child = Ind(**self.kwargs)
        child._gene = new_gene

        return child

    def __repr__(self):
        return self.view(self._gene)
    

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

        pbar.set_description(f"AVG = {np.mean(fitness_list):.2e} | BEST = {best_global.fitness:.2e} | {best_global} |Total Fitness Calculations = {fitness_calls:5d}")

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

    return best_global

