import numpy as np


class Ind(object):
    def __init__(self,custom_genetator=None,metric=None,copy=False,custom_mutation=None,view=None):
        
        self._gene           = None
        self.generator       = custom_genetator
        self.metric          = metric
        self._fitness        = None  
        self.custom_mutation = custom_mutation
        self.view            = view
        if not copy:
            self._gene       = self.generate()

    def generate(self,type=None):
        if self.generator:
            return self.generator()
        
    @property
    def fitness(self):
        if self._fitness == None:
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
        c = Ind(copy=True)
        c._gene           = self._gene
        c.generator       = self.generator
        c.metric          = self.metric
        c._fitness        = self._fitness  
        c.custom_mutation = self.custom_mutation
        c.view            = self.view

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


        child = self.copy()

        child._gene = new_gene

        return child

    def __repr__(self):
        return self.view(self._gene)
