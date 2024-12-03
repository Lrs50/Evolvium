# All of the functions inside the library
import numpy as np


class Ind(object):
    def __init__(self,custom_genetator=None,metric=None,copy=False):
        
        self.gene       = None
        self.generator  = custom_genetator
        self.metric     = metric

        if not copy:
            self.gene       = self.generate

    def generate(self,type=None,):
        if self.generator:
            return self.generator()
        

    def score(self,metric=None):
        self.fitness = metric(self.gene,self.decode)
    
    def mutate(self,dad,type = "Random Split"):
        pass
