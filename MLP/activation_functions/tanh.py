import numpy as np
from .fn_activation import Activation

class Tanh(Activation):
    __instance = None

    def __new__(cls): #Design Patterns Singleton
        if cls.__instance == None:
            cls.__instance = super(Tanh, cls).__new__(cls)
        return cls.__instance
    """Função de ativação Tanh"""
    def __call__(self, x):
        e_x = np.exp(x)
        e_neg_x = np.exp(-x)
        return (e_x - e_neg_x) / (e_x + e_neg_x)
    
    def derivative(self, x):
        tanh = self(x)
        return 1 - tanh ** 2