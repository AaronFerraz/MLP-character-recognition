# activation_functions/leakyReLU.py
import numpy as np
from .fn_activation import Activation

class LeakyReLU(Activation):
    
    def __init__(self, alpha=0.01):
        self.__alpha = alpha
    
    """Função de ativação LeakyReLU """
    def __call__(self, x): #LeakyReLU(10) = y
        return np.where(x > 0, x, self.__alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1.0, self.__alpha)
    
    def getParams(self):
        return {"alpha": self.__alpha}
