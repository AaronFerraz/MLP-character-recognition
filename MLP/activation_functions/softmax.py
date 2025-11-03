# activation_functions/softmax.py
import numpy as np
from .fn_activation import Activation

class Softmax(Activation):
    __instance = None

    def __new__(cls): #Design Patterns Singleton
        if cls.__instance is None:
            cls.__instance = super(Softmax, cls).__new__(cls)
        return cls.__instance
    
    """Função de ativação Softmax"""
    def __call__(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # estabilidade numérica
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def derivative(self, x):
       softmax = self(x)
       batch_size, num_classes = softmax.shape
       jacobian = np.zeros((batch_size, num_classes, num_classes))
       for i in range(batch_size):
           s = softmax[i].reshape(-1, 1)
           jacobian[i] = np.diagflat(s) - np.dot(s, s.T)
       return jacobian
    
    def dactivation(self, dvalues, z): #Gradiente da SoftMax
        return np.einsum('ijk,ik->ij', self.derivative(z), dvalues)
    