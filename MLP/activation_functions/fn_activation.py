from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def __call__(self, x):
        pass
    @abstractmethod
    def derivative(self, x):
        pass
    #Cálculo do gradiente
    def dactivation(self, dvalues, z):
        return dvalues * self.derivative(z)
    #Retorna o nome da função de ativação
    def getName(self):
        return self.__class__.__name__.lower()
    
    def getParams(self):
        return None #default value