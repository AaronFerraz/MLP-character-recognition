import numpy as np
from .activation_functions.softmax import Softmax

class Layer:
    def __init__(self, activation_function, n_neurons, n_inputs):
        """
        Inicializa uma camada da rede neural com pesos, bias e função de ativação.

        Parâmetros:
        - activation_function: objeto com a função de ativação a ser usada
        - n_neurons: número de neurônios na camada
        - n_inputs: número de entradas que cada neurônio recebe

        Saída: nenhuma (construtor)
        """
        self.__activation = activation_function
        # Inicializa pesos (n_inputs x n_neurons) e biases (1 x n_neurons)
        self.__weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.__biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Executa o passo de forward propagation na camada.

        Parâmetros:
        - inputs: entrada recebida da camada anterior (ou dos dados de entrada)

        Saída: resultado da ativação aplicado à combinação linear (z)
        """
        self.__inputs = inputs           # armazenar para backprop
        self.__z = np.dot(inputs, self.__weights) + self.__biases
        return self.__activation(self.__z)

    def backward(self, dvalues, learning_rate): #Retropropaga o erro
        """
        Executa o passo de backpropagation atualizando os pesos e bias.

        Parâmetros:
        - dvalues: derivada da função de perda em relação à saída da camada
        - learning_rate: taxa de aprendizado usada para atualização dos parâmetros

        Saída: derivada da perda em relação à entrada da camada (para retropropagação)
        """        
        dactivation = self.__activation.dactivation(dvalues, self.__z)

        dweights = np.dot(self.__inputs.T, dactivation)
        dbiases = np.sum(dactivation, axis=0, keepdims=True)
        dinputs = np.dot(dactivation, self.__weights.T)

        self.__weights -= learning_rate * dweights
        self.__biases -= learning_rate * dbiases

        return dinputs
    
    def getNameActivation(self): #Devolve o nome da função ativação
        return self.__activation.getName()
    
    def getQuantityNeurons(self):
        return self.__weights.shape[1] #neurons
    
    def reset(self):
        self.__weights = np.random.randn(*self.__weights.shape) * 0.01
    
    def getWeights(self):
        return self.__weights
    
    def getBiases(self):
        return self.__biases
    
    def getParamsActivation(self): #Parametros da função de ativação
        return self.__activation.getParams()
    
    def setWeights(self, weights):
        self.__weights = np.array(weights)
    
    def setBiases(self, biases):
        self.__biases = np.array(biases)