from .layer import Layer
import json
from .activation_functions.ReLU import ReLU
from .activation_functions.softmax import Softmax
from .activation_functions.tanh import Tanh
from .activation_functions.sigmoid import Sigmoid
from .activation_functions.leakyReLU import LeakyReLU
from .activation_functions.ELU import ELU
from .losses.loss_crossentropy import LossCrossEntropy
from .losses.loss_mse import LossMSE
from .trainer import *

class MLP: #MultiLayer Perceptron
    def __init__(self, configs, trainer):
        """
        Inicializa a rede neural MLP com as camadas e o treinador especificado.

        Parâmetros:
        - configs: lista de tuplas (n_neurons, n_inputs, activation_function) para definir cada camada
        - trainer: objeto responsável pelo processo de treinamento da rede

        Saída: nenhuma (construtor)
        """
        self.__layers = [Layer(activation, n_neurons, n_inputs) for n_neurons, n_inputs, activation in configs]
        self.__trainer = trainer 
        self.__show_banner()

    def __show_banner(self):
        """
        Exibe no terminal um resumo da arquitetura da rede MLP.

        Parâmetros: nenhum

        Saída: nenhuma (apenas exibe informações)
        """
        print("\033[95m" + "="*60)
        print("      🧠  Multi-Layer Perceptron Created! 🧠")
        print("\033[94mMulti-Layer Perceptron Configuration")
        print(f"Epochs: {self.__trainer.get_epochs()}")
        print(f"Learning rate: {self.__trainer.get_learning_rate()}")
        print(f"Number of layers: {len(self.__layers)}")
        print("Architecture:")
        for i, layer in enumerate(self.__layers):
            print(f"  Layer {i+1}: {layer.getQuantityNeurons()} neurons - Activation: {layer.getNameActivation()}")
        print("\033[95m" + "="*60)
        print("\033[0m")

    def train(self, X, y, k=None):
        """
        Treina a rede MLP com os dados fornecidos.

        Parâmetros:
        - X: dados de entrada
        - y: rótulos ou saídas esperadas
        - k: valor opcional para validação cruzada (caso aplicável)

        Saída: resultado do método de treino do objeto trainer
        """
        return self.__trainer.train(self.__layers, X, y, k)

    def predict(self, X):
        """
        Realiza a previsão da saída da rede para uma entrada X.

        Parâmetros:
        - X: dados de entrada

        Saída: array com a saída final da rede após passar por todas as camadas
        """
        output = X
        for layer in self.__layers:
            output = layer.forward(output)
        return output
    
    def _setWeights(self, weights, biases): #Coloca os pesos indicados pelo arquivo .model
        """
        Define manualmente os pesos e bias de cada camada da rede.

        Parâmetros:
        - weights: lista de arrays com os pesos de cada camada
        - biases: lista de arrays com os bias de cada camada

        Saída: nenhuma
        """
        i = 0
        for layer in self.__layers:
            layer.setWeights(weights[i])
            layer.setBiases(biases[i])
            i+=1

    def save(self, file_name): #salva o modelo
        """
        Salva a arquitetura, pesos, bias e informações de treinamento da rede em um arquivo JSON.

        Parâmetros:
        - file_name: nome do arquivo onde o modelo será salvo

        Saída: nenhuma (salva em arquivo)
        """        
        model = {
            "weights": [layer.getWeights().tolist() for layer in self.__layers],
            "biases": [layer.getBiases().tolist() for layer in self.__layers],
            "fn_activations": [{"name":layer.getNameActivation(), "params": layer.getParamsActivation()} for layer in self.__layers],
            "training_info": {
                "epochs": self.__trainer.get_epochs(),
                "learning_rate": self.__trainer.get_learning_rate(),
                "loss": self.__trainer.get_lossName(),
                "type": self.__trainer.getType()
            }
        }
        with open(file_name, "w") as file:
            json.dump(model, file, indent=2)

    @staticmethod
    def load_model(path): #carrega o modelo
        """
        Carrega um modelo salvo em JSON e reconstrói a rede MLP com pesos, bias e configuração original.

        Parâmetros:
        - path: caminho para o arquivo JSON contendo o modelo salvo

        Saída: objeto MLP reconstruído com base nas informações do arquivo
        """
        with open(path, "r") as file:
            model = json.load(file)
        activation_factory = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "softmax": Softmax,
            "tanh": Tanh,
            "leakyrelu": LeakyReLU,
            "elu": ELU
        }
        loss_factory = {
            "mse": LossMSE,
            "crossentropy": LossCrossEntropy
        }
        trainer_factory = {
            "BackPropagation": BackPropagation,
            "BackPropagationCV": BackPropagationCV,
            "BackPropagationES": BackPropagationES
        }
        configs = []
        for i in range(0, len(model['weights'])): #Adiciona cada configuração das camadas em configs
            if model['fn_activations'][i]["params"] == None: #Caso a função de ativação não tenha hiperparâmetros
                configs.append((len(model['weights'][i][0]), len(model['weights'][i]), activation_factory[model['fn_activations'][i]['name']]()))
            else:
                configs.append((len(model['weights'][i][0]), len(model['weights'][i]), activation_factory[model['fn_activations'][i]['name']](**model['fn_activations'][i]['params'])))
        loss = loss_factory[model['training_info']['loss']]()
        trainer = trainer_factory[model['training_info']['type']](loss, model['training_info']['learning_rate'], model['training_info']['epochs'])
        mlp = MLP(configs, trainer)
        mlp._setWeights(model['weights'], model['biases']) #Carrega os pesos
        return mlp
            