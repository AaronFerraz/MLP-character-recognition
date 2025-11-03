import numpy as np
import random
import copy

class BackPropagation:
    def __init__(self, loss, learning_rate=0.01, epochs=1000):
        """
        Inicializa o processo de Backpropagation com a função de perda, taxa de aprendizado e número de épocas.

        Parâmetros:
        - loss: função de perda a ser utilizada
        - learning_rate: taxa de aprendizado para atualização dos pesos
        - epochs: número de épocas para treinamento

        Saída: nenhuma (construtor)
        """
        self.__loss_fn = loss
        self.__lr = learning_rate
        self.__epochs = epochs

    def train(self, layers, X, y, k=None):
        """
        Treina a rede neural utilizando o algoritmo de Backpropagation.

        Parâmetros:
        - layers: lista de camadas da rede
        - X: dados de entrada para o treinamento
        - y: rótulos ou saídas esperadas para o treinamento
        - k: número de folds (se aplicável, para validação cruzada)

        Saída: nenhuma (apenas realiza o treinamento e exibe os pesos)
        """
        print('PESOS INICIAIS')
        for layer in layers:
            print(layer.getWeights())
            print('\n')
        print('\n')

        for epoch in range(1, self.__epochs+1):
            # Forward pass
            output = X
            for layer in layers:
                output = layer.forward(output)

            # Calcula loss
            loss = self.__loss_fn.forward(output, y)

            # Inicia backward pass com gradiente da loss
            dvalues = self.__loss_fn.backward(output, y)
            
            # Backprop em cada layer (em ordem reversa)
            for layer in reversed(layers):
                dvalues = layer.backward(dvalues, self.__lr)

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.__epochs} - Loss: {loss:.6f}")

                print('\nPESOS')
                for layer in layers:
                    print(layer.getWeights())
                    print('\n')
                print('\n')

        print('\nPESOS FINAIS')
        for layer in layers:
            print(layer.getWeights())
            print('\n')
        print('\n')

    def get_learning_rate(self):
        return self.__lr
    
    def get_epochs(self):
        return self.__epochs
    
    def get_lossName(self):
        return self.__loss_fn.getName()
    
    def getType(self):
        return "BackPropagation"
    
    def evaluate(self,layers, inputs, targets):
        """
        Avalia a acurácia da rede neural, comparando a saída prevista com os rótulos reais.

        Parâmetros:
        - layers: lista de camadas da rede
        - inputs: dados de entrada para avaliação
        - targets: rótulos reais esperados

        Saída: precisão média (acurácia)
        """
        output = inputs
        for layer in layers:
            output = layer.forward(output)
        predicted = np.argmax(output, axis=1)
        actual = np.argmax(targets, axis=1)
        return np.mean(predicted == actual)
    
    # Garante  estratificação dos dados
    def separate_by_class(self, X, y):
        """
        Separa os dados por classe para posterior uso em validação cruzada e parada antecipada.

        Parâmetros:
        - X: dados de entrada
        - y: rótulos (one-hot encoded)

        Saída: dicionário com dados separados por classe
        """
        combined = np.concatenate((X, y), axis=1)
        class_data = {}
        for row in combined:
            features, label = row[:-y.shape[1]], row[-y.shape[1]:]
            key = tuple(label)
            class_data.setdefault(key, []).append([features, label])
        return class_data

class BackPropagationCV: #BackPropagation com Validação Cruzada
    def __init__(self, loss, learning_rate=0.01, epochs=1000):
        """
        Inicializa o processo de Backpropagation com Validação Cruzada, utilizando a função de perda, taxa de aprendizado e número de épocas.

        Parâmetros:
        - loss: função de perda a ser utilizada
        - learning_rate: taxa de aprendizado para atualização dos pesos
        - epochs: número de épocas para treinamento

        Saída: nenhuma (construtor)
        """
        self.__trainer = BackPropagation(loss, learning_rate, epochs)
    
    def evaluate(self,layers, inputs, targets):
        """
        Avalia a acurácia da rede neural utilizando validação cruzada.

        Parâmetros:
        - layers: lista de camadas da rede
        - inputs: dados de entrada para avaliação
        - targets: rótulos reais esperados

        Saída: precisão média (acurácia)
        """
        return self.__trainer.evaluate(layers, inputs, targets)
     
    def separate_by_class(self, X, y):
        """
        Separa os dados por classe para posterior uso em validação cruzada.

        Parâmetros:
        - X: dados de entrada
        - y: rótulos (one-hot encoded)

        Saída: dicionário com dados separados por classe
        """
        return self.__trainer.separate_by_class(X,y)

    def get_learning_rate(self):
        return self.__trainer.get_learning_rate()
    
    def get_epochs(self):
        return self.__trainer.get_epochs()
    
    def get_lossName(self):
        return self.__trainer.get_lossName()
    
    def getType(self):
        return "BackPropagationCV"

    def train(self,layers, X, y, k=5):
        """
        Realiza o treinamento utilizando validação cruzada com k-folds.

        Parâmetros:
        - layers: lista de camadas da rede
        - X: dados de entrada para treinamento
        - y: rótulos ou saídas esperadas
        - k: número de folds para a validação cruzada

        Saída: resultados de cada fold com previsões e rótulos esperados
        """
        class_data = self.separate_by_class(X, y)
        folds = [[] for _ in range(k)]

        # Distribuir amostras nos folds
        for samples in class_data.values():
            random.shuffle(samples)
            for idx, sample in enumerate(samples):
                folds[idx % k].append(sample)
        results = []
        for i in range(k):
            valid = folds[i]
            train = sum(folds[:i] + folds[i+1:], [])

            X_train = np.array([s[0] for s in train])
            y_train = np.array([s[1] for s in train])
            X_valid = np.array([s[0] for s in valid])
            y_valid = np.array([s[1] for s in valid])

            self.__trainer.train(layers, X_train, y_train)

            output = X_valid
            for layer in layers:
                output = layer.forward(output)
            results.append({
                'predictions': output,
                'targets': y_valid
            })

            if i != k - 1:
                for layer in layers:
                    layer.reset()
        return results

class BackPropagationES: #BackPropagation com Early Stopping
    def __init__(self, loss, learning_rate=0.01, epochs=200):
        """
        Inicializa o processo de Backpropagation com Early Stopping, utilizando a função de perda, taxa de aprendizado e número de épocas.

        Parâmetros:
        - loss: função de perda a ser utilizada
        - learning_rate: taxa de aprendizado para atualização dos pesos
        - epochs: número de épocas para treinamento

        Saída: nenhuma (construtor)
        """
        self.__lr = learning_rate
        self.__epochs = epochs
        self.__loss = loss

    def evaluate(self,layers, inputs, targets):
        """
        Avalia a acurácia da rede neural com early stopping, comparando a saída prevista com os rótulos reais.

        Parâmetros:
        - layers: lista de camadas da rede
        - inputs: dados de entrada para avaliação
        - targets: rótulos reais esperados

        Saída: precisão média (acurácia)
        """
        output = inputs
        for layer in layers:
            output = layer.forward(output)
        predicted = np.argmax(output, axis=1)
        actual = np.argmax(targets, axis=1)
        return np.mean(predicted == actual)
     
    def separate_by_class(self, X, y):
        """
        Separa os dados por classe para posterior uso em early stopping.

        Parâmetros:
        - X: dados de entrada
        - y: rótulos (one-hot encoded)

        Saída: dicionário com dados separados por classe
        """
        combined = np.concatenate((X, y), axis=1)
        class_data = {}
        for row in combined:
            features, label = row[:-y.shape[1]], row[-y.shape[1]:]
            key = tuple(label)
            class_data.setdefault(key, []).append([features, label])
        return class_data

    def get_learning_rate(self):
        return self.__lr
    
    def get_epochs(self):
        return self.__epochs
    
    def get_lossName(self):
        return self.__loss.getName()
    
    def getType(self):
        return "BackPropagationES"
    
    def train(self,layers, X, y, k=None):
        """
        Realiza o treinamento utilizando o método de Backpropagation com Early Stopping.

        Parâmetros:
        - layers: lista de camadas da rede
        - X: dados de entrada para treinamento
        - y: rótulos ou saídas esperadas
        - k: número de folds (caso se deseje validação cruzada)

        Saída: resultados do treinamento com a melhor acurácia
        """
        class_data = self.separate_by_class(X, y)
        train_data, valid_data = [], []

        for samples in class_data.values():
            random.shuffle(samples)
            n_val = max(1, int(0.1 * len(samples)))
            valid_data += samples[:n_val]
            train_data += samples[n_val:]

        X_train = np.array([s[0] for s in train_data])
        y_train = np.array([s[1] for s in train_data])
        X_valid = np.array([s[0] for s in valid_data])
        y_valid = np.array([s[1] for s in valid_data])

        best_acc = -1
        best_epoch = 0
        patience = self.get_epochs()
        epoch = 0

        print('\nPESOS INICIAIS')
        for layer in layers:
            print(layer.getWeights())
            print('\n')
        print('\n')

        while patience > 0:
            output = X_train
            for layer in layers:
                output = layer.forward(output)

            # Calcula loss
            loss = self.__loss.forward(output, y_train)

            # Inicia backward pass com gradiente da loss
            dvalues = self.__loss.backward(output, y_train)
            
            # Backprop em cada layer (em ordem reversa)
            for layer in reversed(layers):
                dvalues = layer.backward(dvalues, self.get_learning_rate())

            epoch += 1

            acc = self.evaluate(layers, X_valid, y_valid)

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch} - Loss: {loss:.6f}   Accuracy: {acc:.4f}")

                print('\nPESOS')
                for layer in layers:
                    print(layer.getWeights())
                    print('\n')
                print('\n')

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                patience = self.get_epochs()

                w = []
                for layer in layers:
                    weights = layer.getWeights()
                    w.append(copy.deepcopy(weights))
            else:
                patience -= 1
            
        print('\nPESOS FINAIS')
        for layer in layers:
            print(layer.getWeights())
            print('\n')
        print('\n')

        print(f"Melhor acurácia\nÉpoca {best_epoch}: {best_acc:.4f}")
        
        for i in range(len(layers)):
            layers[i].setWeights(w[i])