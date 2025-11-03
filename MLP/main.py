import numpy as np
from .mlp import MLP
from .trainer import *
from .losses.loss_crossentropy import LossCrossEntropy
from .activation_functions.ReLU import ReLU
from .activation_functions.softmax import Softmax
from .confusion_matrix import get_confusion_matrix
import os

np.random.seed(42)

# Caminhos para os arquivos
base_dir = os.path.dirname(__file__)
X_txt_path = os.path.join(base_dir, "..", "CARACTERES_COMPLETO", "X.txt")
Y_txt_path = os.path.join(base_dir, "..", "CARACTERES_COMPLETO", "Y_letra.txt")

# Carrega X do .txt manualmente, ignorando valores vazios
X = []
with open(X_txt_path, "r") as f:
    for linha in f:
        valores = [val.strip() for val in linha.strip().split(",") if val.strip() != ""]
        if valores:  # ignora linhas totalmente vazias
            X.append([float(v) for v in valores])

X = np.array(X, dtype=np.float32)

# Verifica se todas as linhas têm o mesmo tamanho
comprimentos = [len(linha) for linha in X]
if len(set(comprimentos)) > 1:
    raise ValueError(f"Linhas com tamanhos diferentes detectadas: {set(comprimentos)}")

# Normaliza X (média 0, desvio 1), com segurança contra divisão por zero
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-7)

# Carrega Y e converte para índices (A=0, ..., Z=25)
with open(Y_txt_path, "r") as f:
    letras = [linha.strip() for linha in f]
letra_to_index = {chr(i + 65): i for i in range(26)}
y_raw = np.array([letra_to_index[l] for l in letras])
n_classes = 26

# One-hot encoding
y = np.zeros((len(y_raw), n_classes))
y[np.arange(len(y_raw)), y_raw] = 1

# Define arquitetura da MLP
n_inputs = X.shape[1]
configs = [
    (64, n_inputs, ReLU()),
    (n_classes, 64, Softmax())
]

# Define conjunto de treino e teste
X_train = X[:-130]
y_train = y[:-130]
X_test = X[-130:]
y_test = y[-130:]

# trainer = BackPropagation(LossCrossEntropy(), learning_rate=0.05, epochs=1000)
# mlp = MLP(configs, trainer)

mlp = MLP.load_model('best_model.json')

# Treinamento
mlp.train(X_train, y_train)

# Avaliação
predictions = mlp.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
real_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == real_classes)
print(f"Acurácia final: {accuracy:.4f}")

labels = np.unique(letras)
get_confusion_matrix(predictions, y_test, labels) #Matriz de confusão
mlp.save("MLP/models/mlp.model") #salva modelo
