import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_confusion_matrix(predictions, targets, class_labels):
    """
    Gera e exibe a matriz de confusão a partir de previsões e rótulos verdadeiros.

    Parâmetros:
    - predictions: array de previsões do modelo
    - targets: array de rótulos reais
    - class_labels: lista com os nomes das classes, usada para rotular os eixos do gráfico

    Saída: exibe a matriz de confusão
    """
    y_true = np.argmax(targets, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    c_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    plt.title("Matriz de Confusão")
    plt.show()