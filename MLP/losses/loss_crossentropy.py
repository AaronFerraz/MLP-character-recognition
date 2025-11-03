# losses/loss_crossentropy.py
import numpy as np

class LossCrossEntropy:
    __instance = None

    def __new__(cls): #Design Patterns Singleton
        if cls.__instance is None:
            cls.__instance = super(LossCrossEntropy, cls).__new__(cls)
        return cls.__instance
    
    """Cross-Entropy Loss para classificação"""

    def forward(self, predictions, targets):
        """Calcula o erro"""
        # Clip para evitar log(0)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        # Se targets estiver one-hot
        if len(targets.shape) == 2:
            correct_confidences = np.sum(predictions_clipped * targets, axis=1)
        else: # targets como rótulos inteiros
            correct_confidences = predictions_clipped[range(len(predictions_clipped)), targets]

        # Negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

    def backward(self, predictions, targets):
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        grad = -targets / predictions_clipped
        return grad / predictions.shape[0]

    def getName(self):
        return "crossentropy"
