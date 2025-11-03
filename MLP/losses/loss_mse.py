import numpy as np

class LossMSE:
    __instance = None

    def __new__(cls): #Design Patterns Singleton
        if cls.__instance is None:
            cls.__instance = super(LossMSE, cls).__new__(cls)
        return cls.__instance

    """Mean Squared Error"""
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets) ** 2)

    def backward(self, predictions, targets):
        # derivada MSE: 2*(pred-target)/n
        samples = targets.shape[0]
        dvalues = 2 * (predictions - targets) / samples
        return dvalues
    
    def getName(self):
        return "mse"