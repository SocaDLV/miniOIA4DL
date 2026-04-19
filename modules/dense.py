from modules.utils import *
from modules.layer import Layer

import numpy as np

class Dense(Layer):
    def __init__(self, in_features, out_features,weight_init="he"):
        self.in_features = in_features
        self.out_features = out_features

        if weight_init == "he":
            std = np.sqrt(2.0 / in_features)
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        elif weight_init == "custom":
            self.weights = np.zeros((in_features, out_features), dtype=np.float32)
        else:
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * (1 / in_features**0.5)

        self.biases = np.zeros(out_features, dtype=np.float32)

        self.input = None

    def forward(self, input, training=True):  # input: [batch_size x in_features]
        self.input = np.array(input).astype(np.float32)  # Ensure input is float for numerical stability
        batch_size = self.input.shape[0]

        # --- INICIO BLOQUE GENERADO CON IA ---
        # Sustituimos la función lenta matmul_biasses por operaciones vectorizadas de NumPy (BLAS)
        output = np.dot(self.input, self.weights) + self.biases
        # --- FIN BLOQUE GENERADO CON IA ---
        
        self.output = output
        return output

    def backward(self, grad_output, learning_rate):
        grad_output = np.array(grad_output).astype(np.float32)  # Ensure grad_output is float for numerical stability
        batch_size = grad_output.shape[0]

        # --- INICIO BLOQUE GENERADO CON IA ---
        # Reemplazamos los triples bucles anidados por multiplicaciones matriciales
        
        # Gradient w.r.t. weights: (in_features, batch_size) @ (batch_size, out_features)
        grad_weights = np.dot(self.input.T, grad_output)
        
        # Gradient w.r.t. biases: Suma a lo largo del eje del batch
        grad_biases = np.sum(grad_output, axis=0)

        # Gradient w.r.t. input: (batch_size, out_features) @ (out_features, in_features)
        grad_input = np.dot(grad_output, self.weights.T)
        # --- FIN BLOQUE GENERADO CON IA ---
        
        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
    
    def get_weights(self):
        return {'weights': self.weights, 'biases': self.biases}

    def set_weights(self, weights):
        self.weights = weights['weights']
        self.biases = weights['biases']