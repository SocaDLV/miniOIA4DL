from modules.layer import Layer
import numpy as np

# --- INICIO BLOQUE GENERADO CON IA ---
try:
    import numexpr as ne
    NUMEXPR_AVAILABLE = True
except ImportError:
    NUMEXPR_AVAILABLE = False
# --- FIN BLOQUE GENERADO CON IA ---

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, x, training=True):
        self.input = np.array(x, dtype=np.float32)
        
        # --- INICIO BLOQUE GENERADO CON IA ---
        if NUMEXPR_AVAILABLE:
            # NumExpr compila la expresión matemática y la ejecuta en C++ usando múltiples hilos
            inp = self.input
            # Equivalente a np.maximum(0, inp) pero hiper-optimizado
            return ne.evaluate("where(inp > 0, inp, 0)")
        else:
            return np.maximum(0, self.input)
        # --- FIN BLOQUE GENERADO CON IA ---

    def backward(self, grad_output, learning_rate=None):
        # --- INICIO BLOQUE GENERADO CON IA ---
        if NUMEXPR_AVAILABLE:
            inp = self.input
            go = grad_output
            return ne.evaluate("go * (inp > 0)")
        else:
            return grad_output * (self.input > 0)
        # --- FIN BLOQUE GENERADO CON IA ---