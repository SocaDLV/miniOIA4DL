from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        # --- INICIO BLOQUE GENERADO CON IA ---
        # 1. Crear vista vectorizada de las ventanas con as_strided
        s0, s1, s2, s3 = input.strides
        view_shape = (B, C, out_h, out_w, KH, KW)
        view_strides = (s0, s1, s2 * SH, s3 * SW, s2, s3)
        
        strided_input = np.lib.stride_tricks.as_strided(input, shape=view_shape, strides=view_strides)
        
        # 2. Obtener el valor máximo en las dos últimas dimensiones (la ventana)
        output = np.max(strided_input, axis=(4, 5))
        
        # 3. Calcular los índices vectorizados (necesario si luego se llama a backward)
        # Aplanamos la ventana de (KH, KW) a (KH * KW)
        strided_input_flat = strided_input.reshape(B, C, out_h, out_w, -1)
        max_idx_flat = np.argmax(strided_input_flat, axis=4) # Índice relativo (1D) dentro de la ventana
        
        # Convertir índice relativo 1D a coordenadas relativas 2D (h, w)
        max_idx_h = max_idx_flat // KW
        max_idx_w = max_idx_flat % KW
        
        # Crear mallas (grids) base para sumar las posiciones absolutas
        grid_h = np.arange(out_h) * SH
        grid_w = np.arange(out_w) * SW
        
        # Expandir con broadcasting para poder sumar
        grid_h = grid_h.reshape(1, 1, out_h, 1)
        grid_w = grid_w.reshape(1, 1, 1, out_w)
        
        # Coordenadas absolutas = coordenadas base + coordenadas relativas
        abs_idx_h = grid_h + max_idx_h
        abs_idx_w = grid_w + max_idx_w
        
        # Guardar en el formato original
        self.max_indices = np.stack((abs_idx_h, abs_idx_w), axis=-1)
        # --- FIN BLOQUE GENERADO CON IA ---

        return output

    def backward(self, grad_output, learning_rate=None):
        # --- INICIO BLOQUE GENERADO CON IA ---
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        # Crear índices para las dimensiones de Batch y Channel (broadcasting)
        batch_grid = np.arange(B).reshape(B, 1, 1, 1)
        channel_grid = np.arange(C).reshape(1, C, 1, 1)
        
        # Extraer los índices absolutos de alto y ancho que calculamos en el forward
        h_indices = self.max_indices[..., 0]
        w_indices = self.max_indices[..., 1]
        
        # Usar np.add.at para enviar cada gradiente a su píxel de origen
        # Es mucho más rápido que un bucle y soporta superposición segura (overlapping)
        np.add.at(grad_input, (batch_grid, channel_grid, h_indices, w_indices), grad_output)

        return grad_input
        # --- FIN BLOQUE GENERADO CON IA ---