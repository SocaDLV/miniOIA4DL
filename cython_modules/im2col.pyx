# --- INICIO BLOQUE GENERADO CON IA ---
import numpy as np
cimport numpy as np
cimport cython

# Desactivamos comprobaciones de límites para que C vaya al máximo de velocidad
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def im2col_forward_cython(np.ndarray[np.float32_t, ndim=4] input_pad, 
                          int kernel_size, int stride, int out_h, int out_w):
    
    cdef int B = input_pad.shape[0]
    cdef int C = input_pad.shape[1]
    
    cdef int col_rows = C * kernel_size * kernel_size
    cdef int col_cols = out_h * out_w
    
    # Reservamos la memoria para la matriz de columnas de todo el batch
    cdef np.ndarray[np.float32_t, ndim=3] cols = np.zeros((B, col_rows, col_cols), dtype=np.float32)
    
    # Declaración estática de variables iteradoras (crucial en Cython)
    cdef int b, c, i, j, kh, kw
    cdef int col_idx, row_idx
    cdef int h_start, w_start
    
    # Estos bucles anidados ya no duelen porque se compilan a C puro
    for b in range(B):
        col_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                
                row_idx = 0
                for c in range(C):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            cols[b, row_idx, col_idx] = input_pad[b, c, h_start + kh, w_start + kw]
                            row_idx += 1
                col_idx += 1
                
    return cols
# --- FIN BLOQUE GENERADO CON IA ---