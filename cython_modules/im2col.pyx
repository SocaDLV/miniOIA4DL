# --- INICIO BLOQUE GENERADO CON IA ---
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def im2col_forward_cython(np.ndarray[np.float32_t, ndim=4] input_pad, 
                          int kernel_size, int stride, int out_h, int out_w):
    
    cdef int B = input_pad.shape[0]
    cdef int C = input_pad.shape[1]
    
    cdef int col_rows = C * kernel_size * kernel_size
    cdef int col_cols = out_h * out_w
    
    cdef np.ndarray[np.float32_t, ndim=3] cols = np.zeros((B, col_rows, col_cols), dtype=np.float32)
    
    # Declaramos las variables iteradoras explícitamente para que cada hilo tenga las suyas
    cdef int b, c, i, j, kh, kw
    cdef int col_idx, row_idx
    cdef int h_start, w_start
    
    # ¡LA MAGIA DE HPC! prange divide el batch (B) entre los núcleos de la CPU.
    # nogil=True desconecta las protecciones de Python para permitir multihilo real en C
    for b in prange(B, nogil=True):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                
                # Calculamos el índice directo en lugar de sumarlo, vital para paralelismo
                col_idx = i * out_w + j 
                
                for c in range(C):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            row_idx = c * (kernel_size * kernel_size) + kh * kernel_size + kw
                            cols[b, row_idx, col_idx] = input_pad[b, c, h_start + kh, w_start + kw]
                
    return cols
# --- FIN BLOQUE GENERADO CON IA ---