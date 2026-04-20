# --- INICIO BLOQUE GENERADO CON IA ---
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def maxpool_forward_cython(np.ndarray[np.float32_t, ndim=4] input_array,
                           int kernel_size, int stride, int out_h, int out_w):
    
    cdef int B = input_array.shape[0]
    cdef int C = input_array.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=4] output = np.zeros((B, C, out_h, out_w), dtype=np.float32)
    
    cdef int b, c, i, j, kh, kw
    cdef int h_start, w_start
    cdef float max_val, current_val
    
    # Multiproceso con prange
    for b in prange(B, nogil=True):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    
                    max_val = input_array[b, c, h_start, w_start]
                    
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            current_val = input_array[b, c, h_start + kh, w_start + kw]
                            if current_val > max_val:
                                max_val = current_val
                                
                    output[b, c, i, j] = max_val
                    
    return output
# --- FIN BLOQUE GENERADO CON IA ---