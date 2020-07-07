import ctypes

# Load the shared library into c types.
libc = ctypes.CDLL("./libchwe.so")

alloc_func = libc.alloc_C_string
alloc_func.restype = ctypes.POINTER(ctypes.c_char)