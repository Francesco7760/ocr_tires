import numpy as np

## bytorder
## '>' big-endian -> i byte vengo letti partendo dal più significativo (da sinistra a destra)
## '<' little-endian -> i byte vengo letti partenda dal meno significativo (da destra a sinistra)

int8 = np.dtype('>i1') # 8-bit signed integer
uint8 = np.dtype('>u1') # 8-bit unsigned integer
int16 = np.dtype('>i2') # 16-bit signed integer
uint16 = np.dtype('>u2') # 16-bit unsigned integer
int32 = np.dtype('>i4') # 32-bit signed integer
uint32 = np.dtype('>u4') # 32-bit unsigned integer
int64 = np.dtype('>i8') # 64-bit signed integer
uint64 = np.dtype('>u8') # 64-bit unsigned integer
float16 = np.dtype('>f2') # 16-bit floating-point number
float32 = np.dtype('>f4') # 32-bit floating-point number
float64 = np.dtype('>f8') # 64-bit floating-point number
string1 = np.dtype('>S1') # 1-character (8-bit) string