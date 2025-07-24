import functools
import numpy as np
from numba import cuda

from ..utils import dot_nd

sqrtf = cuda.libdevice.sqrtf


@functools.cache
def count_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)
    
    @cuda.jit(device=True)
    def kernel(r, h) -> int:
        d2 = dot(r, r)
        return d2 < h ** 2
    return kernel

@functools.cache
def smoothing_poly6_normalization_nd(DIM, FLOAT=np.float32):
    if DIM == 2:
        def dim2(h: FLOAT) -> FLOAT:
            return 4. / (np.pi * (h ** 8))
        return dim2
    if DIM == 3:
        def dim3(h: FLOAT) -> FLOAT:
            return 315. / (64. * np.pi * (h ** 9))
        return dim3
    raise NotImplementedError()

@functools.cache
def smoothing_poly6_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)
    
    @cuda.jit(device=True)
    def kernel(r, h) -> FLOAT:
        d2 = dot(r, r)
        return max((h ** 2 - d2) ** 3, 0.0)
    return kernel

@functools.cache
def smoothing_poly6_dr_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)
    
    @cuda.jit(device=True)
    def kernel(r, h, out, mult):
        d2 = dot(r, r)
        h2 = h ** 2
        mag = (d2 < h2) and mult * -6. * ((h2 - d2) ** 2)
        for d in range(DIM):
            out[d] += mag * r[d] # intended +=
    return kernel

#### https://pysph.readthedocs.io/en/latest/reference/kernels.html#pysph.base.kernels.WendlandQuintic
@functools.cache
def smoothing_wendlandC2_normalization_nd(DIM, FLOAT=np.float32):
    if DIM == 2:
        def dim2(h: FLOAT) -> FLOAT:
            return 7. / (np.pi * h ** 2)
        return dim2
    elif DIM == 3:
        def dim3(h: FLOAT) -> FLOAT:
            return 21. / (2 * np.pi * h ** 3)
        return dim3
    raise NotImplementedError()

@functools.cache
def smoothing_wendlandC2_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)

    @cuda.jit(device=True)
    def kernel(r, h) -> FLOAT:
        d2 = dot(r, r)
        q = sqrtf(d2 / h ** 2)
        return (q < 1) and (1-q)**4 * (4*q + 1)
    return kernel

@functools.cache
def smoothing_wendlandC2_dr_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)
    
    @cuda.jit(device=True)
    def kernel(r, h, out, mult):
        d2 = dot(r, r)
        q = sqrtf(d2 / h ** 2)
        mag = (q < 1) and mult * -20. * q * (1-q)**3 / h
        for d in range(DIM):
            out[d] += mag * r[d] # intended +=
    return kernel

#### https://pysph.readthedocs.io/en/latest/reference/kernels.html#pysph.base.kernels.WendlandQuinticC4
@functools.cache
def smoothing_wendlandC4_normalization_nd(DIM, FLOAT=np.float32):
    if DIM == 2:
        def dim2(h: FLOAT) -> FLOAT:
            return 9. / (np.pi * h ** 2)
        return dim2
    elif DIM == 3:
        def dim3(h: FLOAT) -> FLOAT:
            return 495. / (32 * np.pi * h ** 3)
        return dim3
    raise NotImplementedError()

@functools.cache
def smoothing_wendlandC4_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)

    @cuda.jit(device=True)
    def kernel(r, h) -> FLOAT:
        d2 = dot(r, r)
        q2 = d2 / h ** 2
        q = sqrtf(q2)
        return (q < 1) and (1-q)**6 * (35*q2 + 18*q + 3) / 3
    return kernel

@functools.cache
def smoothing_wendlandC4_dr_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)
    
    @cuda.jit(device=True)
    def kernel(r, h, out, mult):
        d2 = dot(r, r)
        q = sqrtf(d2 / h ** 2)
        mag = (q < 1) and mult * -56. * q * (1-q)**5 * (1+5*q) / h / 3
        for d in range(DIM):
            out[d] += mag * r[d] # intended +=
    return kernel

### Spiky kernel
@functools.cache
def gradient_spiky_normalization_nd(DIM, FLOAT=np.float32):
    if DIM == 2:
        def dim2(h: FLOAT) -> FLOAT:
            return 10. / (np.pi * (h ** 5))
        return dim2
    if DIM == 3:
        def dim3(h: FLOAT) -> FLOAT:
            return 15. / (np.pi * (h ** 6))
        return dim3
    raise NotImplementedError()

@functools.cache
def gradient_spiky_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)
    
    @cuda.jit(device=True, fastmath=True)
    def kernel(r, h, out, mult=1.0):
        d2 = dot(r,r)
        d = cuda.libdevice.sqrtf(d2)
        di = cuda.libdevice.rsqrtf(d2)
    
        mag = (h - d > 0) and mult * 3. * ((h-d) ** 2) * di
        for i in range(DIM):
            out[i] = r[i] and mag * r[i]
    return kernel

@functools.cache
def gradient_spiky_dr_nd(DIM, FLOAT=np.float32):
    dot = dot_nd(DIM)
    
    @cuda.jit(device=True, fastmath=True)
    def kernel(r, h, out):
        d2 = dot(r,r)
        d = cuda.libdevice.sqrtf(d2)
        di = cuda.libdevice.rsqrtf(d2)
    
        if 0 < d < h:
            hd = h - d
            u = cuda.local.array(DIM, np.float32)
            for d in range(DIM):
                u[d] = -r[d] * di
            mag1 = -6. * hd
            mag2 = 3. * (hd ** 2) * di
            for i in range(DIM):
                for j in range(DIM):
                    uut = u[i]*u[j]
                    out[i,j] = mag1*uut + mag2*((i == j) - uut)
    return kernel
