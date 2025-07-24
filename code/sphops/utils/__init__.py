import functools
import torch
from numba import cuda

@functools.cache
def dot_nd(DIM):
    @cuda.jit(device=True)
    def dot(v1, v2):
        result = 0
        for i in range(DIM):
            result += v1[i] * v2[i]
        return result
    return dot

@functools.cache
def matvecmul_nd(DIM):
    @cuda.jit(device=True)
    def matvecmul(m, v, out):
        for i in range(DIM):
            for j in range(DIM):
                out[i] += m[i,j] * v[j] # intentional sum!
    return matvecmul

def profiled_function(f):
    @functools.wraps(f)
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(f.__name__):
            return f(*args, **kwargs)
    return decorator

def pack(*xx):
    packed_x = torch.cat(xx, dim=0)
    sections = [x.shape[0] for x in xx]

    return packed_x, sections

def unpack(packed_x, sections):
    return torch.split(packed_x, sections)
