import functools
import numpy as np

from . import kernels_impl as impl

SMOOTHING_KERNEL = 'poly6'
GRADIENT_KERNEL = 'spiky'


@functools.cache
def count_kernel_nd(DIM, FLOAT=np.float32):
    return impl.count_nd(DIM, FLOAT)

@functools.cache
def smoothing_kernel_normalization_nd(DIM, FLOAT=np.float32):
    assert SMOOTHING_KERNEL in ['poly6', 'wendlandC2', 'wendlandC4']
    return getattr(impl, f'smoothing_{SMOOTHING_KERNEL}_normalization_nd')(DIM, FLOAT)

@functools.cache
def smoothing_kernel_nd(DIM, FLOAT=np.float32):
    assert SMOOTHING_KERNEL in ['poly6', 'wendlandC2', 'wendlandC4']
    return getattr(impl, f'smoothing_{SMOOTHING_KERNEL}_nd')(DIM, FLOAT)

@functools.cache
def smoothing_kernel_dr_nd(DIM, FLOAT=np.float32):
    assert SMOOTHING_KERNEL in ['poly6', 'wendlandC2', 'wendlandC4']
    return getattr(impl, f'smoothing_{SMOOTHING_KERNEL}_dr_nd')(DIM, FLOAT)


@functools.cache
def gradient_kernel_normalization_nd(DIM, FLOAT=np.float32):
    assert GRADIENT_KERNEL in ['spiky']
    return getattr(impl, f'gradient_{GRADIENT_KERNEL}_normalization_nd')(DIM, FLOAT)

@functools.cache
def gradient_kernel_nd(DIM, FLOAT=np.float32):
    assert GRADIENT_KERNEL in ['spiky']
    return getattr(impl, f'gradient_{GRADIENT_KERNEL}_nd')(DIM, FLOAT)

@functools.cache
def gradient_kernel_dr_nd(DIM, FLOAT=np.float32):
    assert GRADIENT_KERNEL in ['spiky']
    return getattr(impl, f'gradient_{GRADIENT_KERNEL}_dr_nd')(DIM, FLOAT)

