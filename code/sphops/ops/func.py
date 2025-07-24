import math
import torch
import numpy as np

from numba import cuda

from . import operators_batch as ops

MAX_FEATURES = 64
THREADS_PER_BLOCK = 64

NP_FLOAT_TO_TORCH_FLOAT = {
    torch.float16 : np.float16,
    torch.float32 : np.float32,
    torch.float64 : np.float64,
}


class Volume(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, packed_x, h, grid):
        """
        packed_x: f[N,D]
        h: float / f[1]
        grid: HashGrid
        """
        total_points, DIM = packed_x.shape
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_x.dtype]
        
        ctx.backprop_dLdx = packed_x.requires_grad
        ctx.h = h.item() if isinstance(h, torch.Tensor) else h
        ctx.grid = grid
        ctx.normalization = ops.volume_normalization(DIM, ctx.h)
        
        packed_v = torch.zeros((total_points,), dtype=packed_x.dtype, device=packed_x.device)
        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()

        ops.volume_forward_nd(DIM, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
            packed_x.detach(), packed_v, ctx.h, 
            grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
            num_blocks, num_points, num_cells, ctx.normalization
        )

        cuda.synchronize()
        ctx.save_for_backward(packed_x, packed_v) # Only for volumes!
        return packed_v

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.no_grad()
    def backward(ctx, packed_dLdv):
        packed_dLdx = dLdh = dLdgrid = None
        
        packed_x, packed_v = ctx.saved_tensors

        total_points, DIM = packed_x.shape
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_x.dtype]

        backprop_dLdx = ctx.backprop_dLdx
        h = ctx.h
        grid = ctx.grid

        if backprop_dLdx:
            packed_dLdx = torch.zeros_like(packed_x)
            # dLdx = unpack(packed_dLdx, sections)

        EMPTY = torch.empty([], device=packed_x.device)
        dLdx_in = EMPTY if packed_dLdx is None else packed_dLdx

        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()
        ops.volume_backward_nd(
            DIM, backprop_dLdx, FLOAT=ftype
        )[total_blocks, THREADS_PER_BLOCK](
            packed_x, packed_v, packed_dLdv,
            dLdx_in, h,
            grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
            num_blocks, num_points, num_cells, ctx.normalization
        )
        
        cuda.synchronize()
        return packed_dLdx, dLdh, dLdgrid


class Gradient(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, packed_x, packed_v, packed_A, h, grid):
        """
        packed_x: f[N,D]
        packed_v: f[N]
        packed_A: f[N,F]
        h: float / f[1]
        grid: HashGrid
        """
        total_points, DIM = packed_x.shape
        total_points_A, num_features = packed_A.shape
        assert total_points_A == total_points
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_A.dtype]
        
        ctx.backprop_dLdx = packed_x.requires_grad
        ctx.backprop_dLdA = packed_A.requires_grad
        
        ctx.h = h.item() if isinstance(h, torch.Tensor) else h
        ctx.grid = grid
        ctx.normalization = ops.gradient_normalization(DIM, ctx.h)

        packed_GA = torch.zeros((total_points, num_features, DIM), dtype=packed_A.dtype, device=packed_A.device)

        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()
        if num_features <= MAX_FEATURES:
            ops.gradient_forward_nd(DIM, num_features, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
                packed_x.detach(), packed_v, packed_A.detach(), 
                packed_GA, 0, ctx.h, 
                grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                num_blocks, num_points, num_cells, ctx.normalization
            )
        else:
            num_iter = math.ceil(num_features / MAX_FEATURES)
            for i in range(num_iter):
                f_start = i * MAX_FEATURES
                f_end = min(num_features, (i+1) * MAX_FEATURES)
                num_features_batch = f_end - f_start
                
                ops.gradient_forward_nd(DIM, num_features_batch, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
                    packed_x.detach(), packed_v, packed_A.detach(), 
                    packed_GA, f_start, ctx.h, 
                    grid.dims, grid.idx, grid.cell_start, grid.cell_end,
                    num_blocks, num_points, num_cells, ctx.normalization
                )

        cuda.synchronize()
        ctx.save_for_backward(packed_x, packed_v, packed_A)
        return packed_GA

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.no_grad()
    def backward(ctx, packed_dLdGA):
        packed_dLdx = packed_dLdv = packed_dLdA = dLdh = dLdgrid = None
        grid = ctx.grid
        packed_x, packed_v, packed_A = ctx.saved_tensors

        total_points, DIM = packed_x.shape
        total_points_A, num_features = packed_A.shape
        assert total_points_A == total_points
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_x.dtype]

        backprop_dLdx = ctx.backprop_dLdx
        backprop_dLdA = ctx.backprop_dLdA

        if backprop_dLdx:
            packed_dLdx = torch.zeros_like(packed_x)
        if backprop_dLdA:
            packed_dLdA = torch.zeros_like(packed_A)

        EMPTY = torch.empty([], device=packed_x.device)
        dLdx_in = EMPTY if packed_dLdx is None else packed_dLdx
        dLdA_in = EMPTY if packed_dLdA is None else packed_dLdA

        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()
        if num_features <= MAX_FEATURES:
            ops.gradient_backward_nd(
                DIM, num_features, backprop_dLdx, backprop_dLdA, FLOAT=ftype
            )[total_blocks, THREADS_PER_BLOCK](
                packed_x.detach(), packed_v, packed_A.detach(), packed_dLdGA.detach(),
                dLdx_in, dLdA_in, 0, ctx.h,
                grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                num_blocks, num_points, num_cells, ctx.normalization
            )
        else:
            num_iter = math.ceil(num_features / MAX_FEATURES)
            for i in range(num_iter):
                f_start = i * MAX_FEATURES
                f_end = min(num_features, (i+1) * MAX_FEATURES)
                num_features_batch = f_end - f_start
                
                ops.gradient_backward_nd(
                    DIM, num_features_batch, backprop_dLdx, backprop_dLdA, FLOAT=ftype
                )[total_blocks, THREADS_PER_BLOCK](
                    packed_x.detach(), packed_v, packed_A.detach(), packed_dLdGA.detach(),
                    dLdx_in, dLdA_in, f_start, ctx.h,
                    grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                    num_blocks, num_points, num_cells, ctx.normalization
                )

        cuda.synchronize()
        return packed_dLdx, packed_dLdv, packed_dLdA, dLdh, dLdgrid

class Divergence(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, packed_x, packed_v, packed_A, h, grid):
        """
        packed_x: f[N,D]
        packed_v: f[N]
        packed_A: f[N,F,D]
        h: float / f[1]
        grid: HashGrid
        """
        # batch_size = len(sections)
        total_points, DIM = packed_x.shape
        total_points_A, num_features, DIM_A = packed_A.shape
        assert total_points_A == total_points and DIM_A == DIM
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_A.dtype]
        
        ctx.backprop_dLdx = packed_x.requires_grad
        ctx.backprop_dLdA = packed_A.requires_grad
        
        ctx.h = h.item() if isinstance(h, torch.Tensor) else h
        ctx.grid = grid
        ctx.normalization = ops.gradient_normalization(DIM, ctx.h)

        packed_DA = torch.zeros((total_points, num_features), dtype=packed_A.dtype, device=packed_A.device)

        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()
        if num_features <= MAX_FEATURES:
            ops.divergence_forward_nd(DIM, num_features, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
                packed_x.detach(), packed_v, packed_A.detach(), 
                packed_DA, 0, ctx.h, 
                grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                num_blocks, num_points, num_cells, ctx.normalization
            )
        else:
            num_iter = math.ceil(num_features / MAX_FEATURES)
            for i in range(num_iter):
                f_start = i * MAX_FEATURES
                f_end = min(num_features, (i+1) * MAX_FEATURES)
                num_features_batch = f_end - f_start
                
                ops.divergence_forward_nd(DIM, num_features_batch, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
                    packed_x.detach(), packed_v, packed_A.detach(), 
                    packed_DA, f_start, ctx.h, 
                    grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                    num_blocks, num_points, num_cells, ctx.normalization
                )

        cuda.synchronize()
        ctx.save_for_backward(packed_x, packed_v, packed_A)
        return packed_DA

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.no_grad()
    def backward(ctx, packed_dLdDA):
        packed_dLdx = packed_dLdv = packed_dLdA = dLdh = dLdgrid = None
        grid = ctx.grid
        packed_x, packed_v, packed_A = ctx.saved_tensors

        total_points, DIM = packed_x.shape
        total_points_A, num_features, DIM_A = packed_A.shape
        assert total_points_A == total_points and DIM_A == DIM
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_x.dtype]

        backprop_dLdx = ctx.backprop_dLdx
        backprop_dLdA = ctx.backprop_dLdA

        if backprop_dLdx:
            packed_dLdx = torch.zeros_like(packed_x)
        if backprop_dLdA:
            packed_dLdA = torch.zeros_like(packed_A)
        
        EMPTY = torch.empty([], device=packed_x.device)
        dLdx_in = EMPTY if packed_dLdx is None else packed_dLdx
        dLdA_in = EMPTY if packed_dLdA is None else packed_dLdA

        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()
        if num_features <= MAX_FEATURES:
            ops.divergence_backward_nd(
                DIM, num_features, backprop_dLdx, backprop_dLdA, FLOAT=ftype
            )[total_blocks, THREADS_PER_BLOCK](
                packed_x.detach(), packed_v, packed_A.detach(), packed_dLdDA.detach(),
                dLdx_in, dLdA_in, 0, ctx.h,
                grid.dims, grid.idx, grid.cell_start, grid.cell_end, num_blocks, num_points, num_cells, ctx.normalization
            )
        else:
            num_iter = math.ceil(num_features / MAX_FEATURES)
            for i in range(num_iter):
                f_start = i * MAX_FEATURES
                f_end = min(num_features, (i+1) * MAX_FEATURES)
                num_features_batch = f_end - f_start
                
                ops.divergence_backward_nd(
                    DIM, num_features_batch, backprop_dLdx, backprop_dLdA, FLOAT=ftype
                )[total_blocks, THREADS_PER_BLOCK](
                    packed_x.detach(), packed_v, packed_A.detach(), packed_dLdDA.detach(),
                    dLdx_in, dLdA_in, f_start, ctx.h,
                    grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                    num_blocks, num_points, num_cells, ctx.normalization
                )
        
        cuda.synchronize()
        return packed_dLdx, packed_dLdv, packed_dLdA, dLdh, dLdgrid


class Count(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, packed_x, h, grid):
        """
        packed_x: f[N,D]
        h: float / f[1]
        grid: HashGrid
        """
        total_points, DIM = packed_x.shape
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_x.dtype]
        
        ctx.backprop_dLdx = packed_x.requires_grad
        ctx.h = h.item() if isinstance(h, torch.Tensor) else h
        ctx.grid = grid
        ctx.normalization = ops.volume_normalization(DIM, ctx.h)
        
        packed_c = torch.zeros((total_points,), dtype=packed_x.dtype, device=packed_x.device)
        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()

        ops.count_forward_nd(DIM, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
            packed_x.detach(), packed_c, ctx.h, 
            grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
            num_blocks, num_points, num_cells, ctx.normalization
        )

        cuda.synchronize()
        ctx.save_for_backward(packed_x)
        return packed_c

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.no_grad()
    def backward(ctx, packed_dLdc):
        packed_dLdx = dLdh = dLdgrid = None
        
        packed_x = ctx.saved_tensors
        packed_dLdx = torch.zeros_like(packed_x)
        return packed_dLdx, dLdh, dLdgrid

class Blur(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, packed_x, packed_v, packed_A, h, grid):
        """
        packed_x: f[N,D]
        packed_v: f[N]
        packed_A: f[N,F]
        h: float / f[1]
        grid: HashGrid
        """
        total_points, DIM = packed_x.shape
        total_points_A, num_features = packed_A.shape
        assert total_points_A == total_points
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_x.dtype]
        
        ctx.backprop_dLdx = packed_x.requires_grad
        ctx.backprop_dLdA = packed_A.requires_grad
        ctx.h = h.item() if isinstance(h, torch.Tensor) else h
        ctx.grid = grid
        ctx.normalization = ops.volume_normalization(DIM, ctx.h)
        
        packed_SA = torch.zeros((total_points, num_features), dtype=packed_A.dtype, device=packed_A.device)
        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()
        if num_features <= MAX_FEATURES:
            ops.blur_forward_nd(DIM, num_features, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
                packed_x.detach(), packed_v, packed_A.detach(), 
                packed_SA, 0, ctx.h, 
                grid.dims, grid.idx, grid.cell_start, grid.cell_end, num_blocks, num_points, num_cells, ctx.normalization
            )
        else:
            num_iter = math.ceil(num_features / MAX_FEATURES)
            for i in range(num_iter):
                f_start = i * MAX_FEATURES
                f_end = min(num_features, (i+1) * MAX_FEATURES)
                num_features_batch = f_end - f_start
                
                ops.blur_forward_nd(DIM, num_features_batch, FLOAT=ftype)[total_blocks, THREADS_PER_BLOCK](
                    packed_x.detach(), packed_v, packed_A.detach(), 
                    packed_SA, f_start, ctx.h, 
                    grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                    num_blocks, num_points, num_cells, ctx.normalization
                )

        cuda.synchronize()
        ctx.save_for_backward(packed_x, packed_v, packed_A)
        return packed_SA

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.no_grad()
    def backward(ctx, packed_dLdSA):
        packed_dLdx = packed_dLdv = packed_dLdA = dLdh = dLdgrid = None
        packed_x, packed_v, packed_A = ctx.saved_tensors
        
        total_points, DIM = packed_x.shape
        total_points_A, num_features = packed_A.shape
        assert total_points_A == total_points
        ftype = NP_FLOAT_TO_TORCH_FLOAT[packed_x.dtype]

        backprop_dLdx = ctx.backprop_dLdx
        backprop_dLdA = ctx.backprop_dLdA
        grid = ctx.grid

        if backprop_dLdx:
            packed_dLdx = torch.zeros_like(packed_x)
        if backprop_dLdA:
            packed_dLdA = torch.zeros_like(packed_A)

        EMPTY = torch.empty([], device=packed_x.device)
        dLdx_in = EMPTY if packed_dLdx is None else packed_dLdx
        dLdA_in = EMPTY if packed_dLdA is None else packed_dLdA

        num_blocks = grid.get_num_blocks(THREADS_PER_BLOCK)
        num_points = grid.num_points
        num_cells = grid.total_cells
        total_blocks = torch.sum(num_blocks).item()
        if num_features <= MAX_FEATURES:
            ops.blur_backward_nd(
                DIM, num_features, backprop_dLdx, backprop_dLdA, FLOAT=ftype
            )[total_blocks, THREADS_PER_BLOCK](
                packed_x.detach(), packed_v, packed_A.detach(), packed_dLdSA.detach(),
                dLdx_in, dLdA_in, 0, ctx.h,
                grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                num_blocks, num_points, num_cells, ctx.normalization
            )
        else:
            num_iter = math.ceil(num_features / MAX_FEATURES)
            for i in range(num_iter):
                f_start = i * MAX_FEATURES
                f_end = min(num_features, (i+1) * MAX_FEATURES)
                num_features_batch = f_end - f_start
                
                ops.blur_backward_nd(
                    DIM, num_features_batch, backprop_dLdx, backprop_dLdA, FLOAT=ftype
                )[total_blocks, THREADS_PER_BLOCK](
                    packed_x.detach(), packed_v, packed_A.detach(), packed_dLdSA.detach(),
                    dLdx_in, dLdA_in, f_start, ctx.h, 
                    grid.dims, grid.idx, grid.cell_start, grid.cell_end, 
                    num_blocks, num_points, num_cells, ctx.normalization
                )
        
        cuda.synchronize()
        return packed_dLdx, packed_dLdv, packed_dLdA, dLdh, dLdgrid


