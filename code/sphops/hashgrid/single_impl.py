import math
import functools
import torch
import numpy as np

from dataclasses import dataclass
from numba import cuda

from .. import utils

INTTYPE = torch.int32
INTTYPE_NP = np.int32

@functools.cache
def grid_index_nd(DIM):
    @cuda.jit(device=True)
    def grid_index(x, h, gi):
        for i in range(DIM):
            gi[i] = int(math.floor(x[i] / h))
    return grid_index

@functools.cache
def grid_hash_from_index_nd(DIM):
    @cuda.jit(device=True)
    def grid_hash_from_index(gi, dims):
        out = int(0)
        c = int(1)
        for i in range(DIM):
            out += c * (gi[i] % dims[i])
            c *= dims[i]
        return out
    return grid_hash_from_index

@functools.cache
def grid_hash_from_pos_nd(DIM):
    @cuda.jit(device=True)
    def grid_hash_from_pos(x, h, dims):
        out = int(0)
        c = int(1)
        for i in range(DIM):
            out += c * (int(math.floor(x[i] / h)) % dims[i])
            c *= dims[i]
        return out
    return grid_hash_from_pos

@functools.cache
def grid_hash_neighbors_nd(DIM):
    @cuda.jit(device=True)
    def grid_hash_neighbors(h, dgi, dims):
        out = int(0)
        h_gi = h
        c = int(1)
        for i in range(DIM):
            gi = h_gi % dims[i]
            gi = (gi + dims[i] + dgi[i]) % dims[i]
            out += c * gi
            h_gi //= dims[i]
            c *= dims[i]
        return out
    return grid_hash_neighbors

@functools.cache
def grid_hash_kernel_nd(DIM):
    grid_hash_from_pos = grid_hash_from_pos_nd(DIM)
    @cuda.jit
    def grid_hash_kernel(x, hi, h, dims):
        num_points = x.shape[0]
        N = cuda.grid(1)
        if N >= num_points:
            return

        hi[N] = grid_hash_from_pos(x[N,:], h, dims)
    return grid_hash_kernel

def grid_hash(x, h, dims):
    BLOCK = 256

    num_points, DIM = x.shape[-2], x.shape[-1]
    if isinstance(DIM, torch.Tensor):
        DIM = DIM.item()
    if isinstance(num_points, torch.Tensor):
        num_points = num_points.item()

    assert dims.shape[0] == DIM

    hi = torch.zeros((num_points, ), dtype=INTTYPE, device=x.device)

    grid_hash_kernel_nd(DIM)[((num_points-1)//BLOCK+1), (BLOCK)](x, hi, h, dims)
    cuda.synchronize()
    
    return hi

@cuda.jit
def cell_index_init_kernel(hi, cell_start, cell_end):
    num_points = hi.shape[0]
    N = cuda.grid(1)
    if N >= num_points:
        return

    c = hi[N] # current cell_id
    if N == 0:
        cell_start[c] = 0
    else:
        prev_c = hi[N-1]
        if c != prev_c:
            cell_start[c] = N
            cell_end[prev_c] = N
    if N == num_points - 1:
        cell_end[c] = num_points

def cell_index_init(hi, num_cells):
    BLOCK = 256

    num_points = hi.shape[0]
    if isinstance(num_points, torch.Tensor):
        num_points = num_points.item()

    cell_start = torch.full((num_cells, ), -1, dtype=INTTYPE, device=hi.device)
    cell_end = torch.full((num_cells, ), -1, dtype=INTTYPE, device=hi.device)

    cell_index_init_kernel[((num_points-1)//BLOCK+1), (BLOCK)](hi, cell_start, cell_end)
    cuda.synchronize()

    return cell_start, cell_end


@dataclass
class Hashgrid:
    """Class for maintaining Hashgrid"""
    h: float
    dims: torch.Tensor # f[DIM]
    total_cells: int
    
    idx: torch.Tensor | list[torch.Tensor] # i[#Points]
    cell_start: torch.Tensor | list[torch.Tensor] # i[#CELLS]
    cell_end: torch.Tensor | list[torch.Tensor] # i[#CELLS]
    occupied_cells: list[torch.Tensor]
    points_per_cells: list[torch.Tensor]

    num_points: torch.Tensor #i[B]
    
    def __init__(self, h, dims, total_cells):
        self.h = h
        self.dims = dims
        self.total_cells = total_cells

        self.idx = []
        self.cell_start = []
        self.cell_end = []
        self.occupied_cells = []
        self.points_per_cells = []

    def append(self, idx, cell_start, cell_end, occupied_cells, points_per_cells):
        self.idx.append(idx)
        self.cell_start.append(cell_start)
        self.cell_end.append(cell_end)
        self.occupied_cells.append(occupied_cells)
        self.points_per_cells.append(points_per_cells)
    
    def merge(self):
        DEVICE = self.dims.device
        batch_total_points = [idx.shape[0] for idx in self.idx]

        self.num_points = torch.tensor(batch_total_points, dtype=INTTYPE, device=DEVICE)

        self.idx, _ = utils.pack(*self.idx)
        self.cell_start, _ = utils.pack(*self.cell_start)
        self.cell_end, _ = utils.pack(*self.cell_end)

    def get_num_blocks(self, block_size):
        return torch.ceil(self.num_points / block_size).to(INTTYPE)
