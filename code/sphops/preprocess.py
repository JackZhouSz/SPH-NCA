import math
import torch

from . import ops

from .hashgrid import single_impl as impl
from .utils import unpack

Hashgrid = impl.Hashgrid

@torch.no_grad()
def initialize_hashgrid(packed_points, sections, h, dims):
    """
    packed_points: f[N,D] - particle positions (packed)
    sections: i[B] - packed sections
    h: float / float[1] - hashgrid cell size
    dims: i[D] - hashgrid dimensions
    """
    xx = unpack(packed_points.detach(), sections)
    DIM = packed_points.shape[-1]
    
    if isinstance(dims, int):
        dims = torch.full((DIM,), dims, dtype=int, device=packed_points.device)
    if isinstance(dims, tuple):
        dims = torch.tensor(dims, dtype=int, device=packed_points.device)
    
    assert dims.shape[0] == DIM
    total_cells = math.prod(dims).item()

    h = h.item() if isinstance(h, torch.Tensor) else h

    grid = Hashgrid(h, dims, total_cells)

    for p in xx:
        hi = impl.grid_hash(p, h, dims)
        hi, idx = torch.sort(hi, dim=-1)
        cell_start, cell_end = impl.cell_index_init(hi, total_cells)

        occupied_cells, points_per_cells = torch.unique_consecutive(hi, return_counts=True)

        grid.append(idx, cell_start, cell_end, occupied_cells, points_per_cells)
    grid.merge()
    return grid
