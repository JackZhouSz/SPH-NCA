import functools
import numpy as np
from numba import cuda

from . import kernels
from ..hashgrid import single_impl as hashgrid
from .. import utils

DEBUG_MODE = False
FASTMATH_MODE = True

@functools.cache
def volume_normalization(DIM, h, *, FLOAT=np.float32):
    return kernels.smoothing_kernel_normalization_nd(DIM, FLOAT)(h)

@functools.cache
def volume_forward_nd(DIM, *, FLOAT=np.float32):
    print(f"no cache found for `volume_forward_nd`! recompiling for {DIM = }, {FLOAT = }")
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    smoothing_kernel = kernels.smoothing_kernel_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, # in
        v, # out
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)
        
        inv_v = 0.0
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        r = cuda.local.array(DIM, FLOAT)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]
                inv_v += smoothing_kernel(r, h)
                ### neighborhood logic ends
                idx_current += 1
        v[i] = cuda.libdevice.frcp_rn(normalization * inv_v)
    return kernel

@functools.cache
def volume_backward_nd(DIM, BACKPROP_dLdx, *, FLOAT=np.float32):
    print(f"no cache found for `volume_backward_nd`! recompiling for {DIM = }, {BACKPROP_dLdx = }, {FLOAT = }")
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    smoothing_kernel_dr = kernels.smoothing_kernel_dr_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, v, dLdv, # in
        dLdx, # out
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)
        
        vi2 = v[i] ** 2
        dLdvi = dLdv[i]

        dLdxi = cuda.local.array(DIM, FLOAT)
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        r = cuda.local.array(DIM, FLOAT)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]
                vj2 = v[j] ** 2
                dLdvj = dLdv[j]

                if BACKPROP_dLdx:
                    smoothing_kernel_dr(r, h, dLdxi, vi2*dLdvi + vj2*dLdvj)
                ### neighborhood logic ends
                idx_current += 1
        
        if BACKPROP_dLdx:
            for d in range(DIM):
                dLdx[i,d] = normalization * dLdxi[d]
    return kernel


@functools.cache
def gradient_normalization(DIM, h, *, FLOAT=np.float32):
    return kernels.gradient_kernel_normalization_nd(DIM, FLOAT)(h)

@functools.cache
def gradient_forward_nd(DIM, NUM_FEATURES, *, FLOAT=np.float32):
    print(f"no cache found for `gradient_forward_nd`! recompiling for {DIM = }, {NUM_FEATURES = }, {FLOAT = }")
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    gradient_kernel = kernels.gradient_kernel_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, v, A, # in
        GA, # out
        f_start, 
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)

        Ai = cuda.local.array(NUM_FEATURES, FLOAT)
        GAi = cuda.local.array((NUM_FEATURES, DIM), FLOAT)
        for f in range(NUM_FEATURES):
            Ai[f] = A[i,f_start+f]
            for d in range(DIM):
                GAi[f,d] = 0.0

        grad_w = cuda.local.array(DIM, FLOAT)
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        r = cuda.local.array(DIM, FLOAT)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                vj = v[j]
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]
                gradient_kernel(r, h, grad_w, vj)
                for f in range(NUM_FEATURES):
                    dA = A[j,f_start+f] - Ai[f]
                    for d in range(DIM):
                        GAi[f,d] += dA * grad_w[d]
                ### neighborhood logic ends
                idx_current += 1
        for f in range(NUM_FEATURES):
            for d in range(DIM):
                GA[i,f_start+f,d] = normalization * GAi[f,d]
    return kernel

@functools.cache
def gradient_backward_nd(DIM, NUM_FEATURES, BACKPROP_dLdx, BACKPROP_dLdA, *, FLOAT=np.float32):
    print(f"no cache found for `gradient_backward_nd`! recompiling for {DIM = }, {BACKPROP_dLdx = }, {BACKPROP_dLdA = }, {FLOAT = }")
    dot = utils.dot_nd(DIM)
    matvecmul = utils.matvecmul_nd(DIM)
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    gradient_kernel = kernels.gradient_kernel_nd(DIM, FLOAT)
    gradient_kernel_dr = kernels.gradient_kernel_dr_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, v, A, dLdGA, # in
        dLdx, dLdA, # out
        f_start, 
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)
        # assert 0 <= cell < cell_end.shape[-1]

        vi = v[i]
        Ai = cuda.local.array(NUM_FEATURES, FLOAT)
        dLdGAi = cuda.local.array((NUM_FEATURES, DIM), FLOAT)
        for f in range(NUM_FEATURES):
            Ai[f] = A[i,f_start+f]
            for d in range(DIM):
                dLdGAi[f,d] = dLdGA[i,f_start+f,d]

        dLdxi = cuda.local.array(DIM, FLOAT)
        dLdAi = cuda.local.array(NUM_FEATURES, FLOAT)
        for d in range(DIM):
            dLdxi[d] = 0.0
        for f in range(NUM_FEATURES):
            dLdAi[f] = 0.0

        r = cuda.local.array(DIM, FLOAT)
        vec = cuda.local.array(DIM, FLOAT)
        gk = cuda.local.array(DIM, FLOAT)
        gkdr = cuda.local.array((DIM,DIM), FLOAT)
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                vj = v[j]
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]

                gradient_kernel(r, h, gk, 1.0)
                gradient_kernel_dr(r, h, gkdr)
                for f in range(NUM_FEATURES):
                    dA = A[j,f_start+f] - Ai[f]
                    
                    ## dLdxi
                    if BACKPROP_dLdx:
                        for d in range(DIM):
                            vec[d] = dA * (vi*dLdGA[j,f_start+f,d] - vj*dLdGAi[f,d])
                        matvecmul(gkdr, vec, dLdxi)

                    ## dLdAi
                    if BACKPROP_dLdA:
                        for d in range(DIM):
                            vec[d] = -(vi*dLdGA[j,f_start+f,d] + vj*dLdGAi[f,d])
                        dLdAi[f] += dot(gk, vec)
                ### neighborhood logic ends
                idx_current += 1
        if BACKPROP_dLdx:
            for d in range(DIM):
                dLdx[i,d] = normalization * dLdxi[d]
        for f in range(NUM_FEATURES):
            if BACKPROP_dLdA:
                dLdA[i,f_start+f] = normalization * dLdAi[f]
    return kernel


@functools.cache
def divergence_forward_nd(DIM, NUM_FEATURES, *, FLOAT=np.float32):
    print(f"no cache found for `divergence_forward_nd`! recompiling for {DIM = }, {NUM_FEATURES = }, {FLOAT = }")
    dot = utils.dot_nd(DIM)
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    gradient_kernel = kernels.gradient_kernel_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, v, A, # in
        DA, # out
        f_start, 
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)

        Ai = cuda.local.array((NUM_FEATURES, DIM), FLOAT)
        DAi = cuda.local.array(NUM_FEATURES, FLOAT)
        for f in range(NUM_FEATURES):
            DAi[f] = 0.0
            for d in range(DIM):
                Ai[f,d] = A[i,f_start+f,d]

        r = cuda.local.array(DIM, FLOAT)
        gk = cuda.local.array(DIM, FLOAT)
        dA = cuda.local.array(DIM, FLOAT)
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                vj = v[j]
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]
                gradient_kernel(r, h, gk, 1.0)
                for f in range(NUM_FEATURES):
                    for d in range(DIM):
                        dA[d] = A[j,f_start+f,d] - Ai[f,d]
                    DAi[f] += vj * dot(dA, gk)
                ### neighborhood logic ends
                idx_current += 1
        for f in range(NUM_FEATURES):
            DA[i,f_start+f] = normalization * DAi[f]
    return kernel

@functools.cache
def divergence_backward_nd(DIM, NUM_FEATURES, BACKPROP_dLdx, BACKPROP_dLdA, *, FLOAT=np.float32):
    print(f"no cache found for `divergence_backward_nd`! recompiling for {DIM = }, {NUM_FEATURES = }, {BACKPROP_dLdx = }, {BACKPROP_dLdA = }, {FLOAT = }")
    matvecmul = utils.matvecmul_nd(DIM)
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    gradient_kernel = kernels.gradient_kernel_nd(DIM, FLOAT)
    gradient_kernel_dr = kernels.gradient_kernel_dr_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def divergence_backward_kernel(
        x, v, A, dLdDA, # in
        dLdx, dLdA, # out
        f_start, 
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)

        vi = v[i]
        Ai = cuda.local.array((NUM_FEATURES, DIM), FLOAT)
        dLdDAi = cuda.local.array(NUM_FEATURES, FLOAT)
        for f in range(NUM_FEATURES):
            dLdDAi[f] = dLdDA[i,f_start+f]
            for d in range(DIM):
                Ai[f,d] = A[i,f_start+f,d]
        
        dLdxi = cuda.local.array(DIM, FLOAT)
        dLdAi = cuda.local.array((NUM_FEATURES, DIM), FLOAT)
        for d in range(DIM):
            dLdxi[d] = 0.0
        for f in range(NUM_FEATURES):
            for d in range(DIM):
                dLdAi[f,d] = 0.0

        r = cuda.local.array(DIM, FLOAT)
        dA = cuda.local.array(DIM, FLOAT)
        vec = cuda.local.array(DIM, FLOAT)
        gk = cuda.local.array(DIM, FLOAT)
        gkdr = cuda.local.array((DIM,DIM), FLOAT)
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                vj = v[j]
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]

                gradient_kernel(r, h, gk, 1.0)
                gradient_kernel_dr(r, h, gkdr)
                for f in range(NUM_FEATURES):
                    for d in range(DIM):
                        dA[d] = A[j,f_start+f,d] - Ai[f,d]
                    
                    ## dLdxi
                    if BACKPROP_dLdx:
                        for d in range(DIM):
                            vec[d] = -dA[d] * (vi*dLdDA[j,f_start+f] + vj*dLdDAi[d])
                        matvecmul(gkdr, vec, dLdxi)

                    ## dLdAi
                    if BACKPROP_dLdA:
                        for d in range(DIM):
                            dLdAi[f,d] += gk[d] * (vi*dLdDA[j,f_start+f] - vj*dLdDAi[d])
                ### neighborhood logic ends
                idx_current += 1
        if BACKPROP_dLdx:
            for d in range(DIM):
                dLdx[i,d] = normalization * dLdxi[d]
        if BACKPROP_dLdA:
            for f in range(NUM_FEATURES):
                for d in range(DIM):
                    dLdA[i,f_start+f,d] = normalization * dLdAi[f,d]
    return divergence_backward_kernel


@functools.cache
def count_forward_nd(DIM, *, FLOAT=np.float32):
    print(f"no cache found for `count_forward_nd`! recompiling for {DIM = }, {FLOAT = }")
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    count_kernel = kernels.count_kernel_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, # in
        c, # out
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)
        
        count = 0
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        r = cuda.local.array(DIM, FLOAT)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]
                count += count_kernel(r, h)
                ### neighborhood logic ends
                idx_current += 1
        c[i] = count
    return kernel


@functools.cache
def blur_forward_nd(DIM, NUM_FEATURES, *, FLOAT=np.float32):
    print(f"no cache found for `blur_forward_nd`! recompiling for {DIM = }, {NUM_FEATURES = }, {FLOAT = }")
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    smoothing_kernel = kernels.smoothing_kernel_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, v, A, # in
        SA, # out
        f_start, 
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)
        assert 0 <= cell < cell_end.shape[-1]
        
        SAi = cuda.local.array(NUM_FEATURES, FLOAT)
        for f in range(NUM_FEATURES):
            SAi[f] = 0.0
        
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        r = cuda.local.array(DIM, FLOAT)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                vj = v[j]
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]
                wvj = smoothing_kernel(r, h) * vj
                for f in range(NUM_FEATURES):
                    SAi[f] += A[j,f_start+f] * wvj
                ### neighborhood logic ends
                idx_current += 1
        for f in range(NUM_FEATURES):
            SA[i,f_start+f] = SAi[f] * normalization
    return kernel

@functools.cache
def blur_backward_nd(DIM, NUM_FEATURES, BACKPROP_dLdx, BACKPROP_dLdA, *, FLOAT=np.float32):
    print(f"no cache found for `blur_backward_nd`! recompiling for {DIM = }, {NUM_FEATURES = }, {BACKPROP_dLdx = }, {BACKPROP_dLdA = }, {FLOAT = }")
    grid_hash_from_pos = hashgrid.grid_hash_from_pos_nd(DIM)
    grid_hash_neighbors = hashgrid.grid_hash_neighbors_nd(DIM)
    smoothing_kernel = kernels.smoothing_kernel_nd(DIM, FLOAT)
    smoothing_kernel_dr = kernels.smoothing_kernel_dr_nd(DIM, FLOAT)
    
    @cuda.jit(debug=DEBUG_MODE, opt=not DEBUG_MODE, fastmath=FASTMATH_MODE)
    def kernel(
        x, v, A, dLdSA, # in
        dLdx, dLdA, # out
        f_start, 
        h, dims, idx, cell_start, cell_end, 
        num_blocks, num_points, num_cells, normalization
    ):
        TID = cuda.threadIdx.x
        BID = cuda.blockIdx.x
        BW = cuda.blockDim.x

        BATCH = 0
        OFFSET = 0
        for b in range(num_blocks.shape[0]):
            if BID >= num_blocks[b]:
                BID -= num_blocks[b]
                BATCH += 1
                OFFSET += num_points[b]
            else: break
        CELL_OFFSET = num_cells * BATCH
        
        P = TID + BID*BW
        batch_num_points = num_points[BATCH]
        if P >= batch_num_points:
            return
        
        i = OFFSET + idx[OFFSET+P]

        xi = cuda.local.array(DIM, FLOAT)
        for d in range(DIM):
            xi[d] = x[i,d]
        cell = grid_hash_from_pos(xi, h, dims)
        
        Ai = cuda.local.array(NUM_FEATURES, FLOAT)
        dLdSAi = cuda.local.array(NUM_FEATURES, FLOAT)
        for f in range(NUM_FEATURES):
            Ai[f] = A[i,f_start+f]
            for d in range(DIM):
                dLdSAi[f] = dLdSA[i,f_start+f]

        dLdxi = cuda.local.array(DIM, FLOAT)
        dLdAi = cuda.local.array(NUM_FEATURES, FLOAT)
        dSAidh = cuda.local.array(NUM_FEATURES, FLOAT)
        for d in range(DIM):
            dLdxi[d] = 0.0
        for f in range(NUM_FEATURES):
            dLdAi[f] = 0.0
            dSAidh[f] = 0.0

        r = cuda.local.array(DIM, FLOAT)
        grad_w = cuda.local.array(DIM, FLOAT)
        neighbor = cuda.local.array(DIM, hashgrid.INTTYPE_NP)
        for k in range(3 ** DIM):
            for d in range(DIM):
                neighbor[d] = (k % 3) - 1
                k //= 3
            cell_current = CELL_OFFSET + grid_hash_neighbors(cell, neighbor, dims)
            assert CELL_OFFSET <= cell_current < CELL_OFFSET + num_cells
            idx_end = cell_end[cell_current]
            idx_current = cell_start[cell_current]
            assert (idx_end == idx_current == -1) or (0 <= idx_current < idx_end <= batch_num_points)
            while idx_current != idx_end:
                j = OFFSET + idx[OFFSET+idx_current]
                assert OFFSET <= j < OFFSET + batch_num_points
                ### neighborhood logic starts
                vj = v[j]
                for d in range(DIM):
                    r[d] = x[j,d] - xi[d]

                wvj = smoothing_kernel(r, h) * vj
                smoothing_kernel_dr(r, h, grad_w, vj)
                for f in range(NUM_FEATURES):
                    # ## dLdxi
                    dLdSAjf = dLdSA[j, f_start+f]
                    Ajf = A[j,f_start+f]
                    if BACKPROP_dLdx:
                        for d in range(DIM):
                            dLdxi[d] += grad_w[d] * Ajf * (dLdSAjf - dLdSAi[f])

                    ## dLdAi
                    if BACKPROP_dLdA:
                        dLdAi[f] += dLdSAjf * wvj
                ### neighborhood logic ends
                idx_current += 1
        if BACKPROP_dLdx:
            for d in range(DIM):
                dLdx[i,d] = normalization * dLdxi[d]
        for f in range(NUM_FEATURES):
            if BACKPROP_dLdA:
                dLdA[i,f_start+f] = normalization * dLdAi[f]
    return kernel

