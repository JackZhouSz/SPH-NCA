import numpy as np
from tqdm import trange

import torch

import os
import math
from datetime import datetime

import nca as nca
from commons import geometry

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--initial_feature', type=str, choices=['radial', 'random'], default='radial')
parser.add_argument('--initial_feature_radius', type=float, default=-1)
parser.add_argument('--use_alpha', type=str2bool, default=True)
parser.add_argument('--wrap', type=str2bool, default=False)
parser.add_argument('--image_size', type=int, default=-1)
parser.add_argument('--surface', type=str, default='')
parser.add_argument('--surface_scale', type=float, default=1.0)
parser.add_argument('--surface_numpoints', type=int, default=25600)
parser.add_argument('--surface_numseed', type=int, default=10)
parser.add_argument('--steps', type=int, default=128)
parser.add_argument('--nca_update', type=str, choices=['orig', 'gated'], default='gated')
parser.add_argument('--nca_normalize_grad', type=str2bool, default=False)
parser.add_argument('--nca_normalize_perception', type=float, default=-1)
parser.add_argument('--h', type=float, default=0.08)
parser.add_argument('--firerate', type=float, default=0.5)
parser.add_argument('--output_dir', type=str, default='./output/')

args = parser.parse_args()
print(args)

GPU_INDEX = args.gpu_index
SEED = args.seed

USE_ALPHA = args.use_alpha
RANDOM_FEATURE = (args.initial_feature == 'random')

USE_WRAP = args.wrap

H = args.h
DIMS = math.ceil(2 / H)

INITIAL_FEATURE_R = args.initial_feature_radius
if args.initial_feature_radius < 0:
    INITIAL_FEATURE_R = H

SURFACE_TYPE = '3D' if args.surface else 'Image' if args.image_size > 0 else 'Error'
M = args.image_size
SURFACE = args.surface
SURFACE_SCALE = args.surface_scale
SURFACE_NUMPOINTS = args.surface_numpoints
SURFACE_NUMSEED = args.surface_numseed

CHANNEL_N = 16        # Number of CA state channels
CELL_FIRE_RATE = args.firerate

STEPS = args.steps

NCA_UPDATE = args.nca_update
NCA_NORMALIZE_GRAD = args.nca_normalize_grad
NCA_NORMALIZE_PERCEPTION = args.nca_normalize_perception

nca.USE_3D = True

TIMESTAMP = datetime.now()
TIMESTAMP_STR = f'{TIMESTAMP:%m%d%H%M}'
print(TIMESTAMP_STR)

CHECKPOINT = args.checkpoint
OUT_DIR = args.output_dir

#########################
device = torch.device(f'cuda:{GPU_INDEX}')

torch.cuda.set_device(device)

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
#########################
TEST_CONFIGS = {
    'GPU_INDEX': GPU_INDEX,
    'SEED': SEED,
    'INITIAL_FEATURE': args.initial_feature,
    'INITIAL_FEATURE_R': INITIAL_FEATURE_R,
    'USE_ALPHA': USE_ALPHA,
    'WRAP': USE_WRAP,
    'SURFACE_TYPE': SURFACE_TYPE,
    'IMAGE_SIZE': M,
    'SURFACE': SURFACE,
    'SURFACE_SCALE': SURFACE_SCALE,
    'SURFACE_NUMPOINTS': SURFACE_NUMPOINTS,
    'SURFACE_NUMSEED': SURFACE_NUMSEED,
    'STEPS': STEPS,
    'H': H,
    'NCA_UPDATE': NCA_UPDATE,
    'NCA_NORMALIZE_GRAD': NCA_NORMALIZE_GRAD,
    'NCA_NORMALIZE_PERCEPTION': NCA_NORMALIZE_PERCEPTION,
    'CHANNEL_N': CHANNEL_N,
    'CELL_FIRE_RATE': CELL_FIRE_RATE,
    'TIMESTAMP_STR': TIMESTAMP_STR,
    'CHECKPOINT': CHECKPOINT,
    'OUT_DIR': OUT_DIR,
    'DEVICE': device,
}

model = nca.SPHNCA(CHANNEL_N, CELL_FIRE_RATE, update_rule=NCA_UPDATE, normalize_grad=NCA_NORMALIZE_GRAD, use_alpha=USE_ALPHA, normalize_perception=NCA_NORMALIZE_PERCEPTION).to(device)
checkpoint = torch.load(CHECKPOINT, weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

if SURFACE_TYPE == 'Image':
    GSHAPE = torch.tensor([M, M], device=device)
    GMIN = torch.tensor([-1, -1], device=device)
    GSIZE = torch.tensor([2, 2], device=device)

    def get_seed():
        x = geometry.grange(GSHAPE, GMIN, GSIZE).to(device)
        A = torch.zeros((*x.shape[:-1], CHANNEL_N), device=device)

        c = GMIN + GSIZE / 2
        x = x.view(-1, 2)
        A = A.view(-1, CHANNEL_N)

        if RANDOM_FEATURE:
            A = torch.rand_like(A)
        else:
            nca.add_radial_seed(x, A, center=c, R=INITIAL_FEATURE_R)
        return x, A
elif SURFACE_TYPE == '3D':
    import trimesh
    import fpsample
    from commons.sampling import UniformSurfaceSampler

    def get_seed():
        mesh = trimesh.load_mesh(SURFACE)
        mesh.fix_normals()

        mesh.vertices = mesh.vertices[...,[2,0,1]]
        mesh.vertices -= mesh.vertices.mean(axis=-2)
        mesh.vertices /= np.abs(mesh.vertices).max()
        mesh.vertices *= SURFACE_SCALE

        v, f, n = mesh.vertices, mesh.faces, mesh.vertex_normals
        v, f, n = [torch.from_numpy(x).to(device) for x in [v, f, n]]
        v, n = v.float(), n.float()

        vf = v[f,:]
        sampler = UniformSurfaceSampler(vf, return_barycentric=True)
        x3, fi, w = sampler.get(SURFACE_NUMPOINTS * 8)
        N = (n[f,:][fi,:] * w[...,None]).sum(-2)

        idx = fpsample.bucket_fps_kdline_sampling(x3.cpu().numpy(), SURFACE_NUMPOINTS, h=7).astype(np.int64)
        idx = torch.from_numpy(idx).long().to(device)
        x = x3[idx,:]
        sections = (x.shape[0],)
        A = torch.zeros((*x.shape[:-1], CHANNEL_N), device=device)

        N = N[idx,:]
        T0 = torch.zeros_like(N)

        if RANDOM_FEATURE:
            # CREATE CONSISTENT TANGENT VECTORS
            TC = torch.zeros_like(T0)
            for _ in range(10):
                nca.add_radial_seed(x, A, N, TC, R=0.2)
            A = torch.ones_like(A)
            for _ in range(50):
                TC = nca.diffuse(N, TC, x, A, sections, 0.2, 10, lerp_multiplier=0.0)
            A = torch.rand_like(A)
            T0[:,:] = TC
        else:
            SEED_IDX = fpsample.bucket_fps_kdline_sampling(x.cpu().numpy(), SURFACE_NUMSEED, h=5).astype(np.int64)
            for i in range(SURFACE_NUMSEED):
                nca.add_radial_seed(x, A, N, T0, R=INITIAL_FEATURE_R, idx=SEED_IDX[i])
        return x, A, N, T0


seed = get_seed()

with torch.no_grad():
    if SURFACE_TYPE == '3D':
        x, A, N, T0 = seed
        sections = (x.shape[0],)
        _, _, out = nca.sample_mesh(
            model, x, A, N, T0, sections, H, DIMS, out_steps=True, iter_n=STEPS, fire_rate=CELL_FIRE_RATE,
            lerp_multiplier=1.0, w_multiplier=1.0
        )
    elif SURFACE_TYPE == 'Image':
        x, A = seed
        sections = (x.shape[0],)
        if USE_WRAP:
            _, _, out, _, _ = nca.sample_plane_wrapped(
                model, x, A, sections, H, DIMS, GMIN, GSIZE, out_steps=True, iter_n=STEPS, fire_rate=CELL_FIRE_RATE
            )
        else:
            _, _, out = nca.sample_plane(
                model, x, A, sections, H, DIMS, out_steps=True, iter_n=STEPS, fire_rate=CELL_FIRE_RATE
            )
    
out_raw = [A.cpu().numpy() for A in out]
out_col = [nca.get_rgba(A, margin=0).cpu().numpy() for A in out]
if not model.use_alpha:
    out_col = [A[...,:3] for A in out_col]

output_path = f'{OUT_DIR}/sphnca-test-{TIMESTAMP_STR}.pt'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save({
    'configs': TEST_CONFIGS,
    'seed': seed,
    'out_raw': out_raw,
    'out_col': out_col,
}, output_path)

if SURFACE_TYPE == 'Image':
    from PIL import Image
    CHANNELS = 4 if model.use_alpha else 3
    img_dir = f'{OUT_DIR}/sphnca-test-{TIMESTAMP_STR}/'
    os.makedirs(img_dir, exist_ok=True)
    for i in trange(len(out_col), desc='Saving images'):
        img = out_col[i].reshape(M, M, CHANNELS)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        filename = f'{img_dir}/{i:04d}.png'
        try:
            img_pil.save(filename)
        except Exception as e:
            print(f'Error saving image {filename}: {e}')
elif SURFACE_TYPE == '3D':
    import trimesh
    mesh_dir = f'{OUT_DIR}/sphnca-test-{TIMESTAMP_STR}/'
    os.makedirs(mesh_dir, exist_ok=True)
    for i in trange(len(out_col), desc='Saving point clouds'):
        x = seed[0].cpu().numpy().astype(np.float32)
        rgba = out_col[i]
        if rgba.shape[-1] == 3:
            rgba = np.concatenate([rgba, np.ones_like(rgba[..., :1])], axis=-1)
        rgba = (rgba * 255).astype(np.uint8)
        pc = trimesh.PointCloud(x, rgba)
        filename = f'{mesh_dir}/{i:04d}.ply'
        try:
            pc.export(filename)
        except Exception as e:
            print(f'Error saving point cloud {filename}: {e}')
