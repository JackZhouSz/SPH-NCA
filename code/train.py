import numpy as np
from tqdm import trange

import torch

import os
import math
from datetime import datetime

import nca as nca
import losses
from commons import geometry

##########################
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
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--target', type=str, default='')
parser.add_argument('--img', type=str, default='')
parser.add_argument('--initial_feature', type=str, choices=['radial', 'random'], default='radial')
parser.add_argument('--initial_feature_radius', type=float, default=-1)
parser.add_argument('--loss', type=str, choices=['mse_simple', 'ot', 'clip_multiscale'], default='mse_simple')
parser.add_argument('--use_alpha', type=str2bool, default=True)
parser.add_argument('--wrap', type=str2bool, default=False)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--target_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--training_iter', type=int, default=8000)
parser.add_argument('--steps_range', type=str, default='32,48')
parser.add_argument('--steps_increment', type=int, default=5)
parser.add_argument('--loss_weight_color', type=float, default=0.05)
parser.add_argument('--loss_weight_clip', type=float, default=1)
parser.add_argument('--loss_weight_overflow', type=float, default=0.05)
parser.add_argument('--loss_weight_style', type=float, default=1)
parser.add_argument('--clip_guide', type=str, default='')
parser.add_argument('--clip_multiscale_scales', type=str, default='1')
parser.add_argument('--nca_update', type=str, choices=['orig', 'gated'], default='gated')
parser.add_argument('--nca_normalize_grad', type=str2bool, default=True)
parser.add_argument('--nca_normalize_perception', type=float, default=-1)
parser.add_argument('--alpha_premultiply', type=str2bool, default=True)
parser.add_argument('--pretrained_checkpoint', type=str, default='')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--pool_size', type=int, default=1024)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--h', type=float, default=0.08)
parser.add_argument('--output_dir', type=str, default='./checkpoints/')


args = parser.parse_args()
print(args)

GPU_INDEX = args.gpu_index
SEED = args.seed
TARGET_EMOJI = args.target

USE_ALPHA = args.use_alpha
RANDOM_FEATURE = (args.initial_feature == 'random')

USE_WRAP = args.wrap

H = args.h
DIMS = math.ceil(2 / H)

INITIAL_FEATURE_R = args.initial_feature_radius
if args.initial_feature_radius < 0:
    INITIAL_FEATURE_R = H

M = args.image_size
TARGET_SIZE = args.target_size
IMAGE_SCALE = TARGET_SIZE / M

CHANNEL_N = 16        # Number of CA state channels
BATCH_SIZE = args.batch_size
POOL_SIZE = args.pool_size
CELL_FIRE_RATE = 0.5

OPTIMIZER = args.optimizer
LEARNING_RATE = args.lr
TRAINING_ITER = args.training_iter
STEPS_RANGE = [int(s) for s in args.steps_range.split(',')]
STEPS_MEAN = (STEPS_RANGE[0] + STEPS_RANGE[1]) // 2
STEPS_INCREMENT = args.steps_increment

LOSS_WEIGHT_COLOR = args.loss_weight_color
LOSS_WEIGHT_CLIP = args.loss_weight_clip
LOSS_WEIGHT_OVERFLOW = args.loss_weight_overflow
LOSS_WEIGHT_STYLE = args.loss_weight_style

CLIP_GUIDE = args.clip_guide
CLIP_SCALES = [float(s) for s in args.clip_multiscale_scales.split(',')]

NCA_UPDATE = args.nca_update
NCA_NORMALIZE_GRAD = args.nca_normalize_grad
NCA_NORMALIZE_PERCEPTION = args.nca_normalize_perception
if NCA_NORMALIZE_PERCEPTION < 0:
    NCA_NORMALIZE_PERCEPTION = 1.0 / H

ALPHA_PREMULTIPLY = args.alpha_premultiply

PRETRAINED_CHECKPOINT = args.pretrained_checkpoint

nca.USE_3D = True

TIMESTAMP = datetime.now()
TIMESTAMP_STR = f'{TIMESTAMP:%m%d%H%M}'
print(TIMESTAMP_STR)

OUT_DIR = args.output_dir

TITLE = f'NCA-{TARGET_EMOJI + args.img + CLIP_GUIDE}'
RUN_ID = f'{TIMESTAMP_STR}-{TITLE}'


#########################
device = torch.device(f'cuda:{GPU_INDEX}')

torch.cuda.set_device(device)

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
#########################



GSHAPE = torch.tensor([M, M], device=device)
GMIN = torch.tensor([-1, -1], device=device)
GSIZE = torch.tensor([2, 2], device=device)

TRAINING_CONFIGS = {
    'GPU_INDEX': GPU_INDEX,
    'SEED': SEED,
    'TARGET_EMOJI': TARGET_EMOJI,
    'IMG': args.img,
    'INITIAL_FEATURE': args.initial_feature,
    'INITIAL_FEATURE_R': INITIAL_FEATURE_R,
    'H': H,
    'LOSS': args.loss,
    'USE_ALPHA': USE_ALPHA,
    'WRAP': USE_WRAP,
    'IMAGE_SIZE': M,
    'TARGET_SIZE': TARGET_SIZE,
    'LEARNING_RATE': LEARNING_RATE,
    'TRAINING_ITER': TRAINING_ITER,
    'STEPS_RANGE': STEPS_RANGE,
    'STEPS_MEAN': STEPS_MEAN,
    'STEPS_INCREMENT': STEPS_INCREMENT,
    'LOSS_WEIGHT_COLOR': LOSS_WEIGHT_COLOR,
    'LOSS_WEIGHT_CLIP': LOSS_WEIGHT_CLIP,
    'LOSS_WEIGHT_OVERFLOW': LOSS_WEIGHT_OVERFLOW,
    'LOSS_WEIGHT_STYLE': LOSS_WEIGHT_STYLE,
    'CLIP_GUIDE': CLIP_GUIDE,
    'CLIP_SCALES': CLIP_SCALES,
    'NCA_UPDATE': NCA_UPDATE,
    'NCA_NORMALIZE_GRAD': NCA_NORMALIZE_GRAD,
    'NCA_NORMALIZE_PERCEPTION': NCA_NORMALIZE_PERCEPTION,
    'ALPHA_PREMULTIPLY': ALPHA_PREMULTIPLY,
    'CHANNEL_N': CHANNEL_N,
    'BATCH_SIZE': BATCH_SIZE,
    'POOL_SIZE': POOL_SIZE,
    'CELL_FIRE_RATE': CELL_FIRE_RATE,
    'IMAGE_SCALE': IMAGE_SCALE,
    'TIMESTAMP_STR': TIMESTAMP_STR,
    'PRETRAINED_CHECKPOINT': PRETRAINED_CHECKPOINT,
    'TITLE': TITLE,
    'RUN_ID': RUN_ID,
    'OUT_DIR': OUT_DIR,
    'GSHAPE': GSHAPE,
    'GMIN': GMIN,
    'GSIZE': GSIZE,
    'DEVICE': device,
}


###########################
import io
import requests
import PIL.Image

def load_image(url, max_size=TARGET_SIZE, is_path=False):
    if is_path:
        img = PIL.Image.open(url)
    else:
        r = requests.get(url)
        img = PIL.Image.open(io.BytesIO(r.content))
    if img.mode == 'L':
        img = img.convert('RGB')
    img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
    img = np.float32(img)/255.0
    # premultiply RGB by Alpha
    if img.shape[-1] == 4:
        if ALPHA_PREMULTIPLY:
            img[..., :3] *= img[..., 3:]
    elif img.shape[-1] == 3:
        img = np.pad(img, [(0,0),] * (len(img.shape)-1) + [(0,1)], constant_values=1)
    return img

def load_emoji(emoji):
    code_points = [f"{ord(c):04x}" for c in emoji]
    code = '_'.join(code_points)
    url = f'https://github.com/googlefonts/noto-emoji/blob/main/png/512/emoji_u{code}.png?raw=true'
    print(f'{url = }')
    return load_image(url)

if args.target:
    import regex
    clusters = regex.findall(r'\X', args.target)
    if len(clusters) == 1:
        img = load_emoji(clusters[0])
    else:
        img = [load_emoji(c) for c in clusters]
        img = np.stack(img, axis=0)
elif args.img:
    img = load_image(args.img, is_path=True)
else:
    img = np.zeros([TARGET_SIZE, TARGET_SIZE, 3], dtype=np.float32)
    img[...,0] = 1
    img[...,1] = 0.5
    img[...,2] = 0
    # img = np.random.rand(TARGET_SIZE, TARGET_SIZE, 3).astype(np.float32)
img = torch.from_numpy(img).to(device)

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

seed = get_seed()

###########################
torch.manual_seed(SEED)
np.random.seed(SEED)

import random
random.seed(SEED)


###########################
model = nca.SPHNCA(CHANNEL_N, CELL_FIRE_RATE, update_rule=NCA_UPDATE, normalize_grad=NCA_NORMALIZE_GRAD, use_alpha=USE_ALPHA, normalize_perception=NCA_NORMALIZE_PERCEPTION).to(device)
NUM_PARAMS = sum(p.numel() for p in model.parameters())

print(f'Model initialized: #params = {NUM_PARAMS}')

if PRETRAINED_CHECKPOINT:
    print(f'Loading pretrained checkpoint: {PRETRAINED_CHECKPOINT}')
    checkpoint = torch.load(PRETRAINED_CHECKPOINT, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model'])

loss_fn = None
match args.loss:
    case 'mse_simple':
        loss_fn = losses.get_mse_loss(model, img, TRAINING_CONFIGS)
    case 'ot':
        loss_fn = losses.get_ot_loss(model, img, TRAINING_CONFIGS)
    case 'clip_multiscale':
        loss_fn = losses.get_clip_loss(model, CLIP_GUIDE, TRAINING_CONFIGS)


lr = LEARNING_RATE

OptimizerClass = getattr(torch.optim, OPTIMIZER, torch.optim.Adam)
optimizer = OptimizerClass(model.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=2000)

###########################
TRAIN_START_TIME = datetime.now()

pool = nca.Pool(seed, POOL_SIZE, randomized_feat=RANDOM_FEATURE)

# Create checkpoint directory if not exists
os.makedirs(OUT_DIR, exist_ok=True)

for i in (pbar := trange(TRAINING_ITER+1)):
    x0, A0, sections, idx = pool.sample(BATCH_SIZE, device, replace_worst=True, loss_fn=loss_fn, img=img, degrade_prob=0.0)
    max_iter = STEPS_RANGE
    if STEPS_INCREMENT > 0:
        if i < STEPS_MEAN * STEPS_INCREMENT:
            max_iter = (i // STEPS_INCREMENT) + 1

    if USE_WRAP:
        loss, x, A = nca.sample_plane_wrapped(model, x0, A0, sections, H, DIMS, GMIN, GSIZE, iter_n=max_iter, loss_fn=loss_fn, img=img, optimizer=optimizer, scheduler=scheduler)
    else:
        loss, x, A = nca.sample_plane(model, x0, A0, sections, H, DIMS, iter_n=max_iter, loss_fn=loss_fn, img=img, optimizer=optimizer, scheduler=scheduler)

    pool.update(x, A, idx)
    pbar.set_description(f'loss: {loss:>7f}')
    
    if (i+1) % 1000 == 0:
        checkpoint_path = f'{OUT_DIR}/sphnca-{TIMESTAMP_STR}-{i+1:04d}.pt'
        torch.save({
            'configs': TRAINING_CONFIGS,
            'seed': seed,
            'model': model.state_dict(),
            'epoch': i+1,
            'loss': loss
        }, checkpoint_path)

print("Done!")

TRAIN_END_TIME = datetime.now()

FINISHED_TIME = datetime.now()

print(f'Training duration [{(TRAIN_END_TIME - TRAIN_START_TIME).seconds:>7f} s]')
print(f'Total duration [{(FINISHED_TIME - TIMESTAMP).seconds:>7f} s]')
