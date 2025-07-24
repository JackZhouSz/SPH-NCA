from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg19, VGG19_Weights
import sphops

import random

USE_3D = False

    

def cell_activity(A, use_alpha=True):
    alpha = A[...,3]
    if use_alpha:
        return alpha
    return torch.ones_like(alpha)

def default_feature_processs(features):
    """
    features: List of [(A: f[* N F], gA: f[* N F 3]]
    """
    return torch.cat(
        [As for As, _ in features]
        + [Av[...,0] for _, Av in features]
        + [Av[...,1] for _, Av in features]
    , dim=-1)

class SPHNCA(nn.Module):
    def __init__(self, in_features, fire_rate, update_rule='gated', hidden_features=256, base_feature_process=None, normalize_grad=False, use_alpha=True, normalize_perception=-1):
        super().__init__()
        self.in_features = in_features
        self.fire_rate = fire_rate
        self.update_rule = update_rule
        self.base_feature_process = base_feature_process
        if self.base_feature_process is None:
            self.base_feature_process = default_feature_processs
        self.normalize_grad = normalize_grad
        self.normalize_perception = normalize_perception
        self.use_alpha = use_alpha

        out_features = {
            'gated': in_features * 2 + 1,
            'orig': in_features
        }[self.update_rule]
        self.model = nn.Sequential(
            nn.Linear(in_features*3, hidden_features, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features, bias=True),
        )

        if self.update_rule == 'orig':
            self.model[-1].weight.data *= 0
            self.model[-1].bias.data *= 0

    def init_grid(self, x, sections, h, dims):
        grid = sphops.initialize_hashgrid(x, sections, h, dims)
        v = sphops.volume(x, h, grid)
        return grid, v

    def perceive(self, x, v, A, h, grid):
        gA = sphops.gradient(x, v, A, h, grid)
        if self.normalize_perception > 0:
            gA = h * gA * self.normalize_perception
        return [(A, gA)]

    def life_mask(self, x, v, activity, h, grid):
        if activity.shape[-1] > 1:
            activity = cell_activity(activity, self.use_alpha)[:,None]
        mask = activity > 0.1
        
        smoothed_mask = sphops.blur(x, v, mask.float(), h, grid).detach()
        smoothed_mask = (smoothed_mask > 0.1)
        
        return smoothed_mask

    def to_rgba(self, A):
        rgb = A[...,:3]
        a = cell_activity(A, self.use_alpha)[...,None]
        rgba = torch.cat([rgb, a], dim=-1)
        return rgba
    
    def forward(self, x, v, A, h, grid, fire_rate=None, feature_process=None):
        activity = cell_activity(A, self.use_alpha)[:,None]
        prev_mask = self.life_mask(x, v, activity, h, grid)
        
        features = self.perceive(x, v, A, h, grid)
        
        if feature_process is not None:
            features = feature_process(features)
        y = self.base_feature_process(features)
        dA = self.model(y)
        
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = torch.rand(x.shape[0], device=x.device) <= fire_rate

        if self.update_rule == 'gated':
            gate = torch.sigmoid(dA[..., :self.in_features])
            delta = torch.tanh(dA[..., self.in_features:-1])
            mult = torch.sigmoid(dA[...,[-1]])
            nA = A * gate + delta * mult
        elif self.update_rule == 'orig':
            nA = A + dA * self.fire_rate / fire_rate

        nA = torch.where(update_mask[...,None], nA, A)
        
        new_mask = self.life_mask(x, v, cell_activity(nA, self.use_alpha)[...,None], h, grid)
        
        living_mask = (prev_mask & new_mask)
        nA *= living_mask.float()

        return x, nA

    def normalize_grads(self):
        if self.normalize_grad:
            for p in self.parameters():
                p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad

def sample_plane(model, x, A, sections, h, dims, out_steps=False, iter_n=None, loss_fn=None, img=None, optimizer=None, scheduler=None, fire_rate=None):
    if iter_n is None:
        iter_n = torch.randint(iter_min, iter_max, (1,)).item()
    elif isinstance(iter_n, (tuple, list)):
        iter_min, iter_max = iter_n
        iter_n = torch.randint(iter_min, iter_max, (1,)).item()
    out = []

    xdim = x.shape[-1]
    if USE_3D and xdim == 2:
        x = F.pad(x, (0,1))
    
    grid, v = model.init_grid(x, sections, h, dims)

    for _ in range(iter_n):
        out.append(A.clone())
        x, A = model(x, v, A, h, grid, fire_rate=fire_rate, feature_process=None)
    out.append(A.clone())

    loss = 0.0
    if loss_fn is not None:
        if xdim == 2:
            x = x[...,:2]
            
        loss = loss_fn(x, A, img, A0=out[0])
        for nA in random.choices(out, k=4):
            loss += 0.1 * loss_fn(x, nA, img)
        loss.backward()
        model.normalize_grads()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        loss = loss.item()

    if USE_3D and xdim == 2:
        x = x[...,:2]
    if out_steps:
        return loss, x, out
    return loss, x, A


def get_wrapped_points_index(x, A, sections, gmin, gsize, h):
    """
    2 5 8 M
    1 4 7
    0 3 6 m
    m   M
    """
    from sphops.utils import pack, unpack
    xx = unpack(x, sections)
    AA = unpack(A, sections)

    gmax = gmin + gsize
    wrapped_points_idx = []
    for x, A in zip(xx, AA):
        x_mins = (x[:,0] <= gmin[0] + h).nonzero().squeeze().cpu().numpy()
        x_maxs = (x[:,0] >= gmax[0] - h).nonzero().squeeze().cpu().numpy()
        y_mins = (x[:,1] <= gmin[1] + h).nonzero().squeeze().cpu().numpy()
        y_maxs = (x[:,1] >= gmax[1] - h).nonzero().squeeze().cpu().numpy()

        idx = [
            torch.from_numpy(np.intersect1d(x_mins, y_mins, assume_unique=True)).to(x.device),
            torch.from_numpy(x_mins).to(x.device),
            torch.from_numpy(np.intersect1d(x_mins, y_maxs, assume_unique=True)).to(x.device),
            torch.from_numpy(y_mins).to(x.device),
            None,
            torch.from_numpy(y_maxs).to(x.device),
            torch.from_numpy(np.intersect1d(x_maxs, y_mins, assume_unique=True)).to(x.device),
            torch.from_numpy(x_maxs).to(x.device),
            torch.from_numpy(np.intersect1d(x_maxs, y_maxs, assume_unique=True)).to(x.device),
        ]
        wrapped_points_idx.append(idx)
    return wrapped_points_idx

def wrap_tensors(tensors, sections, idxs, gsize, x=None):
    from sphops.utils import pack, unpack

    new_sections = None
    def process(arr, is_position):
        nonlocal new_sections
        arr_unpacked = unpack(arr, sections)
        arr_wrapped = []
        for a, idx in zip(arr_unpacked, idxs):
            new_arr = [a]
            for k in range(9):
                if idx[k] is None: continue
                pad_arr = a[idx[k]]
                if is_position:
                    x_sign = 1 - (k // 3)
                    y_sign = 1 - (k % 3)
                    pad_arr[:,0] += x_sign * gsize[0]
                    pad_arr[:,1] += y_sign * gsize[1]
                new_arr.append(pad_arr)
            arr_wrapped.append(torch.cat(new_arr, dim=0))
        result, new_sections = pack(*arr_wrapped)
        return result

    new_tensors = [process(arr, False) for arr in tensors]
    if x is not None:
        new_x = process(x, True)
        return new_tensors, new_sections, new_x
    return new_tensors, new_sections

def unwrap_tensors(tensors, sections, orig_sections):
    from sphops.utils import pack, unpack
    
    # Unpack the new_x and new_A using the original sections
    def process(arr):
        arr_unpacked = unpack(arr, sections)
        arr_unwrapped = []
        for a, size in zip(arr_unpacked, orig_sections):
            arr_unwrapped.append(a[:size])
        result, _ = pack(*arr_unwrapped)
        return result

    new_tensors = [process(arr) for arr in tensors]
    return new_tensors


def sample_plane_wrapped(model, x, A, sections, h, dims, gmin, gsize, out_steps=False, iter_n=None, loss_fn=None, img=None, optimizer=None, scheduler=None, out_wrapped=False, fire_rate=None):
    """
    ...
    """
    if iter_n is None:
        iter_n = torch.randint(32, 48, (1,)).item()
    elif isinstance(iter_n, (tuple, list)):
        iter_min, iter_max = iter_n
        iter_n = torch.randint(iter_min, iter_max, (1,)).item()
    out = []
    out_wrapped = []

    xdim = x.shape[-1]
    if USE_3D and xdim == 2:
        x = F.pad(x, (0,1))

    wrap_idx = get_wrapped_points_index(x, A, sections, gmin, gsize, h)

    [A_wrapped], sections_wrapped, x_wrapped = wrap_tensors([A], sections, wrap_idx, gsize, x=x)

    grid, v_wrapped = model.init_grid(x_wrapped, sections_wrapped, h, dims)

    for _ in range(iter_n):
        out.append(A.clone())
        out_wrapped.append(A_wrapped.clone())
        x_wrapped, A_wrapped = model(x_wrapped, v_wrapped, A_wrapped, h, grid, fire_rate=fire_rate, feature_process=None)
        [A] = unwrap_tensors([A_wrapped], sections_wrapped, sections)
        [A_wrapped], sections_wrapped = wrap_tensors([A], sections, wrap_idx, gsize)
    out.append(A.clone())
    out_wrapped.append(A_wrapped.clone())

    loss = 0.0
    if loss_fn is not None:
        if xdim == 2:
            x = x[...,:2]
            
        loss = loss_fn(x, A, img, A0=out[0])
        for nA in random.choices(out, k=4):
            loss += 0.1 * loss_fn(x, nA, img)
        loss.backward()
        model.normalize_grads()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        loss = loss.item()

    if USE_3D and xdim == 2:
        x = x[...,:2]
    if out_steps:
        if out_wrapped:
            return loss, x, out, x_wrapped, out_wrapped
        return loss, x, out
    return loss, x, A


##############################################
def normalize(v):
    norm = torch.linalg.norm(v, dim=-1, keepdims=True)
    return v / (1e-8 + norm)

def orthogonalize(N, T):
    NT = (N * T).sum(-1, keepdims=True)
    T2 = T - N * NT
    return normalize(T2)

def diffuse(N, T, x, A, sections, h, dims, v=None, grid=None, lerp_multiplier=1.0, w_multiplier=1.0):
    if v is None or grid is None:
        grid = sphops.initialize_hashgrid(x, sections, h, dims)
        v = sphops.volume(x, h, grid)
    w = torch.clip(cell_activity(A)[...,None], 0, 1)
    
    m = torch.lerp(torch.ones_like(w), w, w_multiplier)
    mT = torch.cat([m, m*T], dim=-1)
    mT2 = sphops.blur(x, v, mT, h, grid)
    T2 = mT2[...,1:] / (1e-8 + mT2[...,[0]])
    T2 = torch.lerp(T2, T, w * lerp_multiplier)
    return orthogonalize(N, T2)

def project_tangent_space(Av, N, T):
    # Av: [N, C, 3]
    B = torch.linalg.cross(N, T)
    TBN = torch.stack([T, B, N], dim=-1) # [N, 3, 3]
    pAv = torch.matmul(Av, TBN)
    return pAv

def feature_process_tangent(N, T):
    def func(features):
        f = [(As, project_tangent_space(Av, N, T)) for As, Av in features]
        return f
    return func

def sample_mesh(model, x, A, N, T, sections, h, dims, out_steps=False, iter_n=None, loss_fn=None, img=None, optimizer=None, scheduler=None, lerp_multiplier=1.0, w_multiplier=1.0, out_np=False, iter_wrapper=None, fire_rate=None):
    if iter_n is None:
        iter_n = torch.randint(32, 48, (1,)).item()
    out = []

    xdim = x.shape[-1]
    if USE_3D and xdim == 2:
        x = F.pad(x, (0,1))
    
    grid, v = model.init_grid(x, sections, h, dims)

    iterator = range(iter_n)
    if iter_wrapper is not None:
        iterator = iter_wrapper(iterator)
    for _ in iterator:
        out.append(A.clone())
        # Perception vector is projected to the tangent space
        x, A = model(x, v, A, h, grid, fire_rate=fire_rate, feature_process=feature_process_tangent(N, T))
        # Spread the tangent vector across the surface
        T[:,:] = diffuse(N, T, x, A, sections, 0.1, 20, lerp_multiplier=lerp_multiplier, w_multiplier=w_multiplier).detach()
    out.append(A.clone())

    loss = 0.0
    if loss_fn is not None:
        if xdim == 2:
            x = x[...,:2]
        
        loss = loss_fn(x, A, img, A0=out[0])
        for nA in random.choices(out, k=4):
            loss += 0.1 * loss_fn(x, nA, img)
        loss.backward()
        model.normalize_grads()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        loss = loss.item()

    if out_np:
        out = [arr.cpu().numpy() for arr in out]
    if out_steps:
        return loss, x, out
    return loss, x, A

def add_radial_seed(x3, A3, N=None, T=None, R=0.2, idx=None, center=None):
    if idx is None:
        idx = np.random.randint(0, x3.shape[0])
    Rc3 = x3[idx,:]
    if center is not None:
        Rc3 = center

    Rc3d2 = ((x3-Rc3) ** 2).sum(dim=-1)
    w = torch.clamp(1-Rc3d2/R**2, 0, 1)**3
    texture = torch.ones_like(A3)
    A3 += texture * w[...,None]

    if N is not None:
        cpi = torch.argmin(Rc3d2)
        T0 = orthogonalize(N[cpi,:], torch.randn((x3.shape[-1],), device=T.device))
        T[cpi,:] = T0

class Pool:
    def __init__(self, seed, total_size, randomized_feat=False, feature_process=None):
        self.total_size = total_size
        self.num_points = seed[0].shape[0]
        self.seed = seed
        self.x = seed[0].cpu().repeat((self.total_size, 1, 1))
        self.A = seed[1].cpu().repeat((self.total_size, 1, 1))
        self.num_features = self.A.shape[-1]
        self.randomized_feat = randomized_feat
        self.feature_process = feature_process

        if self.feature_process:
            x, A = self.get_initial_feature()
            self.x[:] = x.cpu().repeat((self.total_size, 1, 1))
            self.A[:] = A.cpu().repeat((self.total_size, 1, 1))
        else:
            for i in range(self.total_size):
                x, A = self.get_initial_feature()
                self.x[i,:,:] = x.cpu()
                self.A[i,:,:] = A.cpu()
    
    def get_initial_feature(self):
        x, A = self.seed
        if self.randomized_feat:
            A = torch.rand_like(A)
        elif self.feature_process:
            x, A = self.feature_process(x, A)
        return x, A

    def sample(self, batch_size, device, replace_worst=False, loss_fn=None, img=None, degrade_prob=0.0, erase_R=0.0):
        perm = torch.randperm(self.total_size)
        idx = perm[:batch_size]
        x = self.x[idx,:,:].to(device)
        A = self.A[idx,:,:].to(device)
        sections = (self.num_points,) * batch_size

        if replace_worst:
            loss = loss_fn(x, A, img)
            loss = np.array(loss)
            loss_rank = torch.from_numpy(loss.argsort()[::-1].copy())
            x = x[loss_rank,...]
            A = A[loss_rank,...]
            idx = idx[loss_rank]
            x[0,:,:], A[0,:,:] = self.get_initial_feature()
        
        if degrade_prob > 0.0:
            filter = torch.rand_like(A[...,0]) < degrade_prob
            A[filter,:] = torch.rand_like(A[filter,:])
        if erase_R > 0.0:
            for b in range(x.shape[0]):
                i = torch.randint(0, x.shape[1]).item()
                c = x[b,i,:]
                d2 = ((x[b] - c) ** 2).sum(dim=-1)
                filter = d2 < erase_R ** 2
                A[b,filter,:] = 0.0
        return x.view(-1, 2), A.view(-1, self.num_features), sections, idx

    def update(self, x, A, idx):
        batch_size = idx.shape[0]
        self.x[idx,:,:] = x.view(batch_size,-1,2).detach().cpu()
        self.A[idx,:,:] = A.view(batch_size,-1,self.num_features).detach().cpu()

def get_rgba(A, margin=None):
    rgba = A[...,:4].clone()
    if margin is not None:
        rgba = rgba + (torch.clamp(rgba, 0-margin, 1+margin) - rgba).detach()
    return rgba

########################### CLIP loss

def get_clip_loss(text=None, device=None, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

    from torchvision.transforms import Resize, Normalize
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    resizer = Resize((224, 224))
    normalizer = Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)

    text_tokens = None
    text_features_pre = None
    if isinstance(text, str):
        text = [text]
    if text is not None:
        text_tokens = tokenizer(text).to(device)
        text_features_pre = model.encode_text(text_tokens, normalize=True).detach()
    
    def loss(img, text=None):
        """
        img: [B 4 H W]
        """
        nonlocal text_features_pre
        imgs_tensor = normalizer(resizer(img[:,:3,:,:]))

        image_features = model.encode_image(imgs_tensor, normalize=True)

        text_features = text_features_pre
        if text_features is None:
            with torch.no_grad():
                text_tokens = tokenizer(text).to(device)
                text_features = model.encode_text(text_tokens, normalize=True).detach()

        spherical_dist = (image_features - text_features).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        return spherical_dist.mean()
    return loss



########################### Style loss
cnn = None

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean.to(img.device)) / self.std.to(img.device)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a, b, c * d)  # resize F_XL into \hat F_XL

    G = torch.bmm(features, torch.transpose(features, -1, -2))  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


def BNC2BCHW(A, H, W, B=None):
    if B is not None:
        A = A.view(B, -1, A.shape[-1])
    A = torch.transpose(A, 1, 2)
    A = A.view(-1, A.shape[1], W, H)
    A = torch.transpose(A, -1, -2)
    return A

style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_loss(style_img, loss_module=None):
    if loss_module is None:
        loss_module = StyleLoss
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)
    style_losses = []
    
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    cnn.to(style_img.device)
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers_default:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = loss_module(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], loss_module):
            break

    model = model[:(i + 1)]

    model.eval()
    model.requires_grad_(False)

    return model, style_losses

def eval_style_loss(model, style_losses, input_img, layer_mean=False):
    model(input_img)
    style_score = 0

    for sl in style_losses:
        style_score += sl.loss
    if layer_mean:
        style_score /= len(style_layers_default)
    
    return style_score

class OptimalTransportLoss(nn.Module):
    """
    OG code: https://github.com/IVRL/MeshNCA/blob/main/losses/appearance_loss.py
    """
    MAX_SAMPLES = 1024

    def __init__(self, target_feature):
        super(OptimalTransportLoss, self).__init__()
        self.target = rearrange(target_feature, 'b c w h -> b c (w h)')

    @staticmethod
    def pairwise_distances_cos(x, y):
        """
        Pairwise Cosine distance between two flattened feature sets.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b, n, m)
        """
        x_norm = torch.norm(x, dim=2, keepdim=True)  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)  # (b, 1, m)
        dist = 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)  # (b, n, m)
        return dist
    
    @staticmethod
    def style_loss(x, y):
        """
        Relaxed Earth Mover's Distance (EMD) between two sets of features.
        :param x: (b, n, c)
        :param y: (b, m, c)
        :param metric: Either 'cos' or 'L2'

        :return: (b)
        """
        pd = OptimalTransportLoss.pairwise_distances_cos(x, y)
        m1, _ = pd.min(1)
        m2, _ = pd.min(2)
        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))
        return remd
    
    @staticmethod
    def moment_loss(x, y):
        """
        Calculates the distance between the first and second moments of two sets of features.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b)
        """
        mu_x = torch.mean(x, 1, keepdim=True)
        mu_y = torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))

        x_c = x - mu_x
        y_c = y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)

        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def forward(self, input):
        in_features = rearrange(input, 'b c w h -> b c (w h)')
        x, y = in_features, self.target

        b, c_x, n_x = x.shape
        b_y, c_y, n_y = y.shape
        
        batch_size = b // b_y
        assert batch_size * b_y == b, "Batch size must be a multiple of the number of target images"
        
        y = y.repeat(batch_size, 1, 1)
        n_samples = min(n_x, n_y, OptimalTransportLoss.MAX_SAMPLES)

        indices_x = torch.argsort(torch.rand(b, 1, n_x, device=x.device), dim=-1)[..., :n_samples]
        x = x.gather(-1, indices_x.expand(b, c_x, n_samples))

        indices_y = torch.argsort(torch.rand(b, 1, n_y, device=y.device), dim=-1)[..., :n_samples]
        y = y.gather(-1, indices_y.expand(b, c_y, n_samples))

        x = x.transpose(1, 2)  # (b, n_samples, c)
        y = y.transpose(1, 2)  # (b, n_samples, c)

        self.loss = (OptimalTransportLoss.style_loss(x, y) \
            + OptimalTransportLoss.moment_loss(x, y)).mean()
        return input
