import torch
import torch.nn.functional as F

from commons import geometry

import nca

def get_rgba(A, model, margin=None):
    rgba = model.to_rgba(A)
    # rgba = A[...,:4].clone()
    # # rgba[...,:3] = F.sigmoid(rgba[...,:3])
    # rgba[...,3] = cell_activity(A)
    if margin is not None:
        rgba = rgba + (torch.clamp(rgba, 0-margin, 1+margin) - rgba).detach()
    return rgba

def get_mse_loss(model, target_image, configs):
    IMAGE_SCALE = configs.get('IMAGE_SCALE')
    GMIN = configs.get('GMIN')
    GSIZE = configs.get('GSIZE')
    LOSS_WEIGHT_OVERFLOW = configs.get('LOSS_WEIGHT_OVERFLOW')

    def loss_fn(x, A, img, **kwargs):
        img_scale = kwargs.get('img_scale', IMAGE_SCALE)
        if x.dim() > 2:
            batch_size = x.shape[0]

            loss = [loss_fn(x[b,...], A[b,...], img).item() for b in range(batch_size)]
            return loss

        overflow_loss = A.abs().sub(1.).clamp(min=0).sum() if LOSS_WEIGHT_OVERFLOW > 0 else 0

        img_gmin = GMIN * img_scale
        img_gsize = GSIZE * img_scale
        img_x = geometry.bilinear_sample(x, img, img_gmin, img_gsize, dim=-3)
        rgba = get_rgba(A, model)
        
        f_loss = F.mse_loss
        loss = f_loss(rgba, img_x)

        return loss + LOSS_WEIGHT_OVERFLOW*overflow_loss
    return loss_fn

def get_ot_loss(model, target_image, configs):
    IMAGE_SIZE = configs.get('IMAGE_SIZE')
    LOSS_WEIGHT_STYLE = configs.get('LOSS_WEIGHT_STYLE')
    LOSS_WEIGHT_COLOR = configs.get('LOSS_WEIGHT_COLOR')
    LOSS_WEIGHT_OVERFLOW = configs.get('LOSS_WEIGHT_OVERFLOW')

    img_x = nca.BNC2BCHW(target_image[...,:3], IMAGE_SIZE, IMAGE_SIZE, B=1)
    style_model, style_losses = nca.get_style_loss(img_x, loss_module=nca.OptimalTransportLoss)
    
    def loss_fn(x, A, img, **kwargs):
        if x.dim() > 2:
            batch_size = x.shape[0]

            loss = [loss_fn(x[b,...], A[b,...], img).item() for b in range(batch_size)]
            return loss
        
        batch_size = x.shape[0] // IMAGE_SIZE**2

        overflow_loss = A.abs().sub(1.).clamp(min=0).sum() if LOSS_WEIGHT_OVERFLOW > 0 else 0

        rgba = get_rgba(A, model)
        rgb = nca.BNC2BCHW(rgba[...,:3], IMAGE_SIZE, IMAGE_SIZE, batch_size)
        style_loss = nca.eval_style_loss(style_model, style_losses, rgb, True)

        color_loss = F.l1_loss(rgb, img_x)
        return LOSS_WEIGHT_STYLE*style_loss + LOSS_WEIGHT_COLOR*color_loss + LOSS_WEIGHT_OVERFLOW*overflow_loss
    return loss_fn

def get_clip_loss(model, target_text, configs):
    IMAGE_SIZE = configs.get('IMAGE_SIZE')
    CLIP_SCALES = configs.get('CLIP_SCALES')
    DEVICE = configs.get('DEVICE')
    LOSS_WEIGHT_CLIP = configs.get('LOSS_WEIGHT_CLIP')
    LOSS_WEIGHT_OVERFLOW = configs.get('LOSS_WEIGHT_OVERFLOW')

    clip_loss_fn = nca.get_clip_loss(target_text, DEVICE)

    from torchvision.transforms import Resize, RandomCrop
    resizers = [Resize(int(IMAGE_SIZE / s)) if s > 1 else RandomCrop(int(IMAGE_SIZE * s)) for s in CLIP_SCALES]
    
    def loss_fn(x, A, img, **kwargs):
        if x.dim() > 2:
            batch_size = x.shape[0]

            loss = [loss_fn(x[b,...], A[b,...], img).item() for b in range(batch_size)]
            return loss
        
        batch_size = x.shape[0] // IMAGE_SIZE**2
        overflow_loss = (A-0.5).abs().sub(0.5).clamp(min=0).sum() if LOSS_WEIGHT_OVERFLOW > 0 else 0

        rgba = get_rgba(A, model, margin=0)
        rgb = nca.BNC2BCHW(rgba[...,:3], IMAGE_SIZE, IMAGE_SIZE, batch_size)
        
        clip_losses = [clip_loss_fn(resizer(rgb)) for resizer in resizers]
        clip_loss = sum(clip_losses) / len(clip_losses) if LOSS_WEIGHT_CLIP > 0 else 0

        return LOSS_WEIGHT_CLIP*clip_loss + LOSS_WEIGHT_OVERFLOW*overflow_loss

    return loss_fn
