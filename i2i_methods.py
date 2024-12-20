import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def cosine_similarity(X,Y):
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X)*norm(Y)
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity


def normalize(x):
    return x / x.abs().max(dim=0)[0][None, ...]

def velocity_from_denoiser(x, model, sigma, class_labels=None, error_eps=1e-4, stochastic=False, cfg=0.0, **model_kwargs):
    sigma = sigma[:, None, None, None]
    cond_v = (-model(x, sigma, class_labels, **model_kwargs) + x) / (sigma + error_eps)

    if cfg > 0.0:
        dummy_labels = torch.zeros_like(class_labels)
        dummy_labels[:, -1] = 1
        uncond_v = (-model(x, sigma, dummy_labels, **model_kwargs) + x) / (sigma + error_eps)
        v = cond_v + cfg * (cond_v - uncond_v)
    else:
        v = cond_v

    if stochastic:
        v = v * 2

    return v


def get_timesteps(params):
    num_steps = params['num_steps']
    sigma_min, sigma_max = params['sigma_min'], params['sigma_max']
    rho = params['rho']

    step_indices = torch.arange(num_steps, device=params['device'])
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps


def sample_euler(model, noise, params, class_labels=None, method='sdedit', x_src=None, **model_kwargs):
    num_steps = params['num_steps']
    vis_steps = params['vis_steps']
    t_steps = get_timesteps(params)
    x = noise * params['sigma_max']
    x_history = [normalize(noise)]
    
    if method=='ilvr':
        N = params['scale_factor']
        
    if method=='egsde':
        class_model = params['class_model']
        class_model.to(params['device'])
        class_model.eval()
        N = params['scale_factor']
        l_1 = params['l_1']
        l_2 = params['l_2']
        
   
    for i in range(len(t_steps) - 1):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        t_net = t_steps[i] * torch.ones(x.shape[0], device=params['device'])
        if method == 'egsde':
            x_src_t = x_src + torch.randn_like(x_src) * t_next
            x_copy = x.clone().detach().requires_grad_(True)
            dim = tuple(range(1, x.ndim))
            res = ((F.interpolate(F.interpolate(x_src_t, scale_factor=1 / N, mode='area'), scale_factor=N, mode='area') - F.interpolate(F.interpolate(x_copy, scale_factor=1 / N, mode='area'), scale_factor=N, mode='area'))**2).mean(dim=dim)
            proj_grad = (l_1 * torch.autograd.grad(res.sum(), x_copy)[0]).detach()

            class_cos = cosine_similarity(class_model(x_copy, t_net)[0], class_model(x_src_t, t_net)[0])
            class_grad = (l_2 * torch.autograd.grad(class_cos.sum(), x_copy)[0]).detach()
                
            with torch.no_grad():
                x = x + (velocity_from_denoiser(x, model, t_net, class_labels=class_labels, stochastic=params['stochastic'], cfg=params['cfg'], **model_kwargs) - proj_grad - class_grad) * (t_next - t_cur)
        else:
            with torch.no_grad():
                x = x + velocity_from_denoiser(x, model, t_net, class_labels=class_labels, stochastic=params['stochastic'], cfg=params['cfg'], **model_kwargs) * (t_next - t_cur)
                if params['stochastic']:
                    x = x + torch.randn_like(x) * torch.sqrt(torch.abs(t_next - t_cur) * 2 * t_cur)

                if method=='ilvr':
                    x_src_t = x_src + torch.randn_like(x_src) * t_next
                
                    x = x + F.interpolate(F.interpolate(x_src_t, scale_factor=1 / N, mode='area'), scale_factor=N, mode='area') - F.interpolate(F.interpolate(x, scale_factor=1 / N, mode='area'), scale_factor=N, mode='area')
                
        x_history.append(normalize(x).view(-1, 3, *x.shape[2:]))

    x_history = [x_history[0]] + x_history[::-(num_steps // (vis_steps - 2))][::-1] + [x_history[-1]]

    return x, x_history


def sdedit(model, x_source, params):
    noise = torch.randn_like(x_source)
    x = x_source + noise * params['sigma_max']
    out, _ = sample_euler(model, x / params['sigma_max'], params, class_labels=None, method='sdedit')
    return out


def ilvr(model, x_source, params):
    noise = torch.randn_like(x_source)
    out, _ = sample_euler(model, noise, params, class_labels=None, method='ilvr', x_src=x_source)
    return out


def egsde(model, x_source, params):
    noise = torch.randn_like(x_source)
    x = x_source + noise * params['sigma_max']
    out, _ = sample_euler(model, x / params['sigma_max'], params, class_labels=None, method='egsde', x_src=x_source)
    return out