import torch
import lpips
from ignite.metrics import SSIM
from i2i_methods import sdedit, ilvr, egsde
import os
import shutil
from tqdm import tqdm
import math
from PIL import Image


def psnr_val(x,y,data_range = 1.0):
    dim = tuple(range(1, y.ndim))
    mse_error = torch.pow(x.double() - y.double(), 2).mean(dim=dim)
    psnr = 10.0 * torch.log10(data_range ** 2 / (mse_error + 1e-10))
    return psnr.sum()


def mse(x,y,data_range = 1.0):
    dim = tuple(range(1, y.ndim))
    mse_error = torch.pow(x.double() - y.double(), 2).sum(dim=dim)
    return mse_error.sum()


def lpips_val(x, y, vis_model):
    d = vis_model(x, y)
    return d.sum()


def save_model_samples(name, loader, num_samples):
    if os.path.exists(name):
        shutil.rmtree(name)

    os.makedirs(name, exist_ok=True)
    count = 0

    with tqdm(total= num_samples) as pbar:
        while count < num_samples:
            for out in loader:
                    
                batch_size = out.shape[0]
                cur_batch_size = min(num_samples - count, batch_size)
                
                out = (out * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for i in range(cur_batch_size):
                    img = Image.fromarray(out[i])
                    n_digits = len(str(count))
                    img_name = (6 - n_digits) * '0' + str(count) + '.png'
                    img.save(os.path.join(name, img_name))
                    count += 1
                    pbar.update(1)
                    pbar.set_description('%d images saved' % (count,))
                    
                if count >= num_samples:
                    break


def compute_metrics_and_save_imgs(name, src_loader, method, net, sampling_params, to_see=None):
    if os.path.exists(name):
        shutil.rmtree(name)

    os.makedirs(name, exist_ok=True)
    
    count = 0
    seen= 0
    ssim_metric = SSIM(data_range=1.0)
    vgg = lpips.LPIPS(net='vgg').to(sampling_params['device'])
    
    l2, psnr, lp, ssim = 0, 0, 0, 0
    
    for img_src in tqdm(src_loader):
        img_src = img_src.to(sampling_params['device'])

        
        if method == 'ilvr':
            pred = ilvr(net, img_src, sampling_params)
        elif method == 'sdedit':
            pred = sdedit(net, img_src, sampling_params)
        elif method == 'egsde':
            pred = egsde(net, img_src, sampling_params)

        l2 += mse(img_src, pred).item()
        lp += lpips_val(img_src, pred, vgg).item()
        ssim_metric.update([pred, img_src])
        psnr += psnr_val(img_src, pred).item()
        seen += img_src.shape[0]

        out = (pred * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i in range(out.shape[0]):
            img = Image.fromarray(out[i])
            n_digits = len(str(count))
            img_name = (6 - n_digits) * '0' + str(count) + '.png'
            img.save(os.path.join(name, img_name))
            count += 1
                  
        if seen >= to_see:
            break
            
    if to_see is None:
        l2 /= len(src_loader.dataset)
        lp /= len(src_loader.dataset)
        psnr /= len(src_loader.dataset)
        ssim = ssim_metric.compute()
    else:
        l2 /= seen
        lp /= seen
        psnr /= seen
        ssim = ssim_metric.compute()
    return {
        'SSIM': ssim,
        'L2': math.sqrt(l2),
        'LPIPS': lp,
        'PSNR': psnr
    }