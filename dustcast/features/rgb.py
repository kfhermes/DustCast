import copy
import numpy as np
import torch

def gammacorr(band, gamma):
    return np.power(band, 1.0/gamma)


def bt_to_rgb(bt, channel=['r', 'g', 'b'], gamma=[1.0, 1.0, 1.0], lims = [[0,0],[0,0],[0,0]], channel_dim=0):
    """"
    Convert brightness temperature (BT) to RGB.
    """
    bt = copy.deepcopy(bt)
    
    idx_r = channel.index('r')
    idx_g = channel.index('g')
    idx_b = channel.index('b')
    
    r = np.take(bt, idx_r,channel_dim)
    g = np.take(bt, idx_g,channel_dim)
    b = np.take(bt, idx_b,channel_dim)

    r_min, r_max = np.min(lims[0]), np.max(lims[0])
    g_min, g_max = np.min(lims[1]), np.max(lims[1])
    b_min, b_max = np.min(lims[2]), np.max(lims[2])

    r_n = (r - r_min) / (r_max-r_min)
    g_n = (g - g_min) / (g_max-g_min)
    b_n = (b - b_min) / (b_max-b_min)
    
    r_n = np.clip(r_n, 0, 1)
    g_n = np.clip(g_n, 0, 1)
    b_n = np.clip(b_n, 0, 1)
    
    r_ng=gammacorr(r_n, gamma[0])
    g_ng=gammacorr(g_n, gamma[1])
    b_ng=gammacorr(b_n, gamma[2])
    
    rgb = np.stack([r_ng, g_ng, b_ng], axis=0)
    rgb_out = np.clip(rgb, 0, 1)
    if not channel_dim == 0 :
        rgb_out.swapaxes(channel_dim,-1)
    return np.moveaxis(rgb_out, 0,-1)


def norm_to_bt(data_norm, normlims, channel_dim=0):
    """
    Convert normalized data back to brightness temperature (BT) using provided normalization limits.
    """
    data_bt = copy.deepcopy(data_norm)
    np.clip(data_norm, 0,1)
    norm = []
    for ii, ch in enumerate(normlims):
        norm.append(np.take(data_bt,ii,channel_dim) * (ch[1] - ch[0]) + ch[0])
    data_bt = np.stack(norm, axis=channel_dim)
    return data_bt


def resize_image(img_rgb_bg, custom_resolution):
    '''This is much faster with pytorch than numpy'''
    img_tensor = torch.from_numpy(img_rgb_bg).float()
    resized_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=custom_resolution, mode='bilinear', align_corners=False).squeeze(0)
    resized_image = resized_tensor.numpy()
    return resized_image