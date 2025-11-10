import numpy as np
import os
import random
import time

import torch
from fastprogress import progress_bar
from torch.utils.data import ConcatDataset

from dustcast.features.datasets import SeviriDataset


def mkdir_if_not_exists(odir):
    if not os.path.exists(odir):
        os.makedirs(odir)


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stack_frames_channels(data):
    "Stack the first two dimensions of a 4D array (num_frames, num_channels, img_size_x, img_size_y) into one dimension."
    assert len(data.shape) == 4 , f"Input with 4 dimensions expected, received {len(data.shape)}"
    num_frames, num_channels, img_size_x, img_size_y  = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
    dimensions = (num_frames * num_channels, img_size_x, img_size_y)
    data = data.reshape(dimensions)
    return data


def sample_full_images(model, sampler, start_frames, n_future_frames, n_channels):
    assert start_frames.shape[1] == 3*n_channels
    frames = start_frames
    model.eval()
    print(f'Sample full next frames of validation batch: {frames.shape}')
    for ii in range(n_future_frames):
        print(f"Generating frame {ii} ...")
        input_frames = frames[:,-3*n_channels:,...]
        new_frame = sampler(model, input_frames)
        frames = torch.cat([frames, new_frame.to(frames.device)], dim=1)

    forecast_frames = frames[:,-n_future_frames*n_channels:,...]
    return forecast_frames


def generateDatasets(files:list, frac_train: float = 0.8, len_block: int = 1000, **kwargs):
    train_ds, valid_ds = [], []

    len_total = len(files)
    n_blocks = int(np.floor(len_total/len_block))
    
    train_len = int(frac_train*len_block)
    valid_len = int((1-frac_train)*len_block)
    
    print(f"Creating Dataset from {n_blocks} train/valid blocks of length {train_len}/{valid_len} each. Not using remaining {len_total-n_blocks*len_block} steps.")
    
    for i in range(n_blocks):
        train_block = files[:train_len]
        valid_block = files[train_len:train_len+valid_len]
        
        train_ds.append(SeviriDataset(files=train_block, **kwargs))
        valid_ds.append(SeviriDataset(files=valid_block, **kwargs))
        
        # Remove used data from original df
        files = files[train_len+valid_len:]

    # Concatenate blocks into datasets for train and eval
    train_ds = ConcatDataset(train_ds)
    valid_ds = ConcatDataset(valid_ds)
    print(f"len(train_ds) = {len(train_ds)}")
    print(f"len(valid_ds) = {len(valid_ds)}")
    return train_ds, valid_ds


def generate_forecast_ens(model, sampler, input_frames, n_future_frames, ens_size, n_channels=3):
    "Generate ensemble forecast, number of input frames fixed to 3."
    model.eval()

    if model.device.type == 'cpu':
        print("Converting model input to float32 for CPU inference.")
        input_frames = input_frames.float()
        
    fcframes = []    
    start = time.perf_counter()
    frames = input_frames.repeat(ens_size,1,1,1) # speed up with putting ens input into batch dimension
    print(f"Generating forecast:\n {n_future_frames} future frames \n {ens_size} ensemble members\n")

    for ii in progress_bar(range(n_future_frames), leave=False):
        new_frame = sampler(model, frames[:,-3*n_channels:,...])
        fcframes.append(new_frame.to(frames.device))
        frames = torch.cat([frames, new_frame.to(frames.device)], dim=1)
    
    end = time.perf_counter()
    time_complete = end - start
    print(f'Complete prediction took: {time_complete:.02f}s')
    fcframes = torch.stack(fcframes, dim=0).cpu()
    return fcframes
