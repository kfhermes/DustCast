from functools import partial

import torch
from diffusers.schedulers import DDIMScheduler
from fastprogress import progress_bar


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float32) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


@torch.no_grad()
def _diffusers_sampler(model, cond_frames, sched, same_noise=True, silent=False, **kwargs):
    model.eval()
    device = next(model.parameters()).device
    
    if device.type=='cpu':
        print("Converting model input to float32 for CPU inference.")
        cond_frames = cond_frames.float()

    new_frame = torch.randn_like(cond_frames[:,-3:], dtype=cond_frames.dtype, device=device)

    if same_noise:
        new_frame[:,-2] = new_frame[:,-3]
        new_frame[:,-1] = new_frame[:,-3]

    preds = []
    if silent:
        pbar = sched.timesteps
    else:
        pbar = progress_bar(sched.timesteps, leave=False)

    for t in pbar:
        pbar.comment = f"DDIM Sampler: frame {t}"
        cond_frames = torch.clamp(cond_frames, min=0.0, max=1.0)
        conditioned_input = torch.cat([cond_frames, new_frame], dim=1)
        noise = model(conditioned_input, t)
        new_frame = sched.step(noise, t, new_frame, **kwargs).prev_sample
        preds.append(new_frame)
    return preds[-1].float().cpu()


def ddim_sampler(steps=333, eta=1., **kwargs):
    ddim_sched = DDIMScheduler()
    ddim_sched.set_timesteps(steps)
    return partial(_diffusers_sampler, sched=ddim_sched, eta=eta, **kwargs)
