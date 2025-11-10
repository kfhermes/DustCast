import os
from tempfile import NamedTemporaryFile

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import wandb


def to_wandb_image(img):
    "Convert a tensor to a wandb.Image"
    return wandb.Image(torch.cat(img.split(1), dim=-1).cpu().numpy())


def log_images_to_wandb(logger, name_str, obs, samples, plot_frame=0, num_channels=3, dim_cat=2):
    "Log sampled images to wandb"
    device = samples.device
    istart = plot_frame*num_channels
    istop = (plot_frame+1)*num_channels
    img_top = obs[:, -num_channels:].to(device)
    img_bot = samples[:,istart:istop]
    frames = torch.cat([img_top, img_bot], dim=dim_cat)
    wandb_image_list = [to_wandb_image(img) for img in frames]
    logger.log({name_str: wandb_image_list})


def save_trim(filename, format='png', dpi=300, bbox_inches='tight'):
    plt.tight_layout()
    plt.savefig(f'{filename}.{format}', format=format, dpi=dpi, bbox_inches=bbox_inches)
    command = f"convert {filename}.{format} -trim +repage {filename}.{format}"
    with os.popen(command) as process:
        output = process.read()
        
        
def save_trim_ani(filename, animation, dpi=300, fps = 2):
    plt.tight_layout()
    animation.save(f"{filename}.gif", writer="imagemagick", dpi = dpi, fps = fps)
    command = f"convert {filename}.gif -coalesce -repage 0x0 -trim +repage {filename}.gif"
    with os.popen(command) as process:
        output = process.read()


def wandb_trim(label, dpi=300, bbox_inches='tight'):
    with NamedTemporaryFile(suffix=".png") as f:
        plt.tight_layout() 
        plt.savefig(f.name, dpi=dpi, bbox_inches=bbox_inches)
        command = f"convert {f.name} -trim +repage {f.name}"
        with os.popen(command) as process:
            output = process.read()
        wandb.log({label: wandb.Image(f.name)})


def wandb_trim_ani(label, animation, dpi=300, fps = 2):
    with NamedTemporaryFile(suffix=".gif") as f:
        plt.tight_layout()
        animation.save(f.name, writer="imagemagick", dpi=dpi, fps=fps)
        command = f"convert {f.name} -coalesce -repage 0x0 -trim +repage {f.name}"
        with os.popen(command) as process:
            output = process.read()
        wandb.log({label: wandb.Image(f.name)})
        
        
def wandb_plot_ens(label, config, obs, prd, times=None, frame=-1, rgb=False, img_extent=[-180,180,-90,90]):
    # If selected last frame (-1), get index of last frame
    if frame == -1:
        frame = obs.shape[0]-1
    num_plots = config.ens + 1
    
    # Set number of columns based on the number of plots
    if num_plots <= 6:
        n_cols = num_plots  # If 6 or fewer, plot all in one row
        n_rows = 1
    else:
        n_cols = min(max(2, num_plots // 2), 6)  # At least 2 columns, but no more than 6
        n_rows = 2  # Always 2 rows if more than 6 plots
    # Set figure dimensions dynamically, adjusting the size per plot
    fig_width = n_cols * 4 
    fig_height = n_rows * 4.25#3.25
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.ravel()

    if rgb:
        img_obs = obs[frame]
    else:
        img_obs = obs[frame, 0]
        
    axs[0].clear()
    axs[0].imshow(img_obs, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')
    axs[0].set_extent(img_extent, crs=ccrs.PlateCarree())
    axs[0].add_feature(cf.COASTLINE, alpha=0.75, color='black')
    axs[0].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
    if times is not None:
        axs[0].set_title(f'obs\n{times[frame]}')
    for ii in range(config.ens):
        iax = ii+1
        
        if rgb:
            img_prd = prd[frame, ii]
        else:
            img_prd = prd[frame, ii, 0]
        
        axs[iax].clear()
        axs[iax].imshow(img_prd, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')
        
        axs[iax].set_title(f'pred_ens_{ii}    +{(frame+1)*15} min')
            
        axs[iax].set_extent(img_extent, crs=ccrs.PlateCarree())
        axs[iax].add_feature(cf.COASTLINE, alpha=0.75, color='black')
        axs[iax].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black') 
    wandb_trim(label)


def wandb_ct_obs_predP(label, img_obs, img_prd, img_extent):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the initial observation and prediction images
    im0 = axs[0].imshow(img_obs, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), vmin=0, vmax=1)
    im1 = axs[1].imshow(img_prd, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), vmin=0, vmax=1)

    # Set the extent and add map features for the first frame
    for ax in axs[:-1]:
        ax.set_extent(img_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cf.COASTLINE, alpha=0.75, color='black')
        ax.add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')

    # Set the third axis for the colorbar
    axs[2].remove()
    ax_cbar = fig.add_axes([0.665, 0.325, 0.01, 0.35])  # [left, bottom, width, height]
    cbar = fig.colorbar(im0, cax=ax_cbar, orientation='vertical')
    cbar.set_label('P(IR108 < -40°C)')
    wandb_trim(label)
    
    
def wandb_ani(label, config, obs_in_bt, obs_bt, prd_bt, times_in, times, img_extent, opt=None):
    rng = range(0, config.num_frames+config.future_frames)
    
    fig, axs =  plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    ens_pos = 1 #0
    def update(frame):
        if opt=='rgb':
            if frame < config.num_frames:
                fr = frame
                img_prd = obs_in_bt[fr]
                img_obs = obs_in_bt[fr]
            else:
                fr = frame-config.num_frames
                img_prd = prd_bt[fr, ens_pos] # plot first ens member at position 0
                img_obs = obs_bt[fr]
        else:
            if frame < config.num_frames:
                fr = frame
                img_prd = obs_in_bt[fr, 0,...]
                img_obs = obs_in_bt[fr, 0,...]
            else:
                fr = frame-config.num_frames
                img_prd = prd_bt[fr, 0, 0,...] # plot first ens member at position 0
                img_obs = obs_bt[fr, 0,...]
                
                
        # print(f'img_prd {img_prd.shape}')
        # print(f'img_obs {img_obs.shape}')

        axs[0].clear()
        axs[1].clear()
        axs[0].imshow(img_obs, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')
        axs[1].imshow(img_prd, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')
        
        if frame < config.num_frames:
            axs[0].set_title(f'input {fr}\n {times_in[fr]}')
            axs[1].set_title(f'input {fr}\n {times_in[fr]}')
        else:
            axs[0].set_title(f'obs\n {times[fr]}')
            axs[1].set_title(f'pred\n +{(fr+1)*15} min ({fr+1} frames)')
            
        axs[0].set_extent(img_extent, crs=ccrs.PlateCarree())
        axs[1].set_extent(img_extent, crs=ccrs.PlateCarree())
        axs[0].add_feature(cf.COASTLINE, alpha=0.75, color='black')
        axs[1].add_feature(cf.COASTLINE, alpha=0.75, color='black')
        axs[0].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
        axs[1].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=rng, interval=500, blit=False)
    wandb_trim_ani(label=label, animation=ani)
    


def wandb_ani_ens(label, config, obs_bt, prd_bt, times, img_extent, opt=None):
    rng = range(0, config.future_frames)  
    num_plots = config.ens + 1
    if num_plots <= 6:
        n_cols = num_plots
        n_rows = 1
    else:
        n_cols = min(max(2, num_plots // 2), 6) 
        n_rows = 2
    fig_width = n_cols * 4 
    fig_height = n_rows * 5.25#3.25
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.ravel()

    def update(frame):
        if opt=='rgb':
            img_obs = obs_bt[frame]
        else:
            img_obs = obs_bt[frame, 0,...]
        axs[0].clear()
        axs[0].imshow(img_obs, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')
        axs[0].set_title(f'obs\n{times[frame]}')
        axs[0].set_extent(img_extent, crs=ccrs.PlateCarree())
        axs[0].add_feature(cf.COASTLINE, alpha=0.75, color='black')
        axs[0].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
        
        for ii in range(config.ens):
            iax = ii+1      
            if opt=='rgb':
                img_prd = prd_bt[frame, ii]
            else:
                img_prd = prd_bt[frame, ii, 0,...]
            # print(f'img_prd {img_prd.shape}')
            # print(f'img_obs {img_obs.shape}')
            axs[iax].clear()
            axs[iax].imshow(img_prd, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')
            axs[iax].set_title(f'pred_ens_{ii}    +{(frame+1)*15} min')
            axs[iax].set_extent(img_extent, crs=ccrs.PlateCarree())
            axs[iax].add_feature(cf.COASTLINE, alpha=0.75, color='black')
            axs[iax].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')   
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=rng, interval=500, blit=False)
    wandb_trim_ani(label=label, animation=ani)


def wandb_ani_ctP(label, config, ct_mask_obs, ct_P, times, img_extent):
    rng = range(0, config.future_frames)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    img_prd = ct_P[0]
    img_obs = ct_mask_obs[0]
    
    # Plot first frame
    im0 = axs[0].imshow(img_obs, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), vmin=0, vmax=1)
    im1 = axs[1].imshow(img_prd, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), vmin=0, vmax=1)
    for ax in axs[:-1]:
        ax.set_extent(img_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cf.COASTLINE, alpha=0.75, color='black')
        ax.add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
    
    # Third axis for colorbar
    axs[2].remove()
    ax_cbar = fig.add_axes([0.665, 0.325, 0.01, 0.35]) # [left, bottom, width, height]
    cbar = fig.colorbar(im0, cax=ax_cbar, orientation='vertical')
    cbar.set_label('P(IR108 < -40°C)')
    
    # Update the data in the imshow plots without re-creating them
    def update(frame):
        img_prd = ct_P[frame]
        img_obs = ct_mask_obs[frame]
        im0.set_array(img_obs)
        im1.set_array(img_prd)
        axs[0].set_title(f'obs\n {times[frame]}')
        axs[1].set_title(f'pred\n +{(frame+1)*15} min ({frame+1} frames)')
        return [im0, im1]
    ani = animation.FuncAnimation(fig, update, frames=rng, interval=500, blit=False)
    wandb_trim_ani(label=label, animation=ani)
    

def wandb_log_mapplot(label, data, extent):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
    ax.imshow(data, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap='Greys')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cf.COASTLINE, alpha=0.75, color='black')
    ax.add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
    wandb_trim(label)


def wandb_ani_ens_at_frame(label, config, obs, prd, time, img_extent, opt=None):
    rng = range(0, config.ens)
    
    fig, axs =  plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    def update(ens):
        if opt=='rgb':
            img_prd = prd[ens]
            img_obs = obs
        else:
            img_prd = prd[ens, 0, ...]
            img_obs = obs[0, ...]

        # print(f'img_prd {img_prd.shape}')
        # print(f'img_obs {img_obs.shape}')

        axs[0].clear()
        axs[1].clear()
        axs[0].imshow(img_obs, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')
        axs[1].imshow(img_prd, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='Greys')

        axs[0].set_title(f'obs\n {time}')
        axs[1].set_title(f'ens_{ens}')
            
        axs[0].set_extent(img_extent, crs=ccrs.PlateCarree())
        axs[1].set_extent(img_extent, crs=ccrs.PlateCarree())
        axs[0].add_feature(cf.COASTLINE, alpha=0.75, color='black')
        axs[1].add_feature(cf.COASTLINE, alpha=0.75, color='black')
        axs[0].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
        axs[1].add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')

    ani = animation.FuncAnimation(fig, update, frames=rng, interval=1500, blit=False)
    wandb_trim_ani(label=label, animation=ani, fps=0.75)