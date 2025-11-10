import numpy as np
import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

from dustcast.evaluation.mask import get_ct_mask, get_du_mask
from dustcast.evaluation.io import load_background_rgb_from_file
from dustcast.features.utils import mkdir_if_not_exists
from dustcast.features.rgb import bt_to_rgb


def plot_rgb_image_coords(opath, img, title, img_extent, dpi=150):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
    ax.set_extent(img_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cf.COASTLINE, alpha=0.75, color='black')
    ax.add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(opath, dpi=dpi)


def plot_rgb_image(opath, img, title, dpi=150):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, origin='upper')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(opath, dpi=dpi)


def generate_rgb_images(odir, prd_bt, time_init, time_valid, lat=None, lon=None, ens=0):
    time_size, ens_size, lat_size, lon_size, n_ch = prd_bt.shape

    img_prd = bt_to_rgb(prd_bt, gamma=[1.0, 2.5, 1.0], lims = [[-4,2],[0,15],[261,289]], channel_dim=2)

    leadtime_minutes = ((time_valid - time_init).total_seconds() / 60).astype(int)
    time_init_str = time_init.strftime('%Y%m%d-%H%M')

    timetitle = [f"{time_init_str} +{int(lt)}min" for lt in leadtime_minutes]
    timefile = [f"{time_init_str}_{int(lt)}.png" for lt in leadtime_minutes]

    mkdir_if_not_exists(odir)

    if lat is not None and lon is not None:
        print("Generating RGB images with coordinates.")
        for fr in range(time_size):
            opath = os.path.join(odir, timefile[fr])
            img_extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            plot_rgb_image_coords(opath, img_prd[fr, ens, :, :, :], timetitle[fr], img_extent)
    else:
        print("Generating RGB images without coordinates.")
        for fr in range(time_size):
            opath = os.path.join(odir, timefile[fr])
            plot_rgb_image(opath, img_prd[fr, ens, :, :, :], timetitle[fr])


def plot_P_image_coords(opath, img, title, img_extent, dpi=150):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    pcm = ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap='viridis')
    ax.set_extent(img_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cf.COASTLINE, alpha=0.75, color='black')
    ax.add_feature(cf.BORDERS, linestyle=':', alpha=0.75, color='black')
    ax.set_title(title, fontsize=12)
    plt.colorbar(pcm, ax=ax, orientation='vertical', label='Probability')
    plt.tight_layout()
    plt.savefig(opath, dpi=dpi)


def plot_P_image(opath, img, title, dpi=150):
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.imshow(img, origin='upper', cmap='viridis')
    ax.set_title(title, fontsize=12)
    plt.colorbar(pcm, ax=ax, orientation='vertical', label='Probability')
    plt.tight_layout()
    plt.savefig(opath, dpi=dpi)


def generate_P_images(odir, prd_bt, time_init, time_valid, idir_btbg, lat=None, lon=None, ens=0):
    time_size, ens_size, channel, lat_size, lon_size = prd_bt.shape
    try:
        rgb_bg = load_background_rgb_from_file(idir_btbg, time_init)
    except Exception as e:
        print(f"Error loading background RGB image: {e}")
        pass
    if rgb_bg is None:
        print("Performing dust detection without anomaly criteria")
        rgb_bg = np.ones((3, lat_size, lon_size), dtype=np.uint8) * 100 # dummy file, anomaly criteria will always yield 'True'

    prd_mask_ct = get_ct_mask(prd_bt, channel_dim=2)
    prd_mask_du = get_du_mask(prd_bt, rgb_bg, channel_dim=2)

    P_ct = np.mean(prd_mask_ct, axis=1)
    P_du = np.mean(prd_mask_du, axis=1)

    leadtime_minutes = ((time_valid - time_init).total_seconds() / 60).astype(int)
    time_init_str = time_init.strftime('%Y%m%d-%H%M')

    timetitle = [f"{time_init_str} +{int(lt)}min" for lt in leadtime_minutes]
    timefile = [f"{time_init_str}_{int(lt)}" for lt in leadtime_minutes]

    mkdir_if_not_exists(odir)

    if lat is not None and lon is not None:
        print("Generating RGB images with coordinates.")
        for fr in range(time_size):
            opath = os.path.join(odir, timefile[fr])
            img_extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            plot_P_image_coords(f'{opath}_du.png', P_du[fr], f"Predicted Dust - {timetitle[fr]}", img_extent)
            plot_P_image_coords(f'{opath}_ct.png', P_ct[fr], f"Predicted Convection - {timetitle[fr]}", img_extent)
    else:
        print("Generating RGB images without coordinates.")
        for fr in range(time_size):
            opath = os.path.join(odir, timefile[fr])
            plot_P_image(f'{opath}_du.png', P_du[fr], f"Predicted Dust - {timetitle[fr]}")
            plot_P_image(f'{opath}_ct.png', P_ct[fr], f"Predicted Convection - {timetitle[fr]}")