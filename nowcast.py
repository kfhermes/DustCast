import os
from datetime import datetime, timedelta

import fire
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset, date2num
from omegaconf import OmegaConf

from dustcast.evaluation.utils import get_coords
from dustcast.features.datasets import SeviriDataset
from dustcast.features.utils import generate_forecast_ens, mkdir_if_not_exists, set_seed, stack_frames_channels
from dustcast.features.rgb import norm_to_bt
from dustcast.features.plot import generate_rgb_images, generate_P_images
from dustcast.model.setup import UNet2D, get_parameters_for_UNet2DModel
from dustcast.model.utils_ddpm import ddim_sampler
from dustcast.evaluation.io import find_files_in_dir



def get_files(arg_init, idir, n_inframes, frame_spacing=15):    
    input_datetime = pd.to_datetime(arg_init)

    start_time = input_datetime - timedelta(minutes=frame_spacing*(n_inframes-1))
    date_range = pd.date_range(start=start_time, end=input_datetime, inclusive='both', freq=f'{frame_spacing}min')

    files_avail = find_files_in_dir(idir)
    files_avail = np.array(files_avail)

    time_avail = [path.stem for path in files_avail]
    time_avail = pd.to_datetime(time_avail, format='%Y%m%d-%H%M')

    idx_select = np.in1d(time_avail, date_range)
    ifl_select = files_avail[idx_select]

    if len(ifl_select) != len(date_range):
        raise ValueError(f"Not all required files are available for the specified initial time {arg_init}.")
    return ifl_select


def create_variable(nc, name, dtype, dimensions, long_name, units=None, zlib=True, complevel=4):
    """
    Helper function to create a NetCDF variable with attributes.
    """
    var = nc.createVariable(name, dtype, dimensions, zlib=zlib, complevel=complevel)
    var.long_name = long_name
    if units:
        var.units = units
    return var


def save_nowcast_netcdf(
    odir,
    btd120_108,
    btd108_087,
    bt108,
    time_init,
    time_valid,
    lat=None,
    lon=None,
    time_units="hours since 2000-01-01 00:00:00",
    calendar="gregorian",
    compression_level=4,
):
    time_size, ens_size, lat_size, lon_size = btd120_108.shape

    time_valid_numeric = date2num(time_valid.to_pydatetime(), units=time_units, calendar=calendar)

    output_file = f"{time_init.strftime('%Y%m%d-%H%M')}.nc"
    mkdir_if_not_exists(odir)
    opath = os.path.join(odir, output_file)

    if lat is None:
        print("Warning: No lat coordinates specified. Writing dummy values.")
        lat = np.arange(lat_size)
    if lon is None:
        print("Warning: No lon coordinates specified. Writing dummy values.")
        lon = np.arange(lon_size)

    try:
        with Dataset(opath, "w", format="NETCDF4") as nc:
            nc.createDimension("time", time_size)
            nc.createDimension("ens", ens_size)
            nc.createDimension("lat", lat_size)
            nc.createDimension("lon", lon_size)

            time_fc_var = create_variable(nc, "time_valid", "f8", ("time",), "Valid time of forecast", time_units)
            lat_var = create_variable(nc, "lat", "f4", ("lat",), "Latitude", "degrees_north")
            lon_var = create_variable(nc, "lon", "f4", ("lon",), "Longitude", "degrees_east")

            btd120_108_var = create_variable(
                nc, "btd120_108", "f4", ("time", "ens", "lat", "lon"),
                "Predicted brightness temperatures difference between IR120 and IR108.", "K", zlib=True, complevel=compression_level
            )
            btd108_087_var = create_variable(
                nc, "btd108_087", "f4", ("time", "ens", "lat", "lon"),
                "Predicted brightness temperatures difference between IR108 and IR087.", "K", zlib=True, complevel=compression_level
            )
            bt108_var = create_variable(
                nc, "bt108", "f4", ("time", "ens", "lat", "lon"),
                "Predicted brightness temperatures at IR108.", "K", zlib=True, complevel=compression_level
            )

            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            nc.title = "dust-cast nowcast"
            nc.summary = (
                "Nowcast of SEVIRI brightness temperature channels and brightness temperature differences "
                "with the 'dust-cast' model."
            )
            nc.history = f"File created on {current_time}"
            nc.time_init = time_init.strftime("%Y-%m-%d %H:%M:%S")

            time_fc_var[:] = time_valid_numeric
            lat_var[:] = lat
            lon_var[:] = lon
            btd120_108_var[:, :, :, :] = btd120_108
            btd108_087_var[:, :, :, :] = btd108_087
            bt108_var[:, :, :, :] = bt108

        if os.path.exists(output_file):
            print(f"NetCDF file '{output_file}' created successfully.")

    except Exception as e:
        print(f"Error creating NetCDF file '{output_file}': {e}")


def _setup(
        arg_init=None,
        idir_data='.',
        load_model_from_checkpoint=False,
        load_model_from_artifact=False,
        idir_coords='.',
        n_inframes=3,
        sampler_steps=100,
        same_noise=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        img_size=128,
        model_name='unet_256_rgb',
        normlims = [[-4, 2], [0, 15], [200, 320]],
        is_state_dict=False,
        frame_spacing=15,
    ):

    files = get_files(arg_init=arg_init, idir=idir_data, n_inframes=n_inframes, frame_spacing=frame_spacing)
    init_ds = SeviriDataset(files=files, num_frames=n_inframes, img_size=img_size, normlims=normlims, stack_chfr=False)
    sampler = ddim_sampler(sampler_steps, same_noise=same_noise, silent=False)
    model_params = get_parameters_for_UNet2DModel(model_name)

    if load_model_from_artifact:
        print(f"Loading model from wandb artifact: {load_model_from_artifact}")
        model = UNet2D.from_wandb_artifact(model_params, load_model_from_artifact, is_state_dict=is_state_dict).to(device)
    elif load_model_from_checkpoint:
        model = UNet2D.from_lightning_ckpoint(model_params, load_model_from_checkpoint, is_state_dict=is_state_dict).to(device)
    else:
        raise ValueError("No checkpoint or wandb artifact specified. Please provide one to load the model weights.")

    coords = get_coords(idir_coords, img_size)
    return init_ds, model, sampler, coords


def generate_nowcast(
        arg_init='20240607T1200',
        idir_data='.',
        idir_btbg='.',
        odir='./nowcasts',
        load_model_from_checkpoint=False,
        load_model_from_artifact=False,
        is_state_dict=False,
        idir_coords=False,
        model_name='unet_512_rgb',
        sampler_steps=100,
        img_size=256,
        n_inframes=3,
        n_channels=3,
        n_fcframes=8,
        n_ens=5,
        frame_spacing=15,
        same_noise=False,
        normlims=[[-4, 2], [0, 15], [200, 320]],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_netcdf=True,
        save_plots_rgb=True,
        save_plots_P=True,
        ):
    
    init_ds, model, sampler, coords = _setup(
        arg_init=arg_init,
        idir_data=idir_data,
        load_model_from_checkpoint=load_model_from_checkpoint,
        load_model_from_artifact=load_model_from_artifact,
        is_state_dict=is_state_dict,
        idir_coords=idir_coords,
        n_inframes=n_inframes,
        sampler_steps=sampler_steps,
        same_noise=same_noise,
        device=device,
        img_size=img_size,
        model_name=model_name,
        frame_spacing=frame_spacing,
        normlims=normlims,
        )
    
    time_init = pd.to_datetime(arg_init)
    input = stack_frames_channels(init_ds[0])
    input = input.clone().detach().to(device)
    
    # Generate forecasts
    with torch.amp.autocast(model.device.type):
        forecast_ens = generate_forecast_ens(
            model=model,
            sampler=sampler,
            input_frames=input,
            n_future_frames=n_fcframes,
            n_channels=n_channels,
            ens_size=n_ens,
            )

    # Prepare data for saving
    prd = forecast_ens.cpu().numpy()
    prd_bt = norm_to_bt(prd, normlims, channel_dim=2)
    time_valid = [time_init + pd.Timedelta(minutes=frame_spacing*(i+1)) for i in range(n_fcframes)]
    time_valid = pd.DatetimeIndex(time_valid)
    
    if save_netcdf:
        save_nowcast_netcdf(
            odir=odir,
            btd120_108=prd_bt[:,:,0,:,:],
            btd108_087=prd_bt[:,:,1,:,:],
            bt108=prd_bt[:,:,2,:,:],
            time_init=time_init,
            time_valid=time_valid,
            **coords
            )
    if save_plots_rgb:
        generate_rgb_images(odir=odir, prd_bt=prd_bt, time_init=time_init, time_valid=time_valid, **coords)

    if save_plots_P:
        generate_P_images(odir=odir, prd_bt=prd_bt, time_init=time_init, time_valid=time_valid, idir_btbg=idir_btbg, **coords)



def main(config='./dustcast/config/nowcast.yaml', **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    set_seed(1, reproducible=True)
    generate_nowcast(**config)

if __name__ == "__main__":
    fire.Fire(main)