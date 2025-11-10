from pathlib import Path

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset, num2date


def fill_nan_with_mean(data):
    idx = np.isnan(data)
    data[idx] = np.nanmean(data)
    return data


def find_files_in_dir(idir, pattern='*', recursive=False):
    path = Path(idir)
    if recursive:
        files = path.rglob(pattern)
    else:
        files = path.glob(pattern)
    return sorted(list(files))


def sort_len_str(files:list):
    return sorted(files, key=lambda x: (len(str(x)), str(x)))


def load_nowcast_from_file(file):
    ds = Dataset(file)
    r = ds.variables['btd120_108'][:]
    g = ds.variables['btd108_087'][:]
    b = ds.variables['bt108'][:]
    r_obs = ds.variables['btd120_108_obs'][:]
    g_obs = ds.variables['btd108_087_obs'][:]
    b_obs = ds.variables['bt108_obs'][:]
    r_inp = ds.variables['btd120_108_inp'][:]
    g_inp = ds.variables['btd108_087_inp'][:]
    b_inp = ds.variables['bt108_inp'][:]
    obs_flag = ds.variables['obs_flag'][:]
    time_valid = ds.variables['time_valid']
    time_input = ds.variables['time_input']
    time_init = ds.getncattr("time_init")
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    prd_bt = np.stack((r, g, b), axis=2)
    obs_bt = np.stack((r_obs, g_obs, b_obs), axis=1)
    inp_bt = np.stack((r_inp, g_inp, b_inp), axis=1)
    time_input = num2date(time_input[:], units=time_input.units, calendar=getattr(time_input, "calendar", "standard"))
    time_input = pd.to_datetime([t.isoformat() for t in time_input])
    time_valid = num2date(time_valid[:], units=time_valid.units, calendar=getattr(time_valid, "calendar", "standard"))
    time_valid = pd.to_datetime([t.isoformat() for t in time_valid])
    time_init = pd.to_datetime(time_init)
    return inp_bt, prd_bt, obs_bt, obs_flag, time_input, time_valid, time_init, lat, lon


def load_background_rgb_from_file(dir, time_init):
    path = Path(dir)
    filepath = path.rglob(f"*{time_init.strftime('%Y%m%d')}*.nc")
    filepath = list(filepath)
    if len(filepath) == 1:
        ds = Dataset(filepath[0])
        r = (ds['IR_120'][:] - ds['IR_108'])[0,...]
        g = (ds['IR_108'][:] - ds['IR_087'])[0,...]
        b = ds['IR_108'][0,...]
        r = fill_nan_with_mean(r)
        g = fill_nan_with_mean(g)
        b = fill_nan_with_mean(b)
        rgb = np.stack((r, g, b))
        return rgb
    else:
        print(f"rgb_bg file unavailable or more than one file for timestep, files found: {filepath}")
        return None
    

def save_metrics(fnm, cnt, metrics_data):
    print(f'Saving temporary output for {cnt} forecasts to: {fnm}')
    tmp_df = pd.DataFrame(metrics_data)
    tmp_df.to_hdf(fnm, key='df', mode='w')
    with pd.HDFStore(fnm, mode='w') as store:
        store['df'] = tmp_df
