import numpy as np
from pathlib import Path

def get_coords(idir_coords, img_size):
    if idir_coords:
        path = Path(idir_coords)
        lon_path = list(path.rglob('*lon.npy'))
        lat_path = list(path.rglob('*lat.npy'))
        lon = np.load(lon_path[0])
        lat = np.load(lat_path[0])
        lon_img = np.linspace(lon[0], lon[-1], img_size)
        lat_img = np.linspace(lat[0], lat[-1], img_size)
        return dict(lat=lat_img, lon=lon_img)
    else:
        return dict(lat=None, lon=None)
    

def check_all_data_avail(bool_array):
    if np.all(bool_array):
        return True
    else:
        print(f"Input missing or incomplete.")
        return False


def check_time_is_selected(time, minutes):
    if time.minute in minutes:
        return True
    else:
        return False