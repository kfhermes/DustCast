import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from fastprogress import progress_bar

class SeviriDataset:
    def __init__(self, 
                 files, # list of paths to numpy files
                 num_frames=4, # how many consecutive frames to collate into one sample
                 num_channels=3, # channels per frame
                 img_size=64, # resize resolution
                 normlims=False, # custom limits for normalisation [min, max],
                 stack_chfr=False, # stack channels and frames into one dimension
                 load_into_memory=False, # load all data into memory
                 precision=torch.float16):

        self.files = files
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.img_size = img_size
        self.normlims = normlims
        self.stack_chfr = stack_chfr
        self.load_into_memory = load_into_memory
        self.precision = precision
        
        tfms = [T.Resize((self.img_size, self.img_size), antialias=True)] if self.img_size is not None else []
        self.tfms = T.Compose(tfms)
        self.load_all_data() if self.load_into_memory else None


    @staticmethod
    def _nan_to_mean(data):
        for dim in range(data.shape[0]):
            if np.isnan(data[dim, :, :]).all():
                nanmean = 0
                raise ValueError(f"Values are NaN in frame {dim}.")
            else:
                nanmean = np.nanmean(data[dim, :, :])
            data[dim,:,:] = np.nan_to_num(data[dim,:,:], nan=nanmean, posinf=nanmean, neginf=nanmean)
        return data

    @staticmethod
    def _normalize(batch, limits):
        if len(limits) == batch.shape[1]:
            for ii in range(batch.shape[1]):
                min_val = limits[ii][0]
                max_val = limits[ii][1]
                batch[:,ii,...] = (batch[:,ii,...]  - min_val) / (max_val - min_val)
            batch = np.clip(batch, 0, 1)
            return batch
        else:
            raise ValueError(f"Limits specified in self.framenorm ({limits}) do not match data channels ({batch.shape[1]}). Limits should be supplied in shape [[min1, max1],[min2,max2],...].") 
    
    def _stack(self, data):
        out = data.reshape(self.num_channels*self.num_frames, self.img_size, self.img_size)
        return out
    
    def get_file_datetime_info(self, idx):
        files_batch = self.files[idx : idx + self.num_frames]
        datestr = []
        for path in files_batch:
            datestr.append(os.path.splitext(os.path.basename(path))[0])
        files_dt = pd.to_datetime(datestr)
        
        t0 = files_dt[0]
        dt_expect = pd.date_range(start=t0, periods=self.num_frames, freq='15min')
        dt_avail = dt_expect.intersection(files_dt)
        dt_index = dt_expect.get_indexer(dt_avail)
        dt_bool = dt_expect.isin(dt_avail)
        return dt_expect, dt_index, dt_bool
    
    def check_files(self, idx):
        _, _, dt_bool = self.get_file_datetime_info(idx)
        if np.any(dt_bool):
            return idx
        else:
            # Return randomly selected instead
            idx_rand = np.random.randint(0, __len__(self), seed=42)
            print(f'Warning: No complete set of input data available for index {idx}. Returning data for randomly selected index: {idx_rand}')
            return idx_rand
    
    def load_data(self, idx):
        idx =  self.check_files(idx)
        data = []
        for ii in range(idx, idx + self.num_frames):
            try:
                d_frame = np.load(self.files[ii], allow_pickle=False)
                d_frame = self._nan_to_mean(d_frame)
            except:
                raise ValueError(f"Invalid frame {ii}, idx {idx}. Possibly all values are NaN.")
            if np.isnan(d_frame).any():
                raise ValueError(f"After filling: Values are NaN in frame {ii}, idx {idx}.")
            d_frame = np.nan_to_num(d_frame, nan=0, posinf=0, neginf=0)
            data.append(d_frame)
        one_set = np.stack(data, axis=0)
        one_set = self._normalize(batch = one_set, limits = self.normlims)
        one_set = self.tfms(torch.from_numpy(one_set))
        return one_set
    
    def load_all_data(self):
        all_data = []
        length = len(self.files) - self.num_frames + 1
        print(f'Loading {length} frames...')
        for idxs in (pbar:=progress_bar(range(length))): # length is equivalent to __len__
            data = self.load_data(idxs)
            all_data.append(data)
        self.data = torch.stack(all_data)
        print(f'Data shape:  {self.data.shape}')
        print(f'Data in mem: {self.data.nbytes/(1024**2):.1f} MiB')
        print('Data loaded.')

    def get_data(self, idx):
        if self.load_into_memory:
            return self.data[idx]
        else:
            return self.load_data(idx)
    
    def __len__(self): return len(self.files) - self.num_frames + 1
    
    def __getitem__(self, idx):
        out = self.get_data(idx)
        if self.stack_chfr:
            out = self._stack(out)
        return out.type(self.precision)
        
    
  
class SeviriDatasetEvaluation(SeviriDataset):
    def load_data(self, idx):
        dt_expect, dt_index, dt_bool = self.get_file_datetime_info(idx)
        data = []
        for ii in range(idx, idx+np.sum(dt_bool)):
            try:
                d_frame = np.load(self.files[ii], allow_pickle=False)
                d_frame = self._nan_to_mean(d_frame)
            except:
                dt_bool[ii-idx] = False
                print(f"Warning: Invalid frame {ii-idx}, idx {idx}. Flagging as missing.")

            data.append(d_frame)
        one_set = np.stack(data, axis=0)
        one_set = self._normalize(batch = one_set, limits = self.normlims)
        one_set = self.tfms(torch.from_numpy(one_set))
        return one_set, dt_expect, dt_index, dt_bool
    
    def load_all_data(self):
        all_data = []
        all_times = []
        all_dtidx = []
        all_avail = []
        length = len(self.files) - self.num_frames + 1
        print(f'Loading {length} frames...')
        for idxs in (pbar:=progress_bar(range(length))): # length is equivalent to __len__
            data, dt_expect, dt_index, dt_bool = self.load_data(idxs)
            all_data.append(data)
            all_times.append(dt_expect)
            all_dtidx.append(dt_index)
            all_avail.append(dt_bool)
        self.data = torch.stack(all_data, axis=0)
        self.times = all_times
        self.dtidx = all_dtidx
        self.avail = all_avail
        print(f'Data shape:  {self.data.shape}')
        print(f'Data in mem: {self.data.nbytes/(1024**2):.1f} MiB')
        print('Data loaded.')

    def get_data(self, idx):
        if self.load_into_memory:
            return self.data[idx], self.times[idx], self.dtidx[idx], self.avail[idx]
        else:
            return self.load_data(idx)
        
    def __getitem__(self, idx):
        """
        Retrieve and process data for a given index.

        This method loads the data for the specified index, processes it, and returns the data along with time and data availability.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple:
                - out (numpy.ndarray): The processed data array with values assigned to the indices where data is available.
                - dt (DatetimeIndex): Times of all obs. Hardcoded with 15 min spacing.
                - dt_bool (numpy.ndarray): Boolean array indicating which timesteps have valid data. True where obs available, False where filled with zeros.
        """
        data, dt_expect, dt_index, dt_bool = self.get_data(idx)
        out_shape = (len(dt_expect), *data.shape[1:])
        out = torch.zeros(out_shape, dtype=data.dtype)
        out[dt_index] = data
        
        if self.stack_chfr:
            out = out.reshape(self.num_channels*self.num_frames, self.img_size, self.img_size)

        return out.type(self.precision), dt_expect, dt_bool
        