import netCDF4
import numpy as np
import joblib
import os
from itertools import product

import torch.nn
import torch.utils.data as data


class EarthMantleDataset(data.Dataset):
    features = ['vx', 'vy', 'vz',
                'temperature', 'temperature anomaly',
                'thermal conductivity', 'thermal expansivity',
                'spin transition-induced density anomaly']
    x_features = ['vx', 'vy', 'vz',
                  'temperature', 'temperature anomaly',
                  'thermal conductivity', 'thermal expansivity']
    y_features = ['spin transition-induced density anomaly']

    def __init__(self, data_file_path: str,
                 depth: int, height: int, width: int,
                 scaler_dir: str):
        self.data_file_path = data_file_path
        self.netcdf = netCDF4.Dataset(self.data_file_path)
        self.keys = self.netcdf.variables.keys()
        
        self.depth = depth
        self.height = height
        self.width = width
        d, h, w = self.depth, self.height, self.width

        self.channel_in = len(self.x_features)
        self.channel_out = len(self.y_features)

        self.r_len = self.netcdf['r'].shape[0]
        self.lat_len = self.netcdf['lat'].shape[0]  # 위도(height)
        self.lon_len = self.netcdf['lon'].shape[0]  # 경도(width)

        self.volume_size = (self.r_len, self.lat_len, self.lon_len)

        self.x_size = (self.channel_in, self.r_len+2*d, self.lat_len+2*h, self.lon_len+2*w)
        self.y_size = (self.channel_out, self.r_len, self.lat_len, self.lon_len)

        # (Channel, Depth, Height, Width)
        self.x = np.zeros(self.x_size, dtype=np.float32)
        self.y = np.zeros(self.y_size, dtype=np.float32)

        for c, k in enumerate(self.x_features):
            sc = joblib.load(os.path.join(scaler_dir, f'{k}.sc'))
            _data = np.transpose(self.netcdf[k][:], (1, 0, 2))
            self.x[c, d:-d, h:-h, w:-w] = (_data - sc.mean_) / sc.scale_

        for c, k in enumerate(self.y_features):
            sc = joblib.load(os.path.join(scaler_dir, f'{k}.sc'))
            _data = np.transpose(self.netcdf[k][:], (1, 0, 2))
            self.y[c, :] = (_data - sc.mean_) / sc.scale_

        self.netcdf.close()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_x_size(self):
        return self.x_size

    def get_y_size(self):
        return self.y_size


if __name__ == '__main__':
    file_path = 'D:/EarthMantleConvection/mantle01/spherical001.nc'
    dataset = EarthMantleDataset(file_path, 5, 5, 5,
                                 './scalers')
    print(dataset.get_x_size())
    print(dataset.get_y_size())
