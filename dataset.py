import netCDF4
import numpy as np
import joblib
import os
from itertools import product

import torch.nn
import torch.utils.data as data

from typing import Tuple


FEATURES = ['vx', 'vy', 'vz',
            'temperature', 'temperature anomaly',
            'thermal conductivity', 'thermal expansivity',
            'spin transition-induced density anomaly']
X_FEATURES = ['vx', 'vy', 'vz',
              'temperature', 'temperature anomaly',
              'thermal conductivity', 'thermal expansivity']
Y_FEATURES = ['spin transition-induced density anomaly']


def read_cdf(netcdf_file: str,
             depth: int, height: int, width: int,
             scaler_dir: str):
    netcdf = netCDF4.Dataset(netcdf_file)
    if depth % 2 == 0 or height % 2 == 0 or width % 2 == 0:
        raise ValueError('Input dimension must be odd')

    do = depth // 2     # depth offset
    ho = height // 2    # height offset
    wo = width // 2     # width offset

    in_channels = len(X_FEATURES)
    out_channels = len(Y_FEATURES)

    r_len = netcdf['r'].shape[0]
    lat_len = netcdf['lat'].shape[0]    # 위도(height)
    lon_len = netcdf['lon'].shape[0]    # 경도(width)

    volume_size = r_len, lat_len, lon_len

    x_volume_size = (in_channels,
                     r_len + do + do,
                     lat_len + ho + ho,
                     lon_len + wo + wo)
    y_volume_size = (out_channels,
                     r_len + do + do,
                     lat_len + ho + ho,
                     lon_len + wo + wo)

    # (Channel, Depth, Height, Width)
    x_volume = np.zeros(x_volume_size, dtype=np.float32)
    y_volume = np.zeros(y_volume_size, dtype=np.float32)

    for c, k in enumerate(X_FEATURES):
        sc = joblib.load(os.path.join(scaler_dir, f'{k}.sc'))
        _data = np.transpose(netcdf[k][:], (1, 0, 2))
        x_volume[c, do:-do, ho:-ho, wo:-wo] = (_data - sc.mean_) / sc.scale_

    for c, k in enumerate(Y_FEATURES):
        sc = joblib.load(os.path.join(scaler_dir, f'{k}.sc'))
        _data = np.transpose(netcdf[k][:], (1, 0, 2))
        y_volume[c, do:-do, ho:-ho, wo:-wo] = (_data - sc.mean_) / sc.scale_

    netcdf.close()
    return x_volume, y_volume, volume_size


class EarthMantleDataset(data.Dataset):
    def __init__(self,
                 x_volume: np.ndarray,
                 y_volume: np.ndarray,
                 volume_size: Tuple[int, int, int],
                 netcdf_file=''):
        self.x_volume = torch.Tensor(x_volume)
        self.y_volume = torch.Tensor(y_volume)

        self.in_channels = x_volume.shape[0]
        self.depth = x_volume.shape[1] - volume_size[0] + 1
        self.height = x_volume.shape[2] - volume_size[1] + 1
        self.width = x_volume.shape[3] - volume_size[2] + 1

        self.netcdf_file = netcdf_file

        do = self.depth // 2    # depth offset
        ho = self.height // 2   # height offset
        wo = self.width // 2    # width offset

        self.r_len, self.lat_len, self.lon_len = volume_size

        self.indices = list(product(range(do, self.r_len+do),
                                    range(ho, self.lat_len+ho),
                                    range(wo, self.lon_len+wo)))
        self.dl, self.du = do, do + 1
        self.hl, self.hu = ho, ho + 1
        self.wl, self.wu = wo, wo + 1

    def __getitem__(self, index):
        """
        x: (channel_in, depth, height, width)
        y: (channel_out, depth)
        """
        ix, iy, iz = self.indices[index]
        x = self.x_volume[:, ix - self.dl:ix + self.du, iy - self.hl:iy + self.hu, iz - self.wl:iz + self.wu]
        y = self.y_volume[:, ix - self.dl:ix + self.du, iy, iz]
        y = y.view(-1)
        return x, y, (ix, iy, iz)

    def __len__(self):
        return len(self.indices)

    def get_volume_size(self):
        return self.r_len, self.lat_len, self.lon_len

    def get_x_volume_size(self):
        return self.x_volume.shape

    def get_y_volume_size(self):
        return self.y_volume.shape


def my_test():
    file_path = 'D:/EarthMantleConvection/mantle01/spherical001.nc'
    x_volume, y_volume, volume_size = read_cdf(file_path, 5, 5, 5, './scalers')
    dataset = EarthMantleDataset(x_volume, y_volume, volume_size, file_path)
    print(dataset.get_volume_size())
    print(dataset.get_x_volume_size())
    print(dataset.get_y_volume_size())
    print(len(dataset))

    _x, _y, _idx = dataset[0]
    print(_x.shape, _y.shape, _idx)

    _x, _y, _idx = dataset[-1]
    print(_x.shape, _y.shape, _idx)

    loader = data.DataLoader(dataset=dataset, batch_size=128, num_workers=2, shuffle=True)
    for x, y, idx in loader:
        print(x.shape)
        print(y.shape)
        print(torch.stack(idx, dim=1).shape)
        break


if __name__ == '__main__':
    my_test()
