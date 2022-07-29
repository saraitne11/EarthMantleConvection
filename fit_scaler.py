import netCDF4
import joblib
import os

from sklearn.preprocessing import StandardScaler

file_path = 'D:/EarthMantleConvection/mantle01/spherical001.nc'
netcdf = netCDF4.Dataset(file_path)

features = ['vx', 'vy', 'vz',
            'temperature', 'temperature anomaly',
            'thermal conductivity', 'thermal expansivity',
            'spin transition-induced density anomaly']

dir_path = './scalers'
os.makedirs(dir_path, exist_ok=True)
for f in features:
    d = netcdf[f][:]
    x, y, z = d.shape
    d = d.reshape((x * y * z, 1))

    sc = StandardScaler()
    sc.fit(d)
    joblib.dump(sc, os.path.join(dir_path, f'{f}.sc'))
