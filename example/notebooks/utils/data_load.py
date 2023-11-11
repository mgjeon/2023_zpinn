import netCDF4
import numpy as np

def load_nc(filename):
    nc = netCDF4.Dataset(filename, 'r')

    nc_bx = nc.variables['Bx']
    bx = np.array(nc_bx[:].transpose(2, 1, 0))
    nc_by = nc.variables['By']
    by = np.array(nc_by[:].transpose(2, 1, 0))
    nc_bz = nc.variables['Bz']
    bz = np.array(nc_bz[:].transpose(2, 1, 0))

    b = np.stack([bx, by, bz], -1)

    return b 
    