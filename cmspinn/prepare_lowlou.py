# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_lowlou_bottom.ipynb.

# %% auto 0
__all__ = ['PrepareLowLou']

# %% ../nbs/06_lowlou_bottom.ipynb 3
import os
import numpy as np
import matplotlib.pyplot as plt

from .lowlou_mag import LowLouMag

# %% ../nbs/06_lowlou_bottom.ipynb 5
class PrepareLowLou:
    def __init__(self, base_path, bounds=[-1, 1, -1, 1, 0, 2], Nx=64, Ny=64, Nz=64, n=1, m=1, l=0.3, Phi=np.pi/3, abs=True):
        self.lowlou_path = os.path.join(base_path, f'lowlou_{Nx}_{Ny}_{Nz}')
        os.makedirs(self.lowlou_path, exist_ok=True)
        self.bounds = bounds
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.n = n
        self.m = m
        self.l = l
        self.Phi = Phi
        self.abs = abs

    def cal_and_save_lowlou(self):
        b_path = os.path.join(self.lowlou_path, 'b.vtk')
        if os.path.exists(b_path):
            print(f'Exist {b_path}')
            return
        lowlou = LowLouMag(bounds=self.bounds, resolutions=[self.Nx, self.Ny, self.Nz], n=self.n, m=self.m, l=self.l, Phi=self.Phi, abs=self.abs) 
        lowlou.calculate()
        lowlou.redefine_mesh()
        self.grid = lowlou.newgrid 
        self.bottom = lowlou.b_bottom

        self.grid.save(b_path)
        print(f"Saved {b_path}")

        return self.grid 
    
    def plot_and_save_lowlou_bottom(self, plot=True):
        os.makedirs(os.path.join(self.lowlou_path, 'b_bottom'), exist_ok=True)
        b_bottom_path = os.path.join(os.path.join(self.lowlou_path, 'b_bottom'), 'b_bottom.npy')
        if os.path.exists(b_bottom_path):
            print(f'Exist {b_bottom_path}')
            return
        
        b_bottom = np.array(self.bottom)

        with open (b_bottom_path, 'wb') as f:
            np.save(f, b_bottom)
        print(f"Saved {b_bottom_path}")

        if plot is True:
            plt.close()
            fig, ax = plt.subplots(figsize=(6,6))
            CS = plt.imshow(b_bottom[:, :, 2].transpose(), origin='lower', cmap='gray')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(CS, label=r'$B_z$'+f'(z={0})')
            plt.show()
