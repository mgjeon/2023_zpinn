# %% [markdown]
# # Weights and Biases

# %%
from setproctitle import setproctitle
setproctitle("SPINN")

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

# %% [markdown]
# ## Packages

# %%
import jax 
import jax.numpy as jnp
from jax import jvp
import optax
from flax import linen as nn 
from flax.training.early_stopping import EarlyStopping

from typing import Sequence
from functools import partial

import numpy as np

import os
import time
import pickle
from tqdm import trange

import random

import pyvista as pv
import pickle
from cmspinn.mag_viz import create_coordinates

from cmspinn.potential_field import cal_and_save_potential_boundary_for_spinn
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from cmspinn.evaluation import magnetic_energy, divergence, magnitude, curl, laplacian_vector

import glob

# %%
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out
    
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state

def curlx(apply_fn, params, x, y, z):
    # curl vector w/ forward-mode AD
    # w_x = uz_y - uy_z
    vec_z = jnp.ones(z.shape)
    vec_y = jnp.ones(y.shape)
    uy_z = jvp(lambda z: apply_fn(params, x, y, z)[1], (z,), (vec_z,))[1]
    uz_y = jvp(lambda y: apply_fn(params, x, y, z)[2], (y,), (vec_y,))[1]
    wx = uz_y - uy_z
    return wx


def curly(apply_fn, params, x, y, z):
    # curl vector w/ forward-mode AD
    # w_y = ux_z - uz_x
    vec_z = jnp.ones(z.shape)
    vec_x = jnp.ones(x.shape)
    ux_z = jvp(lambda z: apply_fn(params, x, y, z)[0], (z,), (vec_z,))[1]
    uz_x = jvp(lambda x: apply_fn(params, x, y, z)[2], (x,), (vec_x,))[1]
    wy = ux_z - uz_x
    return wy

def curlz(apply_fn, params, x, y, z):
    # curl vector w/ forward-mode AD
    # w_z = uy_x - ux_y
    vec_y = jnp.ones(y.shape)
    vec_x = jnp.ones(x.shape)
    ux_y = jvp(lambda y: apply_fn(params, x, y, z)[0], (y,), (vec_y,))[1]
    uy_x = jvp(lambda x: apply_fn(params, x, y, z)[1], (x,), (vec_x,))[1]
    wz = uy_x - ux_y
    return wz

# %%
class SPINN3d(nn.Module):
    features: Sequence[int]
    r: int
    out_dim: int
    pos_enc: int
    mlp: str
    activation: str

    @nn.compact
    def __call__(self, x, y, z):
        '''
        inputs: input factorized coordinates
        outputs: feature output of each body network
        xy: intermediate tensor for feature merge btw. x and y axis
        pred: final model prediction (e.g. for 2d output, pred=[u, v])
        '''
        if self.pos_enc != 0:
            # positional encoding only to spatial coordinates
            freq = jnp.expand_dims(jnp.arange(1, self.pos_enc+1, 1), 0)
            x = jnp.concatenate((jnp.ones((x.shape[0], 1)), jnp.sin(x@freq), jnp.cos(x@freq)), 1)
            y = jnp.concatenate((jnp.ones((y.shape[0], 1)), jnp.sin(y@freq), jnp.cos(y@freq)), 1)
            z = jnp.concatenate((jnp.ones((z.shape[0], 1)), jnp.sin(z@freq), jnp.cos(z@freq)), 1)

            # causal PINN version (also on time axis)
            #  freq_x = jnp.expand_dims(jnp.power(10.0, jnp.arange(0, 3)), 0)
            # x = x@freq_x
            
        inputs, outputs, xy, pred = [x, y, z], [], [], []
        init = nn.initializers.glorot_normal()

        if self.mlp == 'mlp':
            for X in inputs:
                for fs in self.features[:-1]:
                    X = nn.Dense(fs, kernel_init=init)(X)
                    if self.activation == "tanh":
                        X = nn.activation.tanh(X)
                    elif self.activation == "sin":
                        X = jnp.sin(X)
                X = nn.Dense(self.r*self.out_dim, kernel_init=init)(X)
                outputs += [jnp.transpose(X, (1, 0))]

        elif self.mlp == 'modified_mlp':
            for X in inputs:
                if self.activation == "tanh":
                    U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                    V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                    H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                elif self.activation == "sin":
                    U = jnp.sin(nn.Dense(self.features[0], kernel_init=init)(X))
                    V = jnp.sin(nn.Dense(self.features[0], kernel_init=init)(X))
                    H = jnp.sin(nn.Dense(self.features[0], kernel_init=init)(X))
                for fs in self.features[:-1]:
                    Z = nn.Dense(fs, kernel_init=init)(H)
                    if self.activation == "tanh":
                        Z = nn.activation.tanh(Z)
                    elif self.activation == "sin":
                        Z = jnp.sin(Z)
                    H = (jnp.ones_like(Z)-Z)*U + Z*V
                H = nn.Dense(self.r*self.out_dim, kernel_init=init)(H)
                outputs += [jnp.transpose(H, (1, 0))]
        
        for i in range(self.out_dim):
            xy += [jnp.einsum('fx, fy->fxy', outputs[0][self.r*i:self.r*(i+1)], outputs[1][self.r*i:self.r*(i+1)])]
            pred += [jnp.einsum('fxy, fz->xyz', xy[i], outputs[-1][self.r*i:self.r*(i+1)])]

        if len(pred) == 1:
            # 1-dimensional output
            return pred[0]
        else:
            # n-dimensional output
            return pred

# %%
@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def generate_train_data(nx, ny, nz, n_max_x, n_max_y, n_max_z):

    xc = jnp.linspace(0, n_max_x, nx).reshape(-1, 1)
    yc = jnp.linspace(0, n_max_y, ny).reshape(-1, 1)
    zc = jnp.linspace(0, n_max_z, nz).reshape(-1, 1)

    # # boundary points
    xb = [jnp.linspace(0, n_max_x, nx).reshape(-1, 1), # z=0   bottom
          jnp.linspace(0, n_max_x, nx).reshape(-1, 1), # z=2   top
          jnp.array([[0.]]),                     # x=0   lateral_1
          jnp.array([[n_max_x]]),                     # x=2   lateral_2
          jnp.linspace(0, n_max_x, nx).reshape(-1, 1), # y=0   lateral_3
          jnp.linspace(0, n_max_x, nx).reshape(-1, 1)] # y=2   lateral_4

    yb = [jnp.linspace(0, n_max_y, ny).reshape(-1, 1), 
          jnp.linspace(0, n_max_y, ny).reshape(-1, 1), 
          jnp.linspace(0, n_max_y, ny).reshape(-1, 1), 
          jnp.linspace(0, n_max_y, ny).reshape(-1, 1), 
          jnp.array([[0.]]), 
          jnp.array([[n_max_y]])]

    zb = [jnp.array([[0.]]), 
          jnp.array([[n_max_z]]), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1)]

    return xc, yc, zc, xb, yb, zb

# %%
@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
def generate_train_data_random(key, nx, ny, nz, n_max_x, n_max_y, n_max_z, nc, ncx=None, ncy=None, ncz=None, choice=False):
    
    keys = jax.random.split(key, 4)

    if (ncx is not None) and (ncy is not None) and (ncz is not None):
      xc = jax.random.uniform(keys[1], (ncx, 1), minval=0., maxval=n_max_x)
      yc = jax.random.uniform(keys[2], (ncy, 1), minval=0., maxval=n_max_x)
      zc = jax.random.uniform(keys[3], (ncz, 1), minval=0., maxval=n_max_x)
    
    elif choice is False:
      xc = jax.random.uniform(keys[1], (nc, 1), minval=0., maxval=n_max_x)
      yc = jax.random.uniform(keys[2], (nc, 1), minval=0., maxval=n_max_x)
      zc = jax.random.uniform(keys[3], (nc, 1), minval=0., maxval=n_max_x)
    else:
      xc = jnp.linspace(0, n_max_x, nx).reshape(-1, 1)
      yc = jnp.linspace(0, n_max_y, ny).reshape(-1, 1)
      zc = jnp.linspace(0, n_max_z, nz).reshape(-1, 1)
      xc = jax.random.choice(keys[1], xc, shape=(nc,))
      yc = jax.random.choice(keys[2], yc, shape=(nc,))
      zc = jax.random.choice(keys[3], zc, shape=(nc,))

    # boundary points
    xb = [jnp.linspace(0, n_max_x, nx).reshape(-1, 1), # z=0   bottom
          jnp.linspace(0, n_max_x, nx//4).reshape(-1, 1), # z=2   top
          jnp.array([[0.]]),                     # x=0   lateral_1
          jnp.array([[n_max_x]]),                     # x=2   lateral_2
          jnp.linspace(0, n_max_x, nx//4).reshape(-1, 1), # y=0   lateral_3
          jnp.linspace(0, n_max_x, nx//4).reshape(-1, 1)] # y=2   lateral_4

    yb = [jnp.linspace(0, n_max_y, ny).reshape(-1, 1), 
          jnp.linspace(0, n_max_y, ny//4).reshape(-1, 1), 
          jnp.linspace(0, n_max_y, ny//4).reshape(-1, 1), 
          jnp.linspace(0, n_max_y, ny//4).reshape(-1, 1), 
          jnp.array([[0.]]), 
          jnp.array([[n_max_y]])]

    zb = [jnp.array([[0.]]), 
          jnp.array([[n_max_z]]), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1), 
          jnp.linspace(0, n_max_z, nz).reshape(-1, 1)]

    return xc, yc, zc, xb, yb, zb

# %%
@partial(jax.jit, static_argnums=(0,))
def apply_model_spinn(apply_fn, params, train_boundary_data, w_ff, w_div, w_bc):
    def residual_loss(params, x, y, z, w_ff, w_div):
        # calculate u
        Bx, By, Bz = apply_fn(params, x, y, z)
        B = jnp.stack([Bx, By, Bz], axis=-1)
        
        # calculate J
        Jx = curlx(apply_fn, params, x, y, z)
        Jy = curly(apply_fn, params, x, y, z)
        Jz = curlz(apply_fn, params, x, y, z)
        J = jnp.stack([Jx, Jy, Jz], axis=-1)

        JxB = jnp.cross(J, B, axis=-1) 

        #-----------------------------------------------------------
        # loss_ff = jnp.sum(JxB**2, axis=-1)
        # loss_ff = jnp.mean(loss_ff)

        loss_ff = jnp.sum(JxB**2, axis=-1) / (jnp.sum(B**2, axis=-1) + 1e-7)
        loss_ff = jnp.mean(loss_ff)

        # loss_ff = jnp.mean(JxB**2)
        #-----------------------------------------------------------

        # tangent vector dx/dx
        # assumes x, y, z have same shape (very important)
        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)
        vec_z = jnp.ones(z.shape)
        
        Bx_x = jvp(lambda x: apply_fn(params, x, y, z)[0], (x,), (vec_x,))[1]
        # Bx_y = jvp(lambda y: apply_fn(params, x, y, z)[0], (y,), (vec,))[1]
        # Bx_z = jvp(lambda z: apply_fn(params, x, y, z)[0], (z,), (vec,))[1]

        # By_x = jvp(lambda x: apply_fn(params, x, y, z)[1], (x,), (vec,))[1]
        By_y = jvp(lambda y: apply_fn(params, x, y, z)[1], (y,), (vec_y,))[1]
        # By_z = jvp(lambda z: apply_fn(params, x, y, z)[1], (z,), (vec,))[1]

        # Bz_x = jvp(lambda x: apply_fn(params, x, y, z)[2], (x,), (vec,))[1]
        # Bz_y = jvp(lambda y: apply_fn(params, x, y, z)[2], (y,), (vec,))[1]
        Bz_z = jvp(lambda z: apply_fn(params, x, y, z)[2], (z,), (vec_z,))[1]

        divB = Bx_x + By_y + Bz_z
        
        #-----------------------------------------------------------
        # loss_div = jnp.sum((divB)**2, axis=-1)
        # loss_div = jnp.mean(loss_div)

        loss_div = jnp.mean((divB)**2)
        #-----------------------------------------------------------

        loss = w_ff*loss_ff + w_div*loss_div

        return loss, loss_ff, loss_div

    def boundary_loss(params, x, y, z, *boundary_data):
        
        # loss = 0.
        # for i in np.arange(4):
        #     boundary_data_batched = boundary_batches[i, :, :, :]
        #     xb = boundary_data_batched[:, 0, :][:, 0].reshape(-1, 1)
        #     yb = boundary_data_batched[:, 0, :][:, 1].reshape(-1, 1)
        #     zb = boundary_data_batched[:, 0, :][:, 2].reshape(-1, 1)

        #     Bx, By, Bz = apply_fn(params, xb, yb, zb)
        #     # Bx, By, Bz = Bx.reshape(-1, 1), By.reshape(-1, 1), Bz.reshape(-1, 1)

        #     Bxb = boundary_data_batched[:, 1, :][:, 0].reshape(-1, 1)
        #     Byb = boundary_data_batched[:, 1, :][:, 1].reshape(-1, 1)
        #     Bzb = boundary_data_batched[:, 1, :][:, 2].reshape(-1, 1)

        #     Bxb_mesh, Byb_mesh, Bzb_mesh = jnp.meshgrid(Bxb.ravel(), Byb.ravel(), Bzb.ravel(), indexing='ij')
            
        #     loss += jnp.mean((Bx - Bxb_mesh)**2) + jnp.mean((By - Byb_mesh)**2) + jnp.mean((Bz - Bzb_mesh)**2)

        #0 z=0   bottom
        #1 z=2   top                  
        #2 x=0   lateral_1            
        #3 x=2   lateral_2            
        #4 y=0   lateral_3            
        #5 y=2   lateral_4            

        b_bottom, bp_top, bp_lateral_1, bp_lateral_2, bp_lateral_3, bp_lateral_4 = boundary_data
        
        loss = 0.
        Bx, By, Bz = apply_fn(params,  x[0], y[0], z[0])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += jnp.mean((Bx - b_bottom[:, :, 0])**2) + jnp.mean((By - b_bottom[:, :, 1])**2) + jnp.mean((Bz - b_bottom[:, :, 2])**2)

        #0 z=0   bottom
        #1 z=2   top                  -> Only normal(Bz), Bx=0, By=0
        #2 x=0   lateral_1            -> Only tangential(By, Bz), Bx=0
        #3 x=2   lateral_2            -> Only tangential(By, Bz), Bx=0
        #4 y=0   lateral_3            -> Only tangential(Bx, Bz), By=0
        #5 y=2   lateral_4            -> Only tangential(Bx, Bz), By=0

        # Bx, By, Bz = apply_fn(params,  x[1], y[1], z[1])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(Bx**2) + jnp.mean(By**2)

        # Bx, By, Bz = apply_fn(params,  x[2], y[2], z[2])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(Bx**2)

        # Bx, By, Bz = apply_fn(params,  x[3], y[3], z[3])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(Bx**2)
        
        # Bx, By, Bz = apply_fn(params,  x[4], y[4], z[4])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(By**2)

        # Bx, By, Bz = apply_fn(params,  x[5], y[5], z[5])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(By**2)

        Bx, By, Bz = apply_fn(params,  x[1], y[1], z[1])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += jnp.mean((Bx - bp_top[:, :, 0])**2) + jnp.mean((By - bp_top[:, :, 1])**2) + jnp.mean((Bz - bp_top[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[2], y[2], z[2])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += jnp.mean((Bx - bp_lateral_1[:, :, 0])**2) + jnp.mean((By - bp_lateral_1[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_1[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[3], y[3], z[3])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += jnp.mean((Bx - bp_lateral_2[:, :, 0])**2) + jnp.mean((By - bp_lateral_2[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_2[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[4], y[4], z[4])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += jnp.mean((Bx - bp_lateral_3[:, :, 0])**2) + jnp.mean((By - bp_lateral_3[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_3[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[5], y[5], z[5])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += jnp.mean((Bx - bp_lateral_4[:, :, 0])**2) + jnp.mean((By - bp_lateral_4[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_4[:, :, 2])**2)

        
        return loss

    # unpack data
    train_data = train_boundary_data[0]
    boundary_data = train_boundary_data[1]
    xc, yc, zc, xb, yb, zb = train_data

    # isolate loss func from redundant arguments
    loss_pde, loss_ff, loss_div = residual_loss(params, xc, yc, zc, w_ff, w_div)
    loss_bc = boundary_loss(params, xb, yb, zb, *boundary_data)

    loss_fn = lambda params: residual_loss(params, xc, yc, zc, w_ff, w_div)[0] + w_bc*boundary_loss(params, xb, yb, zb, *boundary_data)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient, [loss_ff, loss_div, loss_bc]

# %%
@partial(jax.jit, static_argnums=(0,6))
def apply_model_spinn_random(apply_fn, params, train_boundary_data, w_ff, w_div, w_bc, bc_batch_size):
    def residual_loss(params, x, y, z, w_ff, w_div):
        # calculate u
        Bx, By, Bz = apply_fn(params, x, y, z)
        B = jnp.stack([Bx, By, Bz], axis=-1)
        
        # calculate J
        Jx = curlx(apply_fn, params, x, y, z)
        Jy = curly(apply_fn, params, x, y, z)
        Jz = curlz(apply_fn, params, x, y, z)
        J = jnp.stack([Jx, Jy, Jz], axis=-1)

        JxB = jnp.cross(J, B, axis=-1) 

        #-----------------------------------------------------------
        # loss_ff = jnp.sum(JxB**2, axis=-1)
        # loss_ff = jnp.mean(loss_ff)

        loss_ff = jnp.sum(JxB**2, axis=-1) / (jnp.sum(B**2, axis=-1) + 1e-7)
        loss_ff = jnp.mean(loss_ff)

        # loss_ff = jnp.mean(JxB**2)
        #-----------------------------------------------------------

        # tangent vector dx/dx
        # assumes x, y, z have same shape (very important)
        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)
        vec_z = jnp.ones(z.shape)
        
        Bx_x = jvp(lambda x: apply_fn(params, x, y, z)[0], (x,), (vec_x,))[1]
        # Bx_y = jvp(lambda y: apply_fn(params, x, y, z)[0], (y,), (vec,))[1]
        # Bx_z = jvp(lambda z: apply_fn(params, x, y, z)[0], (z,), (vec,))[1]

        # By_x = jvp(lambda x: apply_fn(params, x, y, z)[1], (x,), (vec,))[1]
        By_y = jvp(lambda y: apply_fn(params, x, y, z)[1], (y,), (vec_y,))[1]
        # By_z = jvp(lambda z: apply_fn(params, x, y, z)[1], (z,), (vec,))[1]

        # Bz_x = jvp(lambda x: apply_fn(params, x, y, z)[2], (x,), (vec,))[1]
        # Bz_y = jvp(lambda y: apply_fn(params, x, y, z)[2], (y,), (vec,))[1]
        Bz_z = jvp(lambda z: apply_fn(params, x, y, z)[2], (z,), (vec_z,))[1]

        divB = Bx_x + By_y + Bz_z
        
        #-----------------------------------------------------------
        # loss_div = jnp.sum((divB)**2, axis=-1)
        # loss_div = jnp.mean(loss_div)

        loss_div = jnp.mean((divB)**2)
        #-----------------------------------------------------------

        loss = w_ff*loss_ff + w_div*loss_div

        return loss, loss_ff, loss_div

    def boundary_loss(params, x, y, z, bc_batch_size, *boundary_data):
        
        # loss = 0.
        # for i in np.arange(4):
        #     boundary_data_batched = boundary_batches[i, :, :, :]
        #     xb = boundary_data_batched[:, 0, :][:, 0].reshape(-1, 1)
        #     yb = boundary_data_batched[:, 0, :][:, 1].reshape(-1, 1)
        #     zb = boundary_data_batched[:, 0, :][:, 2].reshape(-1, 1)

        #     Bx, By, Bz = apply_fn(params, xb, yb, zb)
        #     # Bx, By, Bz = Bx.reshape(-1, 1), By.reshape(-1, 1), Bz.reshape(-1, 1)

        #     Bxb = boundary_data_batched[:, 1, :][:, 0].reshape(-1, 1)
        #     Byb = boundary_data_batched[:, 1, :][:, 1].reshape(-1, 1)
        #     Bzb = boundary_data_batched[:, 1, :][:, 2].reshape(-1, 1)

        #     Bxb_mesh, Byb_mesh, Bzb_mesh = jnp.meshgrid(Bxb.ravel(), Byb.ravel(), Bzb.ravel(), indexing='ij')
            
        #     loss += jnp.mean((Bx - Bxb_mesh)**2) + jnp.mean((By - Byb_mesh)**2) + jnp.mean((Bz - Bzb_mesh)**2)

        #0 z=0   bottom
        #1 z=2   top                  
        #2 x=0   lateral_1            
        #3 x=2   lateral_2            
        #4 y=0   lateral_3            
        #5 y=2   lateral_4            

        # b_bottom, bp_top, bp_lateral_1, bp_lateral_2, bp_lateral_3, bp_lateral_4 = boundary_data

        # loss = 0.

        # Bx, By, Bz = apply_fn(params,  x[0], y[0], z[0])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        
        # bc_bottom_training_data = jnp.vstack([Bx.flatten(), b_bottom[:, :, 0].flatten()])
        # bc_bottom_training_data = jnp.concatenate([bc_bottom_training_data, jnp.vstack([By.flatten(), b_bottom[:, :, 1].flatten()])], axis=1)
        # bc_bottom_training_data = jnp.concatenate([bc_bottom_training_data, jnp.vstack([Bz.flatten(), b_bottom[:, :, 2].flatten()])], axis=1)

        # Bx, By, Bz = apply_fn(params,  x[1], y[1], z[1])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        # bc_training_data = jnp.vstack([Bx.flatten(), bp_top[:, :, 0].flatten()])
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_top[:, :, 1].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_top[:, :, 2].flatten()])], axis=1)


        # Bx, By, Bz = apply_fn(params,  x[2], y[2], z[2])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_1[:, :, 0].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_1[:, :, 1].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_1[:, :, 2].flatten()])], axis=1)


        # Bx, By, Bz = apply_fn(params,  x[3], y[3], z[3])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_2[:, :, 0].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_2[:, :, 1].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_2[:, :, 2].flatten()])], axis=1)


        # Bx, By, Bz = apply_fn(params,  x[4], y[4], z[4])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_3[:, :, 0].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_3[:, :, 1].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_3[:, :, 2].flatten()])], axis=1)


        # Bx, By, Bz = apply_fn(params,  x[5], y[5], z[5])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_4[:, :, 0].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_4[:, :, 1].flatten()])], axis=1)
        # bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_4[:, :, 2].flatten()])], axis=1)

        # bc_latent_batch_size = bc_batch_size // 6
        # bc_bottom_batch_size = bc_batch_size - bc_latent_batch_size

        # M = bc_training_data.shape[-1]
        # random_indices = random.sample(range(M), bc_latent_batch_size)
        # bc_training_data = bc_training_data[:, random_indices]
        # loss += jnp.mean((bc_training_data[0] - bc_training_data[1])**2)

        # M =  bc_bottom_training_data.shape[-1]
        # random_indices = random.sample(range(M), bc_bottom_batch_size)
        # bc_bottom_training_data = bc_bottom_training_data[:, random_indices]
        # loss += jnp.mean((bc_bottom_training_data[0] - bc_bottom_training_data[1])**2)

        b_bottom, bp_top, bp_lateral_1, bp_lateral_2, bp_lateral_3, bp_lateral_4 = boundary_data
        
        loss = 0.
        Bx, By, Bz = apply_fn(params,  x[0], y[0], z[0])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += (1/6)*jnp.mean((Bx - b_bottom[:, :, 0])**2) + jnp.mean((By - b_bottom[:, :, 1])**2) + jnp.mean((Bz - b_bottom[:, :, 2])**2)

        #0 z=0   bottom
        #1 z=2   top                  -> Only normal(Bz), Bx=0, By=0
        #2 x=0   lateral_1            -> Only tangential(By, Bz), Bx=0
        #3 x=2   lateral_2            -> Only tangential(By, Bz), Bx=0
        #4 y=0   lateral_3            -> Only tangential(Bx, Bz), By=0
        #5 y=2   lateral_4            -> Only tangential(Bx, Bz), By=0

        # Bx, By, Bz = apply_fn(params,  x[1], y[1], z[1])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(Bx**2) + jnp.mean(By**2)

        # Bx, By, Bz = apply_fn(params,  x[2], y[2], z[2])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(Bx**2)

        # Bx, By, Bz = apply_fn(params,  x[3], y[3], z[3])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(Bx**2)
        
        # Bx, By, Bz = apply_fn(params,  x[4], y[4], z[4])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(By**2)

        # Bx, By, Bz = apply_fn(params,  x[5], y[5], z[5])
        # Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        # loss += jnp.mean(By**2)

        Bx, By, Bz = apply_fn(params,  x[1], y[1], z[1])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += (1/6)*jnp.mean((Bx - bp_top[:, :, 0])**2) + jnp.mean((By - bp_top[:, :, 1])**2) + jnp.mean((Bz - bp_top[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[2], y[2], z[2])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += (1/6)*jnp.mean((Bx - bp_lateral_1[:, :, 0])**2) + jnp.mean((By - bp_lateral_1[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_1[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[3], y[3], z[3])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += (1/6)*jnp.mean((Bx - bp_lateral_2[:, :, 0])**2) + jnp.mean((By - bp_lateral_2[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_2[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[4], y[4], z[4])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += (1/6)*jnp.mean((Bx - bp_lateral_3[:, :, 0])**2) + jnp.mean((By - bp_lateral_3[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_3[:, :, 2])**2)

        Bx, By, Bz = apply_fn(params,  x[5], y[5], z[5])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        loss += (1/6)*jnp.mean((Bx - bp_lateral_4[:, :, 0])**2) + jnp.mean((By - bp_lateral_4[:, :, 1])**2) + jnp.mean((Bz - bp_lateral_4[:, :, 2])**2)

        return loss

    # unpack data
    train_data = train_boundary_data[0]
    boundary_data = train_boundary_data[1]
    xc, yc, zc, xb, yb, zb = train_data

    # isolate loss func from redundant arguments
    loss_pde, loss_ff, loss_div = residual_loss(params, xc, yc, zc, w_ff, w_div)
    loss_bc = boundary_loss(params, xb, yb, zb, bc_batch_size, *boundary_data)

    loss_fn = lambda params: residual_loss(params, xc, yc, zc, w_ff, w_div)[0] + w_bc*boundary_loss(params, xb, yb, zb, bc_batch_size, *boundary_data)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient, [loss_ff, loss_div, loss_bc]

# %%
class spinn_cube:
    def __init__(self, param_path, parameters_path):
        self.param_path = param_path
        self.parameters_path = parameters_path
    
    def calculate_magnetic_fields(self):
        param_path = self.param_path
        parameters_path = self.parameters_path

        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        features = parameters['features']
        n_layers = parameters['n_layers']
        feat_sizes = tuple([features for _ in range(n_layers)]) 
        r = parameters['r']
        out_dim = parameters['out_dim']
        Nx = parameters['Nx']
        Ny = parameters['Ny']
        Nz = parameters['Nz']
        b_norm = parameters['b_norm']
        pos_enc = parameters['pos_enc']
        mlp = parameters['mlp']
        n_max_x = parameters['n_max_x']
        n_max_y = parameters['n_max_y']
        n_max_z = parameters['n_max_z']

        subkey = jax.random.PRNGKey(0)
        model = SPINN3d(feat_sizes, r, out_dim, pos_enc=pos_enc, mlp=mlp)
        model.init(
                    subkey,
                    jnp.ones((Nx, 1)),
                    jnp.ones((Ny, 1)),
                    jnp.ones((Nz, 1))
                   )
        apply_fn = jax.jit(model.apply)

        with open(param_path, 'rb') as f:
            params = pickle.load(f)

        x = jnp.linspace(0, n_max_x, Nx).reshape(-1, 1)
        y = jnp.linspace(0, n_max_y, Ny).reshape(-1, 1)
        z = jnp.linspace(0, n_max_z, Nz).reshape(-1, 1)
        x, y, z = jax.lax.stop_gradient(x), jax.lax.stop_gradient(y), jax.lax.stop_gradient(z)

        Bx, By, Bz = apply_fn(params, x, y, z)
        B = jnp.stack([Bx, By, Bz], axis=-1)*b_norm
        
        Bx = B[..., 0]
        By = B[..., 1]
        Bz = B[..., 2]

        co_bounds = (0, Nx-1, 0, Ny-1, 0, Nz-1)
        co_coords = create_coordinates(co_bounds).reshape(-1, 3)
        co_coord = co_coords.reshape(Nx, Ny, Nz, 3)
        x = co_coord[..., 0]
        y = co_coord[..., 1]
        z = co_coord[..., 2]
        mesh = pv.StructuredGrid(x, y, z)
        vectors = np.stack([Bx, By, Bz], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
        mesh['B'] = vectors
        mesh.active_vectors_name = 'B'
        magnitude = np.linalg.norm(vectors, axis=-1)
        mesh['mag'] = magnitude
        mesh.active_scalars_name = 'mag'

        self.grid = mesh 
        return self.grid

# %%
import wandb
wandb.login()

# %%
base_path = os.path.expanduser('~/workspace/workspace_mine/_data/NOAA12673/')
# base_path = os.path.expanduser('~/workspace/workspace_mine/_data/NOAA11158/')

result_path = os.path.join(base_path, 'result')
os.makedirs(result_path, exist_ok=True)

b_bottom_path = os.path.join(base_path, 'b_bottom')

b_bottom_list = sorted(glob.glob(os.path.join(b_bottom_path, '*.npy')))
b_bottom_list

# %%
with open(b_bottom_list[0], 'rb') as f:
    b_bottom = np.load(f)

Nx, Ny, _ = b_bottom.shape
Nz = 160

total_iterations = 100000
log_interval = 1000
loss_threshold = 1e-3

series_iterations = 2000
series_log_interval = 3000
series_lr = 5e-5
series_lr_decay_iterations = None

lr_start = 5e-4
lr_end = 5e-5
lr_decay_iterations = 70000
w_bc_decay_iterations = 50000

spatial_norm = 160
n_max_x = (Nx/spatial_norm)
n_max_y = (Ny/spatial_norm)
n_max_z = (Nz/spatial_norm)

potential_boundary_batch_size = 3000
bc_batch_size = 10000

parameters = {
    'features' : 256,
    'n_layers' : 8,
    'r' : 256,
    'out_dim' : 3,
    'Nx' : Nx,
    'Ny' : Ny,
    'Nz' : Nz,
    'b_norm' : 2500,
    'pos_enc' : 0,
    'mlp' : 'mlp',
    'activation': 'sin',
    'lr': lr_start,
    'series_lr': series_lr,
    'series_lr_decay_iterations': series_lr_decay_iterations,
    'lr_decay_iterations': [lr_decay_iterations],
    'n_max_x': n_max_x,
    'n_max_y': n_max_y,
    'n_max_z': n_max_z,
    'is_random': True,
    'NcxNcyNcz': [None,None,None],
    'Nc': 32,
    'random_interval': 1,
    'w_ff': 0.01,
    'w_div': 0.01,
    'w_ff_w_div':[0.01, 0.01],
    'w_bc': 1000,
    'w_bc_decay_iterations': [w_bc_decay_iterations],
    'bc_batch_size': [bc_batch_size],
    'choice': False,
    'decay_rate': (lr_end / lr_start) ** (1 / lr_decay_iterations),
    'total_iterations': total_iterations,
    'log_interval': log_interval,
    'series_iterations': series_iterations,
    'series_log_interval': series_log_interval,
    'loss_threshold': loss_threshold,
    'potential_boundary_batch_size': potential_boundary_batch_size,
    'clip_val': 0.1,
}

# %%
wandb.init(
    project="test2",
    config = parameters
)

# %%
class SPINN_series_Tra:
    def __init__(self, result_path, wdb):
        os.makedirs(result_path, exist_ok=True)

        self.parameters = wdb.config
        parameters_path = os.path.join(result_path, "parameters.pickle")
        with open(parameters_path, "wb") as f:
            pa = dict(self.parameters)
            pickle.dump(pa, f)

        # logger = logging.getLogger()
        # logger.setLevel(logging.INFO)
        # for hdlr in logger.handlers[:]:
        #     logger.removeHandler(hdlr)
        # logger.addHandler(logging.FileHandler("{0}/{1}.log".format(result_path, "info_log"))) 
        # logger.addHandler(logging.StreamHandler()) 
        # logger.info(self.parameters)
        
        self.result_path = result_path
        self.wandb = wdb
        # self.logger = logger
        
    def train(self, b_bottom_list):

        parameters = self.parameters

        features = parameters['features']
        n_layers = parameters['n_layers']
        feat_sizes = tuple([features for _ in range(n_layers)]) 
        r = parameters['r']
        out_dim = parameters['out_dim']
        pos_enc = parameters['pos_enc']
        mlp = parameters['mlp']
        activation = parameters['activation']

        model = SPINN3d(feat_sizes, r, out_dim, pos_enc=pos_enc, mlp=mlp, activation=activation)

        seed = 111
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key, 2)

        Nx = parameters['Nx']
        Ny = parameters['Ny']
        Nz = parameters['Nz']
        params = model.init(
                        subkey,
                        jnp.ones((Nx, 1)),
                        jnp.ones((Ny, 1)),
                        jnp.ones((Nz, 1))
                    )

        apply_fn = jax.jit(model.apply)
        self.apply_fn = apply_fn

        for i, b_bottom_path in enumerate(b_bottom_list):

            b_bottom_date = os.path.basename(b_bottom_path)[9:-4]
            output_path = os.path.join(self.result_path, b_bottom_date)
            os.makedirs(output_path, exist_ok=True)

            final_params_path = os.path.join(output_path, f"final_params.pickle")

            if os.path.exists(final_params_path):
                with open(final_params_path, 'rb') as f:
                    params = pickle.load(f)
                continue

            if i == 0:
                lr = parameters['lr']
                lr_decay_iterations = parameters['lr_decay_iterations'][0]
                decay_rate = parameters['decay_rate']

                w_ff, w_div = parameters['w_ff_w_div']
                w_bc = parameters['w_bc']
                w_bc_decay_iterations = parameters['w_bc_decay_iterations'][0]
                w_bc_decay = (1 / w_bc) ** (1 / w_bc_decay_iterations) if w_bc_decay_iterations is not None else 1

                total_iterations = parameters['total_iterations']
                log_iterations = parameters['log_interval']
            else:
                lr = parameters['series_lr']
                lr_decay_iterations = None
                decay_rate = None

                w_ff, w_div = parameters['w_ff_w_div']
                w_bc = 1
                w_bc_decay_iterations = None
                w_bc_decay = 1

                total_iterations = parameters['series_iterations']
                log_iterations = parameters['series_log_interval']

            clip_val = parameters['clip_val']

            if lr_decay_iterations is not None: 
                schedule = optax.exponential_decay(
                            init_value=lr, 
                            transition_steps=lr_decay_iterations,
                            decay_rate=decay_rate,
                            )

                optim = optax.chain(
                    optax.clip(clip_val),
                    optax.adam(learning_rate=schedule)
                )
            else:
                optim = optax.chain(
                    optax.clip(clip_val),
                    optax.adam(learning_rate=lr)
                )

            state = optim.init(params)
        
            b_bottom_date = os.path.basename(b_bottom_path)[9:-4]
            output_path  = os.path.join(self.result_path, b_bottom_date)
            os.makedirs(output_path, exist_ok=True)

            BC_path = os.path.join(output_path, 'BC.pickle')
            with open(b_bottom_path, 'rb') as f:
                b_bottom = np.load(f)

            if not os.path.exists(BC_path):
                b_norm = parameters['b_norm']
                potential_boundary_batch_size = parameters['potential_boundary_batch_size']
                
                cal_and_save_potential_boundary_for_spinn(b_bottom, Nz, b_norm, BC_path, potential_boundary_batch_size) 
                
                import torch
                torch.cuda.empty_cache()

            with open(BC_path, 'rb') as f:
                boundary_data = pickle.load(f)

            n_max_x = parameters['n_max_x']
            n_max_y = parameters['n_max_y']
            n_max_z = parameters['n_max_z']
            is_random = parameters['is_random']

            if is_random is True:
                Nc = parameters['Nc']
                Ncx, Ncy, Ncz = parameters['NcxNcyNcz']
                choice = parameters['choice']
                train_data = generate_train_data_random(subkey, Nx, Ny, Nz, n_max_x, n_max_y, n_max_z, Nc, Ncx, Ncy, Ncz, choice)
            else:
                train_data = generate_train_data(Nx, Ny, Nz, n_max_x, n_max_y, n_max_z)

            train_boundary_data = [train_data, boundary_data]

            b_potential_me_path = os.path.join(Path(result_path).parent, 'b_potential_energy')
            b_potential_me_path = os.path.join(b_potential_me_path, f'b_me_{b_bottom_date}.npy') 
            if os.path.exists(b_potential_me_path):
                with open(b_potential_me_path, "rb") as f:
                    pot_me = np.load(f)
            else:
                pot_me = None
            
            bc_batch_size = parameters['bc_batch_size'][0]

            for e in trange(1, total_iterations+1):
                if w_bc > 1:
                    w_bc *= w_bc_decay
                    if w_bc <= 1:
                        w_bc = 1

                if is_random is True:
                    # if e % random_interval == 0:
                    key, subkey = jax.random.split(key, 2)
                    train_data = generate_train_data_random(subkey, Nx, Ny, Nz, n_max_x, n_max_y, n_max_z, Nc, Ncx, Ncy, Ncz, choice)
                    train_boundary_data = [train_data, boundary_data]

                if bc_batch_size is not None:
                    loss, gradient, losses = apply_model_spinn_random(apply_fn, params, train_boundary_data, w_ff, w_div, w_bc, bc_batch_size)
                else:
                    loss, gradient, losses = apply_model_spinn(apply_fn, params, train_boundary_data, w_ff, w_div, w_bc)

                params, state = update_model(optim, gradient, params, state)
                
                loss_ff, loss_div, loss_bc = losses
                self.wandb.log({"train": {"loss":loss.item(), "loss_ff":loss_ff.item(), "loss_div":loss_div.item(), "loss_bc":loss_bc.item(), "w_ff":w_ff, "w_div":w_div, "w_bc":w_bc}})
                if e % log_iterations == 0:
                    self.validation(params, b_bottom, pot_me)
                    # logger.info(f'Epoch: {e}/{total_iterations} --> total loss: {loss:.8f}, div {div:.8f}, ff {ff:.8f}, b_diff {b_diff:.8f}, ratio {ratio:.4f}, total_free_energy {total_free_energy:.4f}, val_loss {val_loss:.8f}')
            
            with open(final_params_path, "wb") as f:
                pickle.dump(params, f)
            
            self.energ_plot(params, pot_me)

    def validation(self, params, b_bottom, pot_me=None):

        parameters = self.parameters
        n_max_x = parameters['n_max_x']
        n_max_y = parameters['n_max_y']
        n_max_z = parameters['n_max_z']
        Nx = parameters['Nx']
        Ny = parameters['Ny']
        Nz = parameters['Nz']
        b_norm = parameters['b_norm']

        x = jnp.linspace(0, n_max_x, Nx).reshape(-1, 1)
        y = jnp.linspace(0, n_max_y, Ny).reshape(-1, 1)
        z = jnp.linspace(0, n_max_z, Nz).reshape(-1, 1)
        x, y, z = jax.lax.stop_gradient(x), jax.lax.stop_gradient(y), jax.lax.stop_gradient(z)

        Bx, By, Bz = self.apply_fn(params, x, y, z)
        B = np.stack([Bx, By, Bz], axis=-1)*b_norm
        J = curl(B)

        div = (divergence(B)**2).mean()

        JxB = np.cross(J, B)
        ff = (np.sum(JxB**2, axis=-1))**(0.5) / (np.sum(B**2, axis=-1) + 1e-7)**(0.5)
        ff = np.mean(ff)

        b_bottom_unnormal = b_bottom * b_norm
        b_diff = np.abs(B[:, :, 0, :] - b_bottom_unnormal)
        b_diff = np.mean((np.sum((b_diff)**2, -1))**(0.5))

        val_loss = div+ff+b_diff

        self.wandb.log({"val": {"div":div, "ff":ff, "b_diff":b_diff, "val_loss":val_loss}})

        jmaps = magnitude(J).sum(2)
        fig, ax = plt.subplots()
        im = ax.imshow(jmaps.T, origin='lower')
        fig.colorbar(im)
        self.wandb.log({"plot": {"J_map":fig}})
        plt.close()

        fig, ax = plt.subplots()
        im = ax.imshow(B[:, :, 0, 2].T, origin='lower', cmap='gray')
        fig.colorbar(im)
        self.wandb.log({"plot": {"Bz":fig}})
        plt.close()

        if pot_me is not None:
            stride = 2
            Mm_per_pixel = stride*0.36
            cm_per_pixel = Mm_per_pixel*1e8
            dV = cm_per_pixel ** 3
    
            me = magnetic_energy(B) * dV
            free_me = me - pot_me 

            total_energy = me.sum()
            total_potential_energy = pot_me.sum()
            ratio = total_energy/total_potential_energy
            total_free_energy = free_me.sum()/1e32

            self.wandb.log({"val": {"ratio_E_and_E_pot":ratio, "free_energy(1e32)":total_free_energy}})

            free_energy_map = free_me.sum(2)
            fig, ax = plt.subplots()
            im = ax.imshow(free_energy_map.T, origin='lower', cmap='jet')
            fig.colorbar(im)
            self.wandb.log({"plot": {"Free_energy_map":fig}})
            plt.close()

        return

    def energ_plot(self, params, pot_me=None):

        parameters = self.parameters
        n_max_x = parameters['n_max_x']
        n_max_y = parameters['n_max_y']
        n_max_z = parameters['n_max_z']
        Nx = parameters['Nx']
        Ny = parameters['Ny']
        Nz = parameters['Nz']
        b_norm = parameters['b_norm']

        x = jnp.linspace(0, n_max_x, Nx).reshape(-1, 1)
        y = jnp.linspace(0, n_max_y, Ny).reshape(-1, 1)
        z = jnp.linspace(0, n_max_z, Nz).reshape(-1, 1)
        x, y, z = jax.lax.stop_gradient(x), jax.lax.stop_gradient(y), jax.lax.stop_gradient(z)

        Bx, By, Bz = self.apply_fn(params, x, y, z)
        B = np.stack([Bx, By, Bz], axis=-1)*b_norm

        if pot_me is not None:
            stride = 2
            Mm_per_pixel = stride*0.36
            cm_per_pixel = Mm_per_pixel*1e8
            dV = cm_per_pixel ** 3
    
            me = magnetic_energy(B) * dV
            free_me = me - pot_me 

            total_energy = me.sum()/1e32
            total_potential_energy = pot_me.sum()/1e32
            ratio = total_energy/total_potential_energy
            total_free_energy = free_me.sum()/1e32

            self.wandb.log({"result": {"total_energy(1e32)":total_energy, "ratio_E_and_E_pot":ratio, "free_energy(1e32)":total_free_energy}})
        return

# %%
trainer = SPINN_series_Tra(result_path, wandb)
trainer.train(b_bottom_list)
wandb.finish()

# %%



