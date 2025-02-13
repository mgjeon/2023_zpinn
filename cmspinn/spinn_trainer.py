# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/33_SPINN_trainer.ipynb.

# %% auto 0
__all__ = ['hvp_fwdfwd', 'update_model', 'SPINN3d', 'generate_train_data', 'generate_train_data_random', 'curlx', 'curly',
           'curlz', 'apply_model_spinn', 'apply_model_spinn_random', 'SPINN_Trainer', 'spinn_cube']

# %% ../nbs/33_SPINN_trainer.ipynb 2
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

# %% ../nbs/33_SPINN_trainer.ipynb 3
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

# %% ../nbs/33_SPINN_trainer.ipynb 4
class SPINN3d(nn.Module):
    features: Sequence[int]
    r: int
    out_dim: int
    pos_enc: int
    mlp: str

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
                    # X = nn.activation.tanh(X)
                    X = jnp.sin(X)
                X = nn.Dense(self.r*self.out_dim, kernel_init=init)(X)
                outputs += [jnp.transpose(X, (1, 0))]

        elif self.mlp == 'modified_mlp':
            for X in inputs:
                U = jnp.sin(nn.Dense(self.features[0], kernel_init=init)(X))
                V = jnp.sin(nn.Dense(self.features[0], kernel_init=init)(X))
                H = jnp.sin(nn.Dense(self.features[0], kernel_init=init)(X))
                for fs in self.features[:-1]:
                    Z = nn.Dense(fs, kernel_init=init)(H)
                    # Z = nn.activation.tanh(Z)
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

# %% ../nbs/33_SPINN_trainer.ipynb 5
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

# %% ../nbs/33_SPINN_trainer.ipynb 6
@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
def generate_train_data_random(key, nx, ny, nz, n_max_x, n_max_y, n_max_z, nc, ncx=None, ncy=None, ncz=None, choice=False):
    
    keys = jax.random.split(key, 4)

    if (ncx is not None) and (ncy is not None) and (ncz is not None):
      xc = jax.random.uniform(keys[1], (ncx, 1), minval=0., maxval=n_max_x)
      yc = jax.random.uniform(keys[2], (ncy, 1), minval=0., maxval=n_max_y)
      zc = jax.random.uniform(keys[3], (ncz, 1), minval=0., maxval=n_max_z)
    
    elif choice is False:
      xc = jax.random.uniform(keys[1], (nc, 1), minval=0., maxval=n_max_x)
      yc = jax.random.uniform(keys[2], (nc, 1), minval=0., maxval=n_max_y)
      zc = jax.random.uniform(keys[3], (nc, 1), minval=0., maxval=n_max_z)
    else:
      xc = jnp.linspace(0, n_max_x, nx).reshape(-1, 1)
      yc = jnp.linspace(0, n_max_y, ny).reshape(-1, 1)
      zc = jnp.linspace(0, n_max_z, nz).reshape(-1, 1)
      xc = jax.random.choice(keys[1], xc, shape=(nc,))
      yc = jax.random.choice(keys[2], yc, shape=(nc,))
      zc = jax.random.choice(keys[3], zc, shape=(nc,))

    # boundary points
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

# %% ../nbs/33_SPINN_trainer.ipynb 7
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

# %% ../nbs/33_SPINN_trainer.ipynb 8
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

        return loss

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
    loss_fn = lambda params: residual_loss(params, xc, yc, zc, w_ff, w_div) + w_bc*boundary_loss(params, xb, yb, zb, *boundary_data)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient

# %% ../nbs/33_SPINN_trainer.ipynb 9
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

        return loss

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

        b_bottom, bp_top, bp_lateral_1, bp_lateral_2, bp_lateral_3, bp_lateral_4 = boundary_data

        loss = 0.

        Bx, By, Bz = apply_fn(params,  x[0], y[0], z[0])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)
        
        bc_bottom_training_data = jnp.vstack([Bx.flatten(), b_bottom[:, :, 0].flatten()])
        bc_bottom_training_data = jnp.concatenate([bc_bottom_training_data, jnp.vstack([By.flatten(), b_bottom[:, :, 1].flatten()])], axis=1)
        bc_bottom_training_data = jnp.concatenate([bc_bottom_training_data, jnp.vstack([Bz.flatten(), b_bottom[:, :, 2].flatten()])], axis=1)


        Bx, By, Bz = apply_fn(params,  x[1], y[1], z[1])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        bc_training_data = jnp.vstack([Bx.flatten(), bp_top[:, :, 0].flatten()])
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_top[:, :, 1].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_top[:, :, 2].flatten()])], axis=1)


        Bx, By, Bz = apply_fn(params,  x[2], y[2], z[2])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_1[:, :, 0].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_1[:, :, 1].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_1[:, :, 2].flatten()])], axis=1)


        Bx, By, Bz = apply_fn(params,  x[3], y[3], z[3])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_2[:, :, 0].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_2[:, :, 1].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_2[:, :, 2].flatten()])], axis=1)


        Bx, By, Bz = apply_fn(params,  x[4], y[4], z[4])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_3[:, :, 0].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_3[:, :, 1].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_3[:, :, 2].flatten()])], axis=1)


        Bx, By, Bz = apply_fn(params,  x[5], y[5], z[5])
        Bx, By, Bz = jnp.squeeze(Bx), jnp.squeeze(By), jnp.squeeze(Bz)

        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bx.flatten(), bp_lateral_4[:, :, 0].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([By.flatten(), bp_lateral_4[:, :, 1].flatten()])], axis=1)
        bc_training_data = jnp.concatenate([bc_training_data, jnp.vstack([Bz.flatten(), bp_lateral_4[:, :, 2].flatten()])], axis=1)

        bc_latent_batch_size = bc_batch_size // 6
        bc_bottom_batch_size = bc_batch_size - bc_latent_batch_size

        M = bc_training_data.shape[-1]
        random_indices = random.sample(range(M), bc_latent_batch_size)
        bc_training_data = bc_training_data[:, random_indices]
        loss += jnp.mean((bc_training_data[0] - bc_training_data[1])**2)

        M =  bc_bottom_training_data.shape[-1]
        random_indices = random.sample(range(M), bc_bottom_batch_size)
        bc_bottom_training_data = bc_bottom_training_data[:, random_indices]
        loss += jnp.mean((bc_bottom_training_data[0] - bc_bottom_training_data[1])**2)

        return loss

    # unpack data
    train_data = train_boundary_data[0]
    boundary_data = train_boundary_data[1]
    xc, yc, zc, xb, yb, zb = train_data

    # isolate loss func from redundant arguments
    loss_fn = lambda params: residual_loss(params, xc, yc, zc, w_ff, w_div) + w_bc*boundary_loss(params, xb, yb, zb, bc_batch_size, *boundary_data)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient

# %% ../nbs/33_SPINN_trainer.ipynb 10
class SPINN_Trainer:
    def __init__(self, output_path, BC_path, b_bottom, Nz, b_norm, transfer_learning_path=None, parameters=None):
        os.makedirs(output_path, exist_ok=True)

        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)
        logger.addHandler(logging.FileHandler("{0}/{1}.log".format(output_path, "info_log")))  # set the new file handler
        logger.addHandler(logging.StreamHandler())  # set the new console handler
        
        self.logger = logger

        Nx, Ny, _ = b_bottom.shape

        if parameters is None:
            features = 512
            n_layers = 8
            feat_sizes = tuple([features for _ in range(n_layers)]) 
            r = 512 
            out_dim = 3 

            lr = 5e-4

            pos_enc = 0
            mlp = 'modified_mlp'

            n_max_x = 2*(Nx/Nx)
            n_max_y = 2*(Ny/Nx)
            n_max_z = 2*(Nz/Nx)

            parameters = {'feat_sizes' : feat_sizes, 
                'r' : r, 
                'out_dim' : out_dim, 
                'Nx' : Nx, 
                'Ny' : Ny, 
                'Nz' : Nz, 
                'b_norm' : b_norm,
                'pos_enc' : pos_enc,
                'mlp' : mlp,
                'lr': lr,
                'n_max_x': n_max_x,
                'n_max_y': n_max_y,
                'n_max_z': n_max_z,}
        else:
            feat_sizes = parameters['feat_sizes']
            r = parameters['r']
            out_dim = parameters['out_dim']
            Nx = parameters['Nx']
            Ny = parameters['Ny']
            Nz = parameters['Nz']
            b_norm = parameters['b_norm']
            pos_enc = parameters['pos_enc']
            mlp = parameters['mlp']
            lr = parameters['lr']
            lr_decay_iterations = parameters['lr_decay_iterations']
            n_max_x = parameters['n_max_x']
            n_max_y = parameters['n_max_y']
            n_max_z = parameters['n_max_z']
            is_random = parameters['is_random']
            Nc = parameters['Nc']
            decay_rate = parameters['decay_rate']
            series_lr = parameters['series_lr']

        logger.info(parameters)
        
        parameters_path = os.path.join(output_path, "parameters.pickle")
        with open(parameters_path, "wb") as f:
            pickle.dump(parameters, f)

        seed = 111
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key, 2)

        model = SPINN3d(feat_sizes, r, out_dim, pos_enc=pos_enc, mlp=mlp)
        if transfer_learning_path is None:
            params = model.init(
                        subkey,
                        jnp.ones((Nx, 1)),
                        jnp.ones((Ny, 1)),
                        jnp.ones((Nz, 1))
                    )
            apply_fn = jax.jit(model.apply)

            if lr_decay_iterations is not None: 
                optim = optax.adam(learning_rate=optax.exponential_decay(init_value=lr, transition_steps=lr_decay_iterations,
                                                                         decay_rate=decay_rate))
            else:
                optim = optax.adam(learning_rate=lr)

            state = optim.init(params)
        else:
            model.init(
                        subkey,
                        jnp.ones((Nx, 1)),
                        jnp.ones((Ny, 1)),
                        jnp.ones((Nz, 1))
                    )
            apply_fn = jax.jit(model.apply)
            with open(transfer_learning_path, 'rb') as f:
                params = pickle.load(f)

            if lr_decay_iterations is not None: 
                optim = optax.adam(learning_rate=optax.exponential_decay(init_value=series_lr, transition_steps=lr_decay_iterations,
                                                                         decay_rate=decay_rate))
            else:
                optim = optax.adam(learning_rate=lr)
                
            state = optim.init(params)

        with open(BC_path, 'rb') as f:
            boundary_data = pickle.load(f)
        
        if is_random is True:
            Ncx = parameters['Ncx']
            Ncy = parameters['Ncy']
            Ncz = parameters['Ncz']
            choice = parameters['choice']
            train_data = generate_train_data_random(subkey, Nx, Ny, Nz, n_max_x, n_max_y, n_max_z, Nc, Ncx, Ncy, Ncz, choice)
        else:
            train_data = generate_train_data(Nx, Ny, Nz, n_max_x, n_max_y, n_max_z)

        train_boundary_data = [train_data, boundary_data]

        self.apply_fn = apply_fn
        self.params = params
        self.train_boundary_data = train_boundary_data
        self.optim = optim
        self.state = state
        self.output_path = output_path

        self.logger = logger

        self.parameters = parameters
        self.boundary_data = boundary_data

        self.key = key

    def train(self, total_iterations, log_iterations, loss_threshold=0.001):
        params = self.params
        state = self.state

        logger = self.logger

        parameters = self.parameters 
        is_random = parameters['is_random']
        Nc = parameters['Nc']
        key = self.key 

        w_ff = parameters['w_ff']
        w_div = parameters['w_div']
        w_bc = parameters['w_bc']
        w_bc_decay_iterations = parameters['w_bc_decay_iterations']
        w_bc_decay = (1 / w_bc) ** (1 / w_bc_decay_iterations) if w_bc_decay_iterations is not None else 1

        bc_batch_size = parameters['bc_batch_size']

        if is_random is True:
            random_interval = parameters['random_interval']
            Nx = parameters['Nx']
            Ny = parameters['Ny']
            Nz = parameters['Nz']
            n_max_x = parameters['n_max_x']
            n_max_y = parameters['n_max_y']
            n_max_z = parameters['n_max_z']
            Ncx = parameters['Ncx']
            Ncy = parameters['Ncy']
            Ncz = parameters['Ncz']
            key, subkey = jax.random.split(key, 2)
            choice = parameters['choice']

        losses = []
        if logger is None:
            print('Complie Start')
        else: 
            logger.info('Complie Start')
        start = time.time()

        if bc_batch_size is not None:
            loss, gradient = apply_model_spinn_random(self.apply_fn, params, self.train_boundary_data, w_ff, w_div, w_bc, bc_batch_size)
        else:
            loss, gradient = apply_model_spinn(self.apply_fn, params, self.train_boundary_data, w_ff, w_div, w_bc)

        losses.append(loss.item())
        params, state = update_model(self.optim, gradient, params, state)

        if w_bc > 1:
            w_bc *= w_bc_decay
            if w_bc <= 1:
                w_bc = 1

        if is_random is True:
            train_data = generate_train_data_random(subkey, Nx, Ny, Nz, n_max_x, n_max_y, n_max_z, Nc, Ncx, Ncy, Ncz, choice)
            self.train_boundary_data = [train_data, self.boundary_data]
        
        if bc_batch_size is not None:
            loss, gradient = apply_model_spinn_random(self.apply_fn, params, self.train_boundary_data, w_ff, w_div, w_bc, bc_batch_size)
        else:
            loss, gradient = apply_model_spinn(self.apply_fn, params, self.train_boundary_data, w_ff, w_div, w_bc)
        losses.append(loss.item())
        params, state = update_model(self.optim, gradient, params, state)
        runtime = time.time() - start
        if logger is None:
            print(f'Complie End --> total: {runtime:.2f}sec')
        else:
            logger.info(f'Complie End --> total: {runtime:.2f}sec')

        
        start = time.time()
        for e in trange(1, total_iterations + 1):

            if w_bc > 1:
                w_bc *= w_bc_decay
                if w_bc <= 1:
                   w_bc = 1

            if is_random is True:
                if e % random_interval == 0:
                    # sample new input data
                    key, subkey = jax.random.split(key, 2)
                    train_data = generate_train_data_random(subkey, Nx, Ny, Nz, n_max_x, n_max_y, n_max_z, Nc, Ncx, Ncy, Ncz, choice)
                    self.train_boundary_data = [train_data, self.boundary_data]

            if bc_batch_size is not None:
                loss, gradient = apply_model_spinn_random(self.apply_fn, params, self.train_boundary_data, w_ff, w_div, w_bc, bc_batch_size)
            else:
                loss, gradient = apply_model_spinn(self.apply_fn, params, self.train_boundary_data, w_ff, w_div, w_bc)
            losses.append(loss.item())
            if loss.item() < loss_threshold:
                if logger is None:
                    print(f'Epoch: {e}/{total_iterations} --> loss: {loss:.8f} < {loss_threshold}')
                else:
                    logger.info(f'Epoch: {e}/{total_iterations} --> loss: {loss:.8f} < {loss_threshold}')
                break
            
            params, state = update_model(self.optim, gradient, params, state)
            
            if e % log_iterations == 0:
                if logger is None:
                    print(f'Epoch: {e}/{total_iterations} --> total loss: {loss:.8f}')
                else:
                    logger.info(f'Epoch: {e}/{total_iterations} --> total loss: {loss:.8f}')
                params_path = os.path.join(self.output_path, f"params_{e}.pickle")
                with open(params_path, "wb") as f:
                    pickle.dump(params, f)

        final_params_path = os.path.join(self.output_path, f"final_params.pickle")
        with open(final_params_path, "wb") as f:
            pickle.dump(params, f)

        np.save(os.path.join(self.output_path, 'losses.npy'), losses)

        runtime = time.time() - start
        if logger is None:
            print(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(total_iterations-1)*1000):.2f}ms/iter.)')
        else:
            logger.info(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(total_iterations-1)*1000):.2f}ms/iter.)')

# %% ../nbs/33_SPINN_trainer.ipynb 11
import pyvista as pv
import pickle
from .mag_viz import create_coordinates

# %% ../nbs/33_SPINN_trainer.ipynb 12
class spinn_cube:
    def __init__(self, param_path, parameters_path):
        self.param_path = param_path
        self.parameters_path = parameters_path
    
    def calculate_magnetic_fields(self):
        param_path = self.param_path
        parameters_path = self.parameters_path

        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        feat_sizes = parameters['feat_sizes']
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
