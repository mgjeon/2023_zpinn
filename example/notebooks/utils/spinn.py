import jax 
import jax.numpy as jnp
from jax import jvp
import optax
from flax import linen as nn 

from typing import Sequence
from functools import partial


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


class SPINN3d(nn.Module):
    features: Sequence[int]
    r: int
    out_dim: int

    @nn.compact
    def __call__(self, x, y, z):
        inputs, outputs, xy, pred = [x, y, z], [], [], []
        init = nn.initializers.glorot_normal()
        for X in inputs:
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.r*self.out_dim, kernel_init=init)(X)
            outputs += [jnp.transpose(X, (1, 0))]

        for i in range(self.out_dim):
            xy += [jnp.einsum('fx, fy->fxy', outputs[0][self.r*i:self.r*(i+1)], outputs[1][self.r*i:self.r*(i+1)])]
            pred += [jnp.einsum('fxy, fz->xyz', xy[i], outputs[-1][self.r*i:self.r*(i+1)])]

        return pred


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def generate_train_data(nc, nx, ny, nz, key):
    
    keys = jax.random.split(key, 3)

    xc = jax.random.uniform(keys[0], (nc, 1), minval=-1., maxval=1)
    yc = jax.random.uniform(keys[1], (nc, 1), minval=-1., maxval=1)
    zc = jax.random.uniform(keys[2], (nc, 1), minval=-1., maxval=1)

    # boundary points
    xb = [jnp.linspace(-1, 1, nx).reshape(-1, 1)]
    yb = [jnp.linspace(-1, 1, ny).reshape(-1, 1)]
    zb = [jnp.array([[-1.]])]

    return xc, yc, zc, xb, yb, zb


@partial(jax.jit, static_argnums=(0,))
def apply_model_spinn(apply_fn, params, xc, yc, zc, xb, yb, zb, b_bottom):
    def residual_loss(params, x, y, z):
        # calculate loss_ff
        Bx, By, Bz = apply_fn(params, x, y, z)
        B = jnp.stack([Bx, By, Bz], axis=-1)
        
        Jx = curlx(apply_fn, params, x, y, z)
        Jy = curly(apply_fn, params, x, y, z)
        Jz = curlz(apply_fn, params, x, y, z)
        J = jnp.stack([Jx, Jy, Jz], axis=-1)

        JxB = jnp.cross(J, B, axis=-1) 

        loss_ff = jnp.sum(JxB**2, axis=-1) / (jnp.sum(B**2, axis=-1) + 1e-7)
        loss_ff = jnp.mean(loss_ff)
        
        # calculate loss_div
        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)
        vec_z = jnp.ones(z.shape)
        
        Bx_x = jvp(lambda x: apply_fn(params, x, y, z)[0], (x,), (vec_x,))[1]
        By_y = jvp(lambda y: apply_fn(params, x, y, z)[1], (y,), (vec_y,))[1]
        Bz_z = jvp(lambda z: apply_fn(params, x, y, z)[2], (z,), (vec_z,))[1]

        divB = Bx_x + By_y + Bz_z

        loss_div = jnp.mean((divB)**2)

        # calculate loss
        loss = loss_ff + loss_div

        return loss

    def boundary_loss(params, x, y, z, b_bottom):
        
        loss = 0.
        bx, by, bz = apply_fn(params,  x[0], y[0], z[0])
        bx, by, bz = jnp.squeeze(bx), jnp.squeeze(by), jnp.squeeze(bz)
        loss += jnp.mean((bx - b_bottom[:, :, 0])**2) + jnp.mean((by - b_bottom[:, :, 1])**2) + jnp.mean((bz - b_bottom[:, :, 2])**2)
        
        return loss

    loss_fn = lambda params: residual_loss(params, xc, yc, zc) + boundary_loss(params, xb, yb, zb, b_bottom)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient
