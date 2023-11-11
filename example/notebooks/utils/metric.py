import numpy as np


def divergence(b_field):  # (x, y, z, (xyz-field))
    div_B = np.stack([np.gradient(b_field[..., i], axis=i, edge_order=2) for i in range(3)], axis=-1).sum(-1)
    return div_B


def curl(b_field):  # (x, y, z)
    _, dFx_dy, dFx_dz = np.gradient(b_field[..., 0], axis=[0, 1, 2], edge_order=2)
    dFy_dx, _, dFy_dz = np.gradient(b_field[..., 1], axis=[0, 1, 2], edge_order=2)
    dFz_dx, dFz_dy, _ = np.gradient(b_field[..., 2], axis=[0, 1, 2], edge_order=2)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    return np.stack([rot_x, rot_y, rot_z], -1)


def lorentz_force(b_field, j_field=None):
    j_field = j_field if j_field is not None else curl(b_field)
    l = np.cross(j_field, b_field, axis=-1)
    return l


def vector_norm(vector):
    return np.sqrt((vector ** 2).sum(-1))


def angle(b_field, j_field):
    norm = vector_norm(b_field) * vector_norm(j_field) + 1e-7
    j_cross_b = np.cross(j_field, b_field, axis=-1)
    sig = vector_norm(j_cross_b) / norm
    return np.arcsin(np.clip(sig, -1. + 1e-7, 1. - 1e-7)) * (180 / np.pi)


def normalized_divergence(b_field):
    return np.abs(divergence(b_field)) / (vector_norm(b_field) + 1e-7)

def normalized_divergence_2(b):
    # return np.abs(divergence(b)) / (vector_norm(b) + 1e-7)
    div_numerator = np.abs(divergence(b))
    div_denominator = vector_norm(b)
    div = np.divide(div_numerator, div_denominator, where=div_denominator!=0)
    return div

def weighted_theta(b, j=None):
    j = j if j is not None else curl(b)
    sigma = vector_norm(lorentz_force(b, j)) / vector_norm(b) / vector_norm(j)
    w_sigma = np.average((sigma), weights=vector_norm(j))
    theta_j = np.arcsin(w_sigma) * (180 / np.pi)
    return theta_j

def weighted_theta_2(b, j=None):
    j = j if j is not None else curl(b)
    # sigma = vector_norm(lorentz_force(b, j)) / vector_norm(b) / vector_norm(j)
    numerator = vector_norm(lorentz_force(b, j))
    denominator = vector_norm(b)*vector_norm(j)
    sigma = np.divide(numerator, denominator,  where=denominator!=0)
    w_sigma = np.average((sigma), weights=vector_norm(j))
    theta_j = np.arcsin(w_sigma) * (180 / np.pi)
    return theta_j


def energy(b):
    return (b ** 2).sum(-1) / (8 * np.pi)


def vector_norm_2(vector):
    return np.linalg.norm(vector, axis=-1)


def dot_product(a, b):
    return (a*b).sum(-1)

def relative_l2(u, u_gt):
    return np.linalg.norm(u-u_gt) / np.linalg.norm(u_gt)

def mse(u, u_gt):
    return np.mean((u-u_gt)**2)

def evaluate(b, B):
    # b : model solution
    # B : reference magnetic field
    eps = 1e-7

    result = {}

    result['c_vec'] = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
    result['c_vec_2'] = np.sum(dot_product(B, b)) / np.sqrt(np.sum(vector_norm(B)**2) * np.sum(vector_norm(b)**2))

    M = np.prod(B.shape[:-1])

    result['c_cs'] = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))
    nu = dot_product(B, b)
    de = vector_norm(B) * vector_norm(b)
    result['c_cs_2'] = (1 / M) * np.sum(np.divide(nu, de, where=de!=0.))
    result['c_cs_2_ep'] = (1 / M) * np.sum(nu/(de + eps))

    result['E_n'] = vector_norm(b - B).sum() / vector_norm(B).sum()
    result['E_n_2'] = np.sum(vector_norm(b - B)) / np.sum(vector_norm(B))

    result['E_m'] = 1 / M * (vector_norm(b - B) / vector_norm(B)).sum()
    nu = vector_norm(b - B)
    de = vector_norm(B)
    result['E_m_2'] = (1 / M) * np.sum(np.divide(nu, de, where=de!=0.))
    result['E_m_ep'] = (1 / M) * np.sum((nu/(de + eps)))

    result['eps'] = (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

    j = curl(b)
    result['sig_J'] = (vector_norm(np.cross(j, b, -1)) / vector_norm(b)).sum() / (vector_norm(j).sum() + 1e-6) * 1e2
    J = curl(B)
    result['sig_J_ll'] = (vector_norm(np.cross(J, B, -1)) / vector_norm(B)).sum() / (vector_norm(J).sum() + 1e-6) * 1e2

    result['L1'] = (vector_norm(np.cross(j, b, -1)) ** 2 / vector_norm(b) ** 2).mean()
    result['L2'] = (divergence(b) ** 2).mean()

    result['L1_B'] = (vector_norm(np.cross(curl(B), B, -1)) ** 2 / vector_norm(B) ** 2).mean()
    result['L2_B'] = (divergence(B) ** 2).mean()

    result['L2n'] = (np.abs(divergence(b)) / (vector_norm(b) + 1e-8)).mean() * 1e2
    result['L2n_B'] = (np.abs(divergence(B)) / (vector_norm(B) + 1e-8)).mean() * 1e2

    result['relative_l2'] = relative_l2(b, B)

    result['mse'] = mse(b, B)

    return result