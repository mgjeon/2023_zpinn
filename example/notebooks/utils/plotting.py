import numpy as np
import matplotlib.pyplot as plt 


def plot_overview(b, B, z=0, b_norm=2500, ret=False):
    fig, axs = plt.subplots(2, 3, figsize=(12, 4))

    ax = axs[0]
    ax[0].imshow(b[..., z, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[0].set_title('bx')
    ax[1].imshow(b[..., z, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[1].set_title('by')
    ax[2].imshow(b[..., z, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[2].set_title('bz')

    ax = axs[1]
    ax[0].imshow(B[..., z, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[0].set_title('Bx')
    ax[1].imshow(B[..., z, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[1].set_title('By')
    ax[2].imshow(B[..., z, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[2].set_title('Bz')

    fig.suptitle(f'z={z}')
    
    plt.tight_layout()

    if ret:
        plt.close()
        return fig
    else:
        plt.show()


def plot_s(mag, title, n_samples):
    fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
    heights = np.linspace(0, 1, n_samples) ** 2 * (mag.shape[2] - 1)  # more samples from lower heights
    heights = heights.astype(np.int32)
    for i in range(3):
        for j, h in enumerate(heights):
            v_min_max = np.max(np.abs(mag[:, :, h, i]))
            axs[i, j].imshow(mag[:, :, h, i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                            origin='lower')
            axs[i, j].set_axis_off()
    for j, h in enumerate(heights):
        axs[0, j].set_title('%.01f' % h, fontsize=20)
    fig.tight_layout()
    fig.suptitle(title, fontsize=25)
    plt.tight_layout()
    plt.show()


def plot_sample(b, B, n_samples=10):
    plot_s(b, 'b', n_samples)
    plot_s(B, 'B', n_samples)