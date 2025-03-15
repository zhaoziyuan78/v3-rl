"""
Visualising animations
"""

import math
import numpy as np
import h5py
from torchvision import transforms


def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))
    if len(shape) == 2:
        order = "C"
    else:
        order = "F"

    def cell(i, j):
        ind = i * n_cols + j
        if i * n_cols + j < array.shape[0]:
            return array[ind].reshape(*shape, order="C")
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


def plot_gif(x_seq, shape, path, filename):
    n_cols = int(np.sqrt(x_seq.shape[0]))
    x_seq = x_seq[: n_cols**2]
    T = x_seq.shape[0]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    x0 = x_seq[0]
    im = plt.imshow(x0.transpose(1, 2, 0), animated=True)
    plt.axis("off")

    def update(t):
        x_frame = x_seq[t].transpose(1, 2, 0)
        im.set_array(x_frame)
        return (im,)

    anim = FuncAnimation(fig, update, frames=np.arange(T), interval=250, blit=True)
    anim.save(path + filename + ".gif", writer="imagemagick")
    print("image saved as " + path + filename + ".gif")


def plot_film(x_seq, shape, path, filename):
    n_cols = int(np.sqrt(x_seq.shape[0]))
    x_seq = x_seq[: n_cols**2]
    T = x_seq.shape[0]
    import matplotlib

    # plot like frames cut from a movie

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    whole_img = np.concatenate(x_seq, axis=2)
    whole_img_tensor = transforms.ToTensor()(whole_img.transpose(1, 2, 0))
    whole_img_tensor = transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.2, hue=0.2
    )(whole_img_tensor)
    whole_img = whole_img_tensor.numpy()
    # whole_img = np.clip(whole_img, 0, 1)  # for float, 0, 1 is enough for plotting

    plt.imshow(whole_img.transpose(1, 2, 0))
    plt.axis("off")
    plt.savefig(path + filename + ".png")
    print("image saved as " + path + filename + ".png")


def plot_gif_from_hdf5(hdf5_path, n=10):
    hdf5 = h5py.File(hdf5_path, "r")
    keys = list(hdf5.keys())
    selected_keys = np.random.choice(keys, n)
    for key in selected_keys:
        x_seq = hdf5[key]["seq"][:]
        plot_gif(x_seq, (64, 64), "./", key)


def plot_film_from_hdf5(hdf5_path, n=10):
    hdf5 = h5py.File(hdf5_path, "r")
    keys = list(hdf5.keys())
    selected_keys = np.random.choice(keys, n)
    for key in selected_keys:
        x_seq = hdf5[key]["seq"][:]
        plot_film(x_seq, (64, 64), "./", key)


if __name__ == "__main__":
    # plot_gif_from_hdf5("../../../data/SpritesAction_hdf5/train.hdf5", n=10)
    plot_film_from_hdf5("../../../data/SpritesAction_hdf5/train.hdf5", n=10)
