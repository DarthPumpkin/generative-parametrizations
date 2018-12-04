import os

from tqdm import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt

import _gpp
from gpp.world_models_vae import ConvVAE


def main():

    """ INIT DATA """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems
    z_size = 16
    batch_size = 32
    data_path = "../data"
    dataset = np.load(os.path.join(data_path, 'push_sphere_v0_imgs.npz'))
    dataset = dataset['arr_0'] / 255.

    """ LOAD MODEL """
    vae = ConvVAE(z_size=z_size, batch_size=batch_size, is_training=False, reuse=False, gpu_mode=False)
    vae.load_json("tf_vae/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json")

    all_original_images = []
    all_predicted_images = []

    for i in tqdm(range(2, 12)):

        batch_images = dataset[i*batch_size: (i+1)*batch_size]
        pred_images = vae.decode(vae.encode(batch_images))

        all_original_images.append(batch_images)
        all_predicted_images.append(pred_images)

    all_original_images = np.concatenate(all_original_images)
    all_predicted_images = np.concatenate(all_predicted_images)

    n_images, rows, cols, chs = all_original_images.shape
    zipped = np.zeros((n_images, rows, cols*2, chs))

    zipped[:, :, :cols] = all_original_images
    zipped[:, :, cols:] = all_predicted_images
    zipped = (zipped * 255.).astype(np.uint8)

    gif_path = './out/vae_test.mov'
    imageio.mimwrite(gif_path, zipped, quality=10)


if __name__ == '__main__':
    main()
