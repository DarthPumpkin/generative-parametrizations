import os
import numpy as np
import cv2
import _gpp
from gpp.world_models_vae import ConvVAE
import matplotlib.pyplot as plt

""" INIT DATA """
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
z_size = 32
batch_size = 32
kl_tolerance = 0.5
learning_rate = 0.001
data_path = "../data"
dataset = np.load(os.path.join(data_path, 'fetch_sphere_big_imgs.npz'))
dataset = dataset['arr_0']
np.random.shuffle(dataset)
dataset = dataset / 255.
new_data = []
for i, d in enumerate(dataset):
    new_data.append(cv2.resize(d, (64, 64), interpolation=cv2.INTER_AREA))
dataset = np.array(new_data)
train_ratio = int(0.8 * len(dataset))
# x_test = dataset[train_ratio:]
x_test = dataset
total_length = len(x_test)
num_batches = int(np.floor(total_length / batch_size))

""" LOAD MODEL """
vae = ConvVAE(z_size=z_size, batch_size=batch_size, learning_rate=learning_rate, kl_tolerance=kl_tolerance,
              is_training=True, reuse=False, gpu_mode=False, reconstruction_option=1,
              kl_option=2)
# vae.load_json("best_models/kl2-rl1-b10vae-fetch950.json")
vae.load_json("best_models/kl1-rl1-b100vae-fetch950.json")
print(num_batches)
all_z = []
for i in range(num_batches):
    batch_z = vae.encode(x_test[i * batch_size: (i + 1) * batch_size])
    all_z.extend(batch_z)
all_z = np.array(all_z)
variances = np.var(all_z, axis=0)
print(variances)
