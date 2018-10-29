import os
import numpy as np
from keras.datasets import cifar10
import _gpp
from tqdm import tqdm
from gpp.vae import ConvVAE
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems


np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Hyperparameters for ConvVAE
z_size = 32
batch_size = 32
learning_rate = 0.0001
kl_tolerance = 0.5

# Parameters for training
NUM_EPOCH = 500
DATA_DIR = "record"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

(x_train, _), (x_test, _) = cifar10.load_data()

# split into batches:
dataset = x_train
total_length = len(dataset)
num_batches = int(np.floor(total_length / batch_size))
print("num_batches", num_batches)

# reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=False)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
train_step = train_loss = r_loss = kl_loss = None
for epoch in range(NUM_EPOCH):
    np.random.shuffle(dataset)
    epoch > 0 and print("step", (train_step + 1), train_loss, r_loss, kl_loss)
    for idx in tqdm(range(num_batches)):
        batch = dataset[idx * batch_size:(idx + 1) * batch_size]

        obs = batch.astype(np.float) / 255.0

        feed = {vae.x: obs, }

        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
            vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
        ], feed)
        # if (train_step + 1) % 5000 == 0:
        #     vae.save_json("tf_vae/vae.json")

# finished, final model:
# vae.save_json("tf_vae/vae.json")

batch_z = vae.encode(x_test[:batch_size])
reconstruct = vae.decode(batch_z)
for i in range(batch_size):
    plt.subplot(1, 2, 1)
    original = x_test[i]
    plt.imshow(original)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstruct[i])
    plt.axis("off")
    plt.savefig("fig{}.png".format(i))
