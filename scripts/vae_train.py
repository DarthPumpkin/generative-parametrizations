import os
import numpy as np
from keras.datasets import cifar10
import _gpp
from gpp.vae import ConvVAE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems


np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Hyperparameters for ConvVAE
z_size = 32
batch_size = 32
learning_rate = 0.0001
kl_tolerance = 0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

(_, _), (x_test, _) = cifar10.load_data()

# split into batches:
dataset = x_test
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
for epoch in range(NUM_EPOCH):
    np.random.shuffle(dataset)
    for idx in range(num_batches):
        batch = dataset[idx * batch_size:(idx + 1) * batch_size]

        obs = batch.astype(np.float) / 255.0

        feed = {vae.x: obs, }

        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
            vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
        ], feed)

        if (train_step + 1) % 500 == 0:
            print("step", (train_step + 1), train_loss, r_loss, kl_loss)
        if (train_step + 1) % 5000 == 0:
            vae.save_json("tf_vae/vae.json")

# finished, final model:
vae.save_json("tf_vae/vae.json")
