import os
import numpy as np
from keras.datasets import cifar10, mnist
import _gpp
from tqdm import tqdm
from gpp.vae import ConvVAE
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems


np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Hyperparameters for ConvVAE
z_size = 32
batch_size = 128
learning_rate = 0.01
kl_tolerance = 0.05

# Parameters for training
NUM_EPOCH = 100
DATA_DIR = "record"
IMG_OUTPUT_DIR = './out'

model_save_path = "tf_vae"
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

mnist_data = False

if mnist_data:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    test_horses = np.where(y_test == 7)[0]
    x_test = x_test[test_horses]
    horses = np.where(y_train == 7)[0]
    x_train = x_train[horses]


total_length = len(x_train)
num_batches = int(np.floor(total_length / batch_size))
print("num_batches", num_batches)

# reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=False,
              mnist=mnist_data)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
train_step = train_loss = r_loss = kl_loss = None

train_loss_list = []
r_loss_list = []
kl_loss_list = []

smoothing = 0.9

for epoch in range(NUM_EPOCH):
    np.random.shuffle(x_train)
    for idx in tqdm(range(num_batches)):
        batch = x_train[idx * batch_size:(idx + 1) * batch_size]

        obs = batch.astype(np.float)
        # obs = (batch.astype(np.float) - 127.5) / 127.5
        feed = {vae.x: obs, }

        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([vae.loss,
                                                                     vae.r_loss,
                                                                     vae.kl_loss,
                                                                     vae.global_step,
                                                                     vae.train_op],
                                                                    feed)

        if epoch == 0 and idx == 0:
            train_loss_list.append(train_loss)
            r_loss_list.append(r_loss)
            kl_loss_list.append(kl_loss)

        train_loss_list.append(train_loss_list[-1]*smoothing+train_loss*(1-smoothing))
        r_loss_list.append(r_loss_list[-1] * smoothing + r_loss * (1-smoothing))
        kl_loss_list.append(kl_loss_list[-1] * smoothing + kl_loss*(1-smoothing))
        # if (train_step + 1) % 5000 == 0:
        #     vae.save_json("tf_vae/vae.json")

    epoch_train_loss = np.mean(train_loss_list[-num_batches:])
    epoch_r_loss = np.mean(r_loss_list[-num_batches:])
    epoch_kl_loss = np.mean(kl_loss_list[-num_batches:])

    epoch > 0 and print(" Epoch: ", epoch,
                        " step: ", (train_step + 1),
                        " train loss: ", epoch_train_loss,
                        " r loss: ", epoch_r_loss,
                        " kl loss: ", epoch_kl_loss)
    # finished, final model:
    # vae.save_json("tf_vae/vae.json")

    plt.plot(train_loss_list, label="total loss")
    plt.plot(r_loss_list, label="rec loss")
    plt.plot(kl_loss_list, label="kl loss")
    plt.legend()

    plt.savefig(f'{IMG_OUTPUT_DIR}/train_loss_history.pdf', format="pdf")

    batch_z = vae.encode(x_test[:batch_size])
    reconstruct = vae.decode(batch_z)
    reconstruct = (reconstruct * 255).astype(np.uint16)
    # reconstruct = (reconstruct * 127.5 + 127.5).astype(np.uint8)
    im2print = 10

    for i in range(im2print):
        plt.subplot(im2print, 2, 1+2*i)
        original = x_test[i]
        plt.imshow(original)
        plt.axis("off")

        plt.subplot(im2print, 2, 2+2*i)
        plt.imshow(reconstruct[i])
        plt.axis("off")

    plt.savefig(f'{IMG_OUTPUT_DIR}/epoch_{epoch}_fig_{i}.png')
    plt.close("all")
