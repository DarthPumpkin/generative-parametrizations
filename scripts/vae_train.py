import os
import numpy as np
from keras.datasets import cifar10, mnist
import _gpp
import datetime
import cv2
from tqdm import tqdm
# from gpp.vae import ConvVAE
# from gpp.vae_modified_architecture import ConvVAE
from gpp.world_models_vae import ConvVAE
import zipfile
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems


np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Hyperparameters for ConvVAE
z_size = 64
batch_size = 32
learning_rate = 0.001
kl_tolerance = 0.5

# Parameters for training
NUM_EPOCH = 5000
DATA_DIR = "record"
IMG_OUTPUT_DIR = './out'

model_save_path = "tf_vae"
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

pendulum_data = False
data_path = "../data"

if pendulum_data:
    dataset = np.load(os.path.join(data_path, 'pendulum_imgs.npy'))
    dataset = dataset[:, 60:190, 60:190]
else:  # fetch_sphere env
    dataset = np.load(os.path.join(data_path, 'fetch_sphere_imgs.npz'))
    dataset = dataset['arr_0']

np.random.shuffle(dataset)
dataset = dataset / 255.
new_data = []
for i, d in enumerate(dataset):
    new_data.append(cv2.resize(d, (64, 64), interpolation=cv2.INTER_AREA))
dataset = np.array(new_data)
train_ratio = int(0.8 * len(dataset))
x_train = dataset[:train_ratio]
x_test = dataset[train_ratio:]

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
              )
# vae.load_json("tf_vae/vae-fetch.json")

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
train_step = train_loss = r_loss = kl_loss = None

train_loss_list = []
r_loss_list = []
kl_loss_list = []
loss_grads_list = []

smoothing = 0.9

for epoch in range(NUM_EPOCH):
    np.random.shuffle(x_train)
    for idx in tqdm(range(num_batches)):
        batch = x_train[idx * batch_size:(idx + 1) * batch_size]

        obs = batch.astype(np.float)
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
        if epoch > 0:
            loss_grads_list.append(train_loss_list[-1] - train_loss_list[-2])
        else:
            loss_grads_list.append(0)

    epoch_train_loss = np.mean(train_loss_list[-num_batches:])
    epoch_r_loss = np.mean(r_loss_list[-num_batches:])
    epoch_kl_loss = np.mean(kl_loss_list[-num_batches:])
    epoch_gradient_loss = np.mean(loss_grads_list[-num_batches:])

    epoch > 0 and print(" Epoch: ", epoch,
                        " step: ", (train_step + 1),
                        " train loss: ", epoch_train_loss,
                        " r loss: ", epoch_r_loss,
                        " kl loss: ", epoch_kl_loss,
                        " derivative: ", loss_grads_list[-1])
    # finished, final model:
    if epoch % 10 == 0:
        vae.save_json("tf_vae/vae-fetch{}.json".format(epoch))
        plt.plot(train_loss_list, label="total loss")
        plt.plot(r_loss_list, label="rec loss")
        plt.plot(kl_loss_list, label="kl loss")
        plt.legend()
        plt.savefig(f'{IMG_OUTPUT_DIR}/train_loss_history.pdf', format="pdf")

        plt.close("all")
        plt.plot(loss_grads_list)
        plt.savefig(f'{IMG_OUTPUT_DIR}/train_loss_gradient.pdf', format="pdf")

        batch_z = vae.encode(x_test[:batch_size])
        reconstruct = vae.decode(batch_z)
        reconstruct = (reconstruct * 255).astype(np.uint8)
        im2print = 10
        # if epoch > 200 and epoch % 50 == 0:
        #     orig = batch_z
        #     values = np.linspace(-4, 4, 50)
        #     p_count = 1
        #     for i in range(batch_z.shape[1]):
        #         batch_z = orig.copy()
        #         for j, n in enumerate(values):
        #             batch_z[0][i] = n
        #             reconstruct = (vae.decode(batch_z) * 255).astype(np.uint8)
        #             plt.subplot(16, 50, p_count)
        #             plt.imshow(reconstruct[0])
        #             plt.axis("off")
        #             p_count += 1
        #     plt.show()

        for i in range(im2print):
            plt.subplot(im2print, 2, 1+2*i)
            original = x_test[i].clip(0, 1)
            plt.imshow(original)
            plt.axis("off")

            plt.subplot(im2print, 2, 2+2*i)
            plt.imshow(reconstruct[i])
            plt.axis("off")

        plt.savefig(f'{IMG_OUTPUT_DIR}/epoch_{epoch}_fig_{i}.png')
        plt.close("all")
