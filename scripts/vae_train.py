import os
import numpy as np
import cv2
from tqdm import tqdm

import _gpp

from gpp.world_models_vae import ConvVAE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['savefig.dpi'] = 500
plt.style.use('ggplot')
plt.rcParams["errorbar.capsize"] = 2


class TrainSetting:
    def __init__(self, kl_opt, reconstruct_opt, b_in, z_size, dataset_name="push_sphere_v0"):
        self.kl_opt = kl_opt,
        self.reconstruct_opt = reconstruct_opt
        self.b = b_in
        self.z_size = z_size
        self.name = "kl{}rl{}-z{}-b{}-{}".format(kl_opt, reconstruct_opt, z_size, b_in, dataset_name)

all_settings = []
"""TRAIN KNOWN BAD MODELS"""
all_settings.append(TrainSetting(0, 0, 0, 32))
all_settings.append(TrainSetting(0, 1, 0, 32))
all_settings.append(TrainSetting(0, 2, 0, 32))

all_settings.append(TrainSetting(1, 0, 100, 32))
all_settings.append(TrainSetting(1, 1, 100, 32))
all_settings.append(TrainSetting(1, 2, 100, 32))

all_settings.append(TrainSetting(2, 0, 100, 32))
all_settings.append(TrainSetting(2, 1, 100, 32))
all_settings.append(TrainSetting(2, 2, 100, 32))
"""TRAIN GOOD MODELS"""
all_settings.append(TrainSetting(2, 1, 10,  32))
all_settings.append(TrainSetting(2, 1, 50,  32))
all_settings.append(TrainSetting(2, 1, 100, 32))
all_settings.append(TrainSetting(2, 1, 150, 32))
all_settings.append(TrainSetting(2, 1, 200, 32))
all_settings.append(TrainSetting(2, 1, 250, 32))
all_settings.append(TrainSetting(2, 1, 300, 32))
all_settings.append(TrainSetting(2, 1, 400, 32))
all_settings.append(TrainSetting(2, 1, 500, 32))
"""TRAIN SUPER MODELS"""
all_settings.append(TrainSetting(2, 1, 150, 16))
all_settings.append(TrainSetting(2, 1, 200, 16))
all_settings.append(TrainSetting(2, 1, 250, 16))
all_settings.append(TrainSetting(2, 1, 300, 16))
all_settings.append(TrainSetting(2, 1, 400, 16))
# all_settings.append(TrainSetting(2, 1, 500, 16))
# all_settings.append(TrainSetting(2, 1, 600, 16))

all_settings = all_settings[::-1]

# all_settings.append(TrainSetting(0, 0, 0, "kl0-rl0-b0"))
# all_settings.append(TrainSetting(0, 1, 0, "kl0-rl1-b0"))
# all_settings.append(TrainSetting(0, 2, 0, "kl0-rl2-b0"))
#
# all_settings.append(TrainSetting(1, 0, 10, "kl1-rl0-b10"))
# all_settings.append(TrainSetting(1, 1, 10, "kl1-rl1-b10"))
# all_settings.append(TrainSetting(1, 2, 10, "kl1-rl2-b10"))
#
# all_settings.append(TrainSetting(1, 0, 50, "kl1-rl0-b50"))
# all_settings.append(TrainSetting(1, 1, 50, "kl1-rl1-b50"))
# all_settings.append(TrainSetting(1, 2, 50, "kl1-rl2-b50"))
#
# all_settings.append(TrainSetting(1, 0, 100, "kl1-rl0-b100"))
# all_settings.append(TrainSetting(1, 1, 100, "kl1-rl1-b100"))
# all_settings.append(TrainSetting(1, 2, 100, "kl1-rl2-b100"))
#
# all_settings.append(TrainSetting(2, 0, 10, "kl2-rl0-b10"))
# all_settings.append(TrainSetting(2, 1, 10, "kl2-rl1-b10"))
# all_settings.append(TrainSetting(2, 2, 10, "kl2-rl2-b10"))
#
# all_settings.append(TrainSetting(2, 0, 50, "kl2-rl0-b50"))
# all_settings.append(TrainSetting(2, 1, 50, "kl2-rl1-b50"))
# all_settings.append(TrainSetting(2, 2, 50, "kl2-rl2-b50"))
#
# all_settings.append(TrainSetting(2, 0, 100, "kl2-rl0-b100"))
# all_settings.append(TrainSetting(2, 1, 100, "kl2-rl1-b100"))
# all_settings.append(TrainSetting(2, 2, 100, "kl2-rl2-b100"))

# all_settings.append(TrainSetting(2, 1, 100, 16, "kl2rl1-b100-z16-colordataset"))
# all_settings.append(TrainSetting(2, 1, 150, 16, "kl2rl1-b150-z16-colordataset"))
# all_settings.append(TrainSetting(2, 1, 200, 16, "kl2rl1-b200-z16-colordataset"))
#
# all_settings.append(TrainSetting(2, 1, 100, 32, "kl2rl1-b100-z32-colordataset"))
# all_settings.append(TrainSetting(2, 1, 150, 32, "kl2rl1-b150-z32-colordataset"))
# all_settings.append(TrainSetting(2, 1, 200, 32, "kl2rl1-b200-z32-colordataset"))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems


np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Hyperparameters for ConvVAE
# z_size = 16
batch_size = 32
learning_rate = 0.001
kl_tolerance = 0.5
# Parameters for training
NUM_EPOCH = 30
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
    dataset = np.load(os.path.join(data_path, 'push_sphere_v0_imgs.npz'))
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
for setting in all_settings:

    final_model_path = "tf_vae/{}vae-fetch{}.json".format(setting.name, NUM_EPOCH-1)
    if os.path.exists(final_model_path):
        print("Model for setting {} exists. Skipping...".format(setting.name))
        continue

    print("Start setting: {}".format(setting.name))
    vae = ConvVAE(z_size=setting.z_size, batch_size=batch_size, learning_rate=learning_rate, kl_tolerance=kl_tolerance,
                  is_training=True, reuse=False, gpu_mode=False, reconstruction_option=setting.reconstruct_opt,
                  kl_option=setting.kl_opt)
    # vae.load_json("tf_vae/kl2-rl1-b100-colordataasetvae-fetch200.json")
    # train loop:
    train_step = train_loss = r_loss = kl_loss = None

    train_loss_list = r_loss_list = kl_loss_list = loss_grads_list = []
    smoothing = 0.9
    init_disentanglement = disentanglement = setting.b  # B
    max_capacity = 25
    capacity_change_duration = num_batches * 100  # arbitrary: = 100 epochs of disentanglement
    c = 0

    for epoch in tqdm(range(NUM_EPOCH), desc='Epoch'):
        np.random.shuffle(x_train)
        for idx in range(num_batches):
            batch = x_train[idx * batch_size:(idx + 1) * batch_size]
            step = epoch * num_batches + idx
            if setting.kl_opt == 1:  # OPTION 1: moving c (increase capacity, from paper)
                c = max_capacity if step > capacity_change_duration else max_capacity * (step / capacity_change_duration)
            if setting.kl_opt == 2:  # OPTION 2: dynamic beta (my experiment)
                disentanglement = max(1, init_disentanglement * (1 - (step / capacity_change_duration)))

            feed = {vae.x: batch.astype(np.float), vae.beta: disentanglement, vae.capacity: c}
            (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([vae.loss, vae.r_loss, vae.kl_loss, vae.global_step,
                                                                         vae.train_op], feed)
            if epoch == idx == 0:
                train_loss_list.append(train_loss)
                r_loss_list.append(r_loss)
                kl_loss_list.append(kl_loss)

            train_loss_list.append(train_loss_list[-1]*smoothing+train_loss*(1-smoothing))
            r_loss_list.append(r_loss_list[-1] * smoothing + r_loss * (1-smoothing))
            kl_loss_list.append(kl_loss_list[-1] * smoothing + kl_loss*(1-smoothing))
            if epoch > 0:
                loss_grads_list.append(loss_grads_list[-1]*smoothing +
                                       (train_loss_list[-1] - train_loss_list[-2])*(1 - smoothing))
            else:
                loss_grads_list.append(0)

        epoch_train_loss = np.mean(train_loss_list[-num_batches:])
        epoch_r_loss = np.mean(r_loss_list[-num_batches:])
        epoch_kl_loss = np.mean(kl_loss_list[-num_batches:])
        epoch_gradient_loss = np.mean(loss_grads_list[-num_batches:])

        # epoch == -1 and print(" Epoch: ", epoch,
        #                     " step: ", (train_step + 1),
        #                     " train loss: ", epoch_train_loss,
        #                     " r loss: ", epoch_r_loss,
        #                     " kl loss: ", epoch_kl_loss,
        #                     " derivative: ", loss_grads_list[-1],
        #                     " last beta: ", disentanglement)
        # finished, final model:
        if epoch == NUM_EPOCH-1:  # or epoch % 50 == 0:
            vae.save_json("tf_vae/{}vae-fetch{}.json".format(setting.name, epoch))
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
            im2print = 20

            plt.figure(figsize=(5 * 2, 5 * im2print))
            plt.suptitle(setting.name)
            for i in range(im2print):
                plt.subplot(im2print, 2, 1+2*i)
                original = x_test[i].clip(0, 1)
                plt.imshow(original)
                plt.axis("off")

                plt.subplot(im2print, 2, 2+2*i)
                plt.imshow(reconstruct[i])
                plt.axis("off")

            plt.savefig(f'{IMG_OUTPUT_DIR}/{setting.name}epoch_{epoch}_fig_.png')
            plt.close("all")
