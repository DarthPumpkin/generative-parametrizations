"""
    1 - CREATE DATA USING TRAINED VAE  👍
    2 - SPLIT DATA INTO TRAIN / TEST SEQUENCES. START WITH 7 - 1 👍
    3 -  ?? NORMALIZE DATA ?? 👍 Might use different method
    4 - CREATE LSTM MODEL
    5 - TRAIN MODEL
    6 - EVALUATE RESULTS BY DECODING PREDICTED LATENT SPACES
"""
import os
from gpp.world_models_vae import ConvVAE
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import Sequential, Model
from keras.layers import LSTM, Dense

data_path = "../data"
batch_size = 32
z_size = 16


def create_encoded_dataset():
    def resize_image(data, shape=(64, 64)):
        new_data = []
        for d in data:
            if len(new_data) == batch_size:
                yield np.array(new_data) / 255
                new_data = []
            new_data.append(cv2.resize(d, shape, interpolation=cv2.INTER_AREA))
    vae = ConvVAE(z_size=z_size, is_training=False, reuse=False, gpu_mode=False)
    vae.load_json("../scripts/tf_vae/kl2-rl1-b50_longervae-fetch199.json")

    dataset = np.load(os.path.join(data_path, 'fetch_sphere_big_longer_imgs.npz'))
    all_z = np.concatenate([vae.encode(dat) for dat in resize_image(dataset['arr_0'])]).reshape((199, 32, 16))
    np.save(os.path.join(data_path, "encoded_dataset"), all_z)


def split_train_test(steps=7):
    dataset = np.load(os.path.join(data_path, "encoded_dataset.npy"))
    x_train = []
    y_train = []
    for d in dataset:
        x_train.append(d[:steps])
        y_train.append(d[steps + 1])
    avg = np.average(dataset)
    std = np.std(dataset)

    return (np.array(x_train) - avg) / std, (np.array(y_train) - avg) / std, avg, std


def plot_previous_images(imgs, pred_idx):
    im2print = pred_idx + 1
    for i in range(im2print):
        plt.subplot(1, im2print, i+1)
        original = imgs[i].clip(0, 1)
        plt.imshow(original)
        plt.axis("off")
    plt.show()


def build_model():
    steps = 7
    vae = ConvVAE(z_size=z_size, is_training=False, reuse=False, gpu_mode=False)
    vae.load_json("../scripts/tf_vae/kl2-rl1-b50_longervae-fetch199.json")
    x, y, avg, std = split_train_test(steps)
    model = Sequential()
    model.add(LSTM(256, input_shape=(x.shape[-2], x.shape[-1]), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(16, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()

    feed = np.zeros((32, 16,))
    for i in range(32):
        model.fit(x, y, batch_size=32, epochs=25, shuffle=False)
        original_images = np.zeros((32, 16))
        original_images[:steps] = x[i]
        original_images = vae.decode(original_images)

        feed[0] = model.predict(x)[i] * std + avg
        predict_img = vae.decode(feed)
        original_images[steps] = predict_img[0]
        plot_previous_images(original_images, steps)


build_model()


