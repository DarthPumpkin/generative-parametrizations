import os
from gpp.world_models_vae import ConvVAE
import numpy as np
from keras import Sequential, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import save_model, load_model
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
np.set_printoptions(suppress=True)


class LSTMModel:
    def __init__(self, vae_path=None, z_size=16, steps=4, training=True,
                 l2l_model_path=None, l2reward_model_path=None, dataset_path=None,
                 dataset_detail_path=None, reward_key='reward'):
        self.z_size = z_size
        self.steps = steps
        self.vae_path = vae_path
        self.l2l_model_path = l2l_model_path
        self.l2reward_model_path = l2reward_model_path
        self.reward_key = reward_key
        self.dataset_path = dataset_path
        self.dataset_detail_path = dataset_detail_path
        self.x = self.y_latent = self.latent_predictions = self.rewards = None
        self.vae = self.__load_vae()
        self.reward_avg = 0
        self.reward_std = 0
        if training:
            self.create_sliding_dataset()
        try:
            self.l2l_model = self.__load_model(l2l_model_path)
            self.l2reward_model = self.__load_model(l2reward_model_path)
        except OSError:
            pass

    def __load_vae(self):
        vae = ConvVAE(z_size=self.z_size, is_training=False, reuse=False, gpu_mode=False)
        # "../scripts/best_models/carlomodel/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json"
        vae.load_json(self.vae_path)
        return vae

    def __load_model(self, model_path) -> Model:
        return load_model(model_path)

    def __load_latent_predictions(self):
        self.latent_predictions = self.predict_l2l(self.x)

    def predict_l2l(self, inp):
        return self.l2l_model.predict(inp)

    def predict_l2reward(self, inp):
        return self.l2reward_model.predict(inp)

    def create_sliding_dataset(self):
        dataset = np.load(self.dataset_path)['arr_0']
        df = pd.read_pickle(self.dataset_detail_path)
        actions = []
        rewards = []
        for i in range(df['episode'].max() + 1):
            ep_df = df[df['episode'] == i]
            action = np.array(ep_df['raw_action'].tolist())
            reward = np.array(ep_df[self.reward_key].tolist())
            rewards.append(reward)
            actions.append(action)
        x = []
        y_latent = []
        y_rewards = []
        for i, d in enumerate(dataset):
            start_idx = 0
            for j in range(self.steps, d.shape[0]-1):
                state_action = np.concatenate([d[start_idx: j], actions[i][start_idx: j]], axis=1)
                x.append(state_action)
                y_latent.append(d[j])
                y_rewards.append(rewards[i][j])
                start_idx += 1
        x = np.array(x)
        y_latent = np.array(y_latent)
        y_rewards = np.array(y_rewards)
        self.x = (x - np.average(x)) / np.std(x)
        self.y_latent = y_latent
        self.rewards = y_rewards

    def build_l2reward_model(self):
        self.__load_latent_predictions()
        samples = int(0.9 * len(self.latent_predictions))
        self.latent_predictions, self.rewards = self.unison_shuffleds(self.latent_predictions, self.rewards)
        x_train = self.latent_predictions[:samples]
        y_train = self.rewards[:samples]
        x_test = self.latent_predictions[samples:]
        y_test = self.rewards[samples:]
        callbacks = [ReduceLROnPlateau(factor=0.7, patience=6, verbose=1), EarlyStopping(patience=10)]

        model = Sequential()
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        model.fit(x_train, y_train, batch_size=32, epochs=250, validation_split=0.1, verbose=2, callbacks=callbacks)
        rewards = model.predict(x_test)
        diff = 0
        for i, (actual, pred) in enumerate(zip(rewards, y_test)):
            act = float(actual[0])
            if i % 20 == 0:
                print(f'{np.round(act, 2)} -- {round(pred, 2)}')
            diff += abs(act - pred)
        print(diff / len(rewards))
        save_model(model, self.l2reward_model_path)

    def build_l2l_model(self):
        callbacks = [ReduceLROnPlateau(factor=0.9, patience=6, verbose=1), EarlyStopping(patience=10)]
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.x.shape[-2], self.x.shape[-1]), return_sequences=True))
        # model.add(Dropout(0.1))
        model.add(LSTM(64))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.y_latent.shape[-1], activation='linear'))
        model.compile(loss='mse', optimizer='nadam')
        model.fit(self.x, self.y_latent, batch_size=32, epochs=100,
                  validation_split=0.1, verbose=2, callbacks=callbacks)
        model.save(self.l2l_model_path)

    @staticmethod
    def unison_shuffled_copies(a, b, c):
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p]

    @staticmethod
    def unison_shuffleds(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # @staticmethod
    # def _plot_previous_images(imgs, pred_idx, predicted_img, target_img):
    #     im2print = pred_idx + 2  # +2 for both target image and predicted imgage
    #     for i in range(im2print - 1):
    #         plt.subplot(1, im2print, i + 1)
    #         original = imgs[i].clip(0, 1)
    #         plt.title("IMG {}".format(i), fontsize=10)
    #         plt.imshow(original)
    #         plt.axis("off")
    #     plt.subplot(1, im2print, im2print - 1)
    #     plt.title("Y", fontsize=10)
    #     plt.imshow(target_img)
    #     plt.axis("off")
    #     plt.subplot(1, im2print, im2print)
    #     plt.title("Pred", fontsize=10)
    #     plt.imshow(predicted_img)
    #     plt.axis("off")
    #     plt.show()

    # def _predict_latent_to_latent(self, x_test, y_test, vae, model, i):
    #     """Tool to generate images for report"""
    #     feed = np.zeros((32, 16,))
    #     original_images = np.zeros((32, 16))
    #     vae_data = x_test[:, :, :16]
    #     original_images[:self.steps] = vae_data[i % len(x_test)]
    #     original_images[self.steps + 1] = y_test[i % len(x_test)]
    #     original_images = vae.decode(original_images)
    #
    #     feed[0] = model.predict(x_test)[i % len(x_test)]
    #     predict_img = vae.decode(feed)
    #     self.plot_previous_images(original_images, self.steps, predict_img[0], original_images[self.steps + 1])

    # def _predict_latent_to_obs(self, x_test, y_test, model, i):
    #     """Old function that is not used anymore since we do not use obs"""
    #     idx = i % len(x_test)
    #     pred = model.predict(x_test)[idx]
    #     for t, y in zip(pred, y_test[idx]):
    #         print("Actual - predicted: {0:.3f} - {1:.3f}".format(y, t))
    # def plot_future(self, x, model, vae):
    #     """Function used to create a plot for our report."""
    #     state_queue = x[200]
    #     state_queue_length = len(state_queue)
    #     feed = np.zeros((32, 16,))
    #
    #     actual_imgs = np.zeros((32, 16,))
    #     actual_imgs[:state_queue_length] = x[200]
    #     actual_imgs[state_queue_length: 2 * state_queue_length] = x[201]
    #
    #     next_pred = model.predict(np.expand_dims(state_queue, axis=0))
    #     state_queue[-1] = next_pred
    #
    #     feed[:len(state_queue)] = state_queue
    #     images = list(vae.decode(feed)[:state_queue_length])
    #     for i in range(4):
    #         next_pred = model.predict(np.expand_dims(state_queue, axis=0))
    #         state_queue = np.roll(state_queue, -1, axis=0)
    #         state_queue[-1] = next_pred
    #         feed[:len(state_queue)] = state_queue
    #         new_img = vae.decode(feed)[state_queue_length - 1]
    #         images.append(new_img)
    #     plt.figure()
    #     for i in range(len(images)):
    #         plt.subplot(2, len(images), i + 1)
    #         plt.axis("off")
    #         plt.imshow(images[i])
    #
    #     for i, img in enumerate(vae.decode(actual_imgs)[:8]):
    #         plt.subplot(2, len(images), len(images) + i + 1)
    #         plt.axis("off")
    #         plt.imshow(img)
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.savefig('lstm_future_plot.pdf', format='pdf')
    #     plt.show()

def main():

    """EXAMPLE CODE TO TRAIN MODELS"""
    all_reward_keys = ['reward_exp0.5_coeff0.2', 'reward_exp0.5_coeff0.5',
                       'reward_exp0.5_coeff1', 'reward_exp0.5_coeff2', 'reward_exp0.5_coeff5',
                       'reward_exp1_coeff0.2', 'reward_exp1_coeff0.5', 'reward_exp1_coeff1',
                       'reward_exp1_coeff2', 'reward_exp1_coeff5', 'reward_exp2_coeff0.2',
                       'reward_exp2_coeff0.5', 'reward_exp2_coeff1', 'reward_exp2_coeff2',
                       'reward_exp2_coeff5', 'reward_exp3_coeff0.2', 'reward_exp3_coeff0.5',
                       'reward_exp3_coeff1', 'reward_exp3_coeff2', 'reward_exp3_coeff5',
                       'reward_original']

    dataset_detail_path = '../data/push_sphere_v2_details.pkl'
    vae_path = '../scripts/colab_models/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json'
    dataset_path = '../data/push_sphere_v2_latent_kl2rl1-z16-b250.npz'
    l2l_modelpath = 'trained_models/l2lmodel.h5'

    for current_key in all_reward_keys:

        reward_key = current_key #'reward_exp2_coeff1'

        l2reward_modelpath = f'trained_models/l2rewardmodel-{reward_key}.h5'

        example_train = LSTMModel(vae_path, z_size=16, steps=4, training=True, l2l_model_path=l2l_modelpath,
                                  l2reward_model_path=l2reward_modelpath, dataset_path=dataset_path,
                                  dataset_detail_path=dataset_detail_path, reward_key=reward_key)

        # example_train.build_l2l_model()
        example_train.build_l2reward_model()
        print("DONE")


if __name__ == '__main__':
    main()