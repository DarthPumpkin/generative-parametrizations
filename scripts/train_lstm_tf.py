import numpy as np

import _gpp
from gpp.lstm import LSTMModel


def main():

    vae_path = './tf_vae/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json'
    l2l_modelpath = './out/push_sphere_v1_l2lmodel.hdf5'
    l2reward_modelpath = './out/push_sphere_v1_l2rewardmodel.hdf5'

    dataset_path = '../data/push_sphere_v1_latent_kl2rl1-z16-b250.npz'
    dataset_detail_path = '../data/push_sphere_v1_details.pkl'

    model = LSTMModel(vae_path, z_size=16, steps=4, training=True, l2l_model_path=l2l_modelpath,
                      l2reward_model_path=l2reward_modelpath, dataset_path=dataset_path,
                      dataset_detail_path=dataset_detail_path, reward_key='reward_modified')

    model.build_l2l_model()
    model.build_l2reward_model()
    print("DONE")


if __name__ == '__main__':
    main()
