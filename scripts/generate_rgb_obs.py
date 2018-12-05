from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from cv2 import resize, INTER_AREA
import numpy as np
import gym


from gpp.world_models_vae import ConvVAE
from gpp.models.utilities import get_observations


def _setup_fetch_sphere_big(env):
    raw_env = env.unwrapped
    model = raw_env.sim.model
    raw_env.target_in_the_air = False
    raw_env.mocap_bodies_visible = False
    object_site_i = model.site_names.index('object0')
    target_site_i = model.site_names.index('target0')
    model.site_size[object_site_i] *= 1.5 * 2.0
    model.site_size[target_site_i] *= 1.5


def _setup_fetch_sphere_big_longer(env):
    raw_env = env.unwrapped
    raw_env.block_gripper = True
    _setup_fetch_sphere_big(env)


def _push_strategy_v0(env, obs):
    raw_env = env.unwrapped
    gripper_pos = obs[:3]
    object_pos = obs[3:6]
    delta = object_pos - gripper_pos
    dir_ = delta / np.linalg.norm(delta)
    action = (np.r_[delta + dir_*0.5, 0.0]) * 5.0
    return action.clip(raw_env.action_space.low, raw_env.action_space.high)


SPHERE_COLORS = dict(
    yellow=(1.0, 1.0, 0.0, 1.0),
    blue=(0.0, 0.0, 1.0, 1.0)
)


SPHERE_MASSES = dict(
    yellow=0.1/5,
    blue=5.0
)


def _reset_fetch_sphere_big_longer_color(env):
    raw_env = env.unwrapped
    model = raw_env.sim.model
    object_site_i = model.site_names.index('object0')
    yellow = (1.0, 1.0, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)
    random_i = np.random.choice(2)
    model.site_rgba[object_site_i] = [yellow, blue][random_i]
    return None


def _reset_pendulum_var_length(env):
    raw_env = env.unwrapped
    raw_env.sampled_mass = np.random.uniform(0.05, 1.5)
    scale_range = (0.2, 1.2)
    new_scale = np.clip((raw_env.sampled_mass + 0.2) * 0.7, *scale_range)
    raw_env.length_scale = new_scale
    return None


def _reset_push_sphere_v0(env):
    raw_env = env.unwrapped
    model = raw_env.sim.model
    object_site_i = model.site_names.index('object0')
    object_body_i = model.body_names.index('object0')

    random_i = np.random.choice(2)
    color_name, color = list(SPHERE_COLORS.items())[random_i]
    mass = SPHERE_MASSES[color_name]

    model.site_rgba[object_site_i] = color
    model.body_mass[object_body_i] = mass

    return dict(
        sphere_color=color_name,
        sphere_mass=mass
    )


CONFIGS = dict(
    pendulum=dict(
        env='Pendulum-v0',
        size=(256, 256),
        episodes=200,
        episode_length=5
    ),
    fetch_sphere=dict(
        env='FetchPickAndPlaceSphere-v1',
        size=(256, 256),
        episodes=200,
        episode_length=5,
        rgb_options=dict(camera_id=3)
    ),
    fetch_sphere_big=dict(
        env='FetchPickAndPlaceSphere-v1',
        size=(256, 256),
        episodes=200,
        episode_length=5,
        rgb_options=dict(camera_id=3),
        env_setup=_setup_fetch_sphere_big
    ),
    fetch_sphere_big_longer=dict(
        env='FetchPickAndPlaceSphere-v1',
        size=(64, 64),
        episodes=200,
        episode_length=32,
        rgb_options=dict(camera_id=3),
        action_period=6,
        env_setup=_setup_fetch_sphere_big_longer
    ),
    fetch_sphere_big_longer_color=dict(
        env='FetchPickAndPlaceSphere-v1',
        size=(64, 64),
        episodes=200,
        episode_length=32,
        rgb_options=dict(camera_id=3),
        action_period=6,
        env_setup=_setup_fetch_sphere_big_longer,
        env_reset=_reset_fetch_sphere_big_longer_color
    ),
    pendulum_var_length=dict(
        env='Pendulum-v0',
        size=(256, 256),
        episodes=200,
        episode_length=5,
        env_reset=_reset_pendulum_var_length
    ),
    push_sphere_v0=dict(
        env='FetchPushSphere-v1',
        size=(64, 64),
        episodes=1000,
        episode_length=10,
        rgb_options=dict(camera_id=3),
        action_strategy_eps=1./4,
        action_strategy=_push_strategy_v0,
        env_setup=_setup_fetch_sphere_big_longer,
        env_reset=_reset_push_sphere_v0
    ),
)


def generate(config_name: str, overwrite=False):

    config = CONFIGS[config_name]
    episodes = config.get('episodes', 200)
    episode_length = config.get('episode_length', 5)
    img_size = config.get('size', (256, 256))
    rgb_options = config.get('rgb_options', None)
    env_setup = config.get('env_setup', None)
    env_reset = config.get('env_reset', None)
    action_strategy = config.get('action_strategy', None)
    action_strategy_eps = config.get('action_strategy_eps', 1.0)

    images_path = f'../data/{config_name}_imgs.npz'
    details_path = f'../data/{config_name}_details.pkl'
    if not overwrite and (Path(images_path).exists() or Path(details_path).exists()):
        print(f'Skipping {config_name}.')
        return

    env = gym.make(config['env'])
    raw_env = env.unwrapped

    render_kwargs = dict(mode='rgb_array')
    if rgb_options is not None:
        render_kwargs['rgb_options'] = rgb_options

    if callable(env_setup):
        env_setup(env)

    print(f'Generating {config_name}...')
    images = []
    details = []

    for e in tqdm(range(episodes)):
        env.reset()

        action = 0.0 * raw_env.action_space.sample()
        episode_info = None
        if callable(env_reset):
            episode_info = env_reset(env)
        episode_info = episode_info or dict()

        rand_action = raw_env.action_space.sample()
        sample_action = np.random.uniform() < action_strategy_eps

        for s in range(episode_length):

            raw_obs = get_observations(env)

            while True:
                img = env.render(**render_kwargs)
                img = resize(img, dsize=img_size, interpolation=INTER_AREA)
                if not is_image_corrupted(img):
                    break
                print('Corrupted image!')

            img_i = len(images)
            images.append(img)

            details.append(dict(
                image_index=img_i,
                step=s,
                episode=e,
                raw_obs=raw_obs,
                raw_action=action,
                **episode_info
            ))

            if sample_action:
                action = rand_action
            elif callable(action_strategy):
                action = action_strategy(env, raw_obs)
            else:
                raise ValueError('Action strategy must be callable!')

            action += np.random.normal(scale=1.0)
            env.step(action)

    images = np.array(images)
    np.savez_compressed(images_path, images)

    details = pd.DataFrame(details)
    pd.to_pickle(details, details_path)

    print('Done.')


def crop(config_name: str, size):

    if isinstance(size, int):
        size = (size, size)
    assert len(size) == 2, 'size must be a tuple of two ints'
    assert isinstance(size[0], int) and isinstance(size[1], int), 'size must be a tuple of two ints'

    file_path = f'../data/{config_name}_imgs.npz'
    if not Path(file_path).exists():
        print(f'Skipping {config_name} as it doesn\'t exist.')
        return

    data = np.load(file_path)['arr_0']
    if data.shape[1:3] == size:
        print(f'Skipping {config_name} as it is already at the desired size.')
        return

    assert data.shape[1] > size[0] and data.shape[2] > size[1], 'size must be smaller than original'

    print(f'Cropping {config_name}...')
    cropped = np.zeros((data.shape[0],) + size + data.shape[3:], dtype=data.dtype)
    for i, img in enumerate(data):
        cropped[i] = resize(img, dsize=size, interpolation=INTER_AREA)

    np.savez_compressed(file_path, cropped)
    print('Done.')


def find_corrupted_idx(data):
    threshold = 8000
    if data.max() > 1:
        threshold *= 255
    magic_threshold = threshold * np.prod(data.shape[1:]) / (64 * 64 * 3)
    sums = np.sum(data, axis=(1, 2, 3))
    corrupted_idx = np.where(sums < magic_threshold)[0]
    return corrupted_idx


def is_image_corrupted(img):
    data = img[None, :, :, :]
    return len(find_corrupted_idx(data)) > 0


def check_images(config_name: str, show_imgs=False):

    file_path = f'../data/{config_name}_imgs.npz'
    data = np.load(file_path)['arr_0']

    if show_imgs:
        idx = np.random.randint(0, len(data), 10)
        for img in data[idx]:
            plt.imshow(img)
            plt.show()

    corrupted_idx = find_corrupted_idx(data)
    print(f'Dataset contains {len(corrupted_idx)} corrupted image(s).')


def images_to_z(config_name: str, vae_model_descr: str, vae_model: Path, **vae_kwargs):

    config = CONFIGS[config_name]
    episodes = config['episodes']
    episode_length = config['episode_length']

    vae = ConvVAE(is_training=False, reuse=False, gpu_mode=False, **vae_kwargs)
    vae.load_json(vae_model)
    batch_size = vae.batch_size
    z_size = vae.z_size

    images_path = f'../data/{config_name}_imgs.npz'
    z_path = f'../data/{config_name}_latent_{vae_model_descr}.npz'

    data = np.load(images_path)['arr_0'] / 255.
    output_z = np.zeros((len(data), z_size))

    n_batches = int(np.ceil(len(data)/batch_size))
    for b in range(n_batches):
        batch = data[batch_size*b: batch_size*(b+1)]
        actual_bs = len(batch)
        if actual_bs < batch_size:
            padded = np.zeros((batch_size,) + data.shape[1:])
            padded[:actual_bs] = batch
            batch = padded
        batch_z = vae.encode(batch)[:actual_bs]
        output_z[batch_size*b: batch_size*(b+1)] = batch_z

    output_z = output_z.reshape((episodes, episode_length, -1))
    np.savez_compressed(z_path, output_z)


if __name__ == '__main__':

    generate('push_sphere_v0')
    check_images('push_sphere_v0', show_imgs=False)
    images_to_z('push_sphere_v0',
                'kl2rl1-z16-b250',
                'tf_vae/kl2rl1-z16-b250-push_sphere_v0vae-fetch199.json',
                z_size=16, batch_size=32)
