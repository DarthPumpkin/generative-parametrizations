from pathlib import Path

from tqdm import tqdm
from cv2 import resize, INTER_AREA
import numpy as np
import gym


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


def _reset_fetch_sphere_big_longer_color(env):
    raw_env = env.unwrapped
    model = raw_env.sim.model
    object_site_i = model.site_names.index('object0')
    yellow = (1.0, 1.0, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)
    random_i = np.random.choice(2)
    model.site_rgba[object_site_i] = [yellow, blue][random_i]


def _reset_pendulum_var_length(env):
    raw_env = env.unwrapped
    raw_env.sampled_mass = np.random.uniform(0.05, 1.5)
    scale_range = (0.2, 1.2)
    new_scale = np.clip((raw_env.sampled_mass + 0.2) * 0.7, *scale_range)
    raw_env.length_scale = new_scale


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
    )
)


def generate(config_name: str):

    config = CONFIGS[config_name]
    episodes = config.get('episodes', 200)
    episode_length = config.get('episode_length', 5)
    img_size = config.get('size', (256, 256))
    rgb_options = config.get('rgb_options', None)
    env_setup = config.get('env_setup', None)
    env_reset = config.get('env_reset', None)
    action_period = config.get('action_period', 1)

    file_path = f'../data/{config_name}_imgs.npz'
    if Path(file_path).exists():
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
    data = []
    for e in tqdm(range(episodes)):
        env.reset()
        if callable(env_reset):
            env_reset(env)
        for s in range(episode_length):
            if s % action_period == 0:
                rand_action = raw_env.action_space.sample()
            img = env.render(**render_kwargs)
            img = resize(img, dsize=img_size, interpolation=INTER_AREA)
            data.append(img)
            env.step(rand_action)

    data = np.array(data)
    np.savez_compressed(file_path, data)
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


if __name__ == '__main__':
    for k in CONFIGS.keys():
        generate(k)
    crop('fetch_sphere_big_longer', 64)
    crop('fetch_sphere_big_longer_color', 64)
