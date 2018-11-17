from pathlib import Path

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
    )
)


def generate(config_name: str):

    config = CONFIGS[config_name]
    episodes = config.get('episodes', 200)
    episode_length = config.get('episode_length', 5)
    img_size = config.get('size', (256, 256))
    rgb_options = config.get('rgb_options', None)
    env_setup = config.get('env_setup', None)

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

    data = []

    for e in range(episodes):
        env.reset()
        for s in range(episode_length):
            img = env.render(**render_kwargs)
            img = resize(img, dsize=img_size, interpolation=INTER_AREA)
            data.append(img)
            rand_action = raw_env.action_space.sample()
            env.step(rand_action)

    data = np.array(data)
    np.savez_compressed(file_path, data)

    print('Done.')


if __name__ == '__main__':
    for k in CONFIGS.keys():
        generate(k)
