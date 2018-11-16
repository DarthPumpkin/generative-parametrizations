from cv2 import resize, INTER_AREA
import numpy as np
import gym

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
    )
)


def generate(config_name: str):

    config = CONFIGS[config_name]
    episodes = config.get('episodes', 200)
    episode_length = config.get('episode_length', 5)
    img_size = config.get('size', (256, 256))
    rgb_options = config.get('rgb_options', None)

    env = gym.make(config['env'])
    raw_env = env.unwrapped

    render_kwargs = dict(mode='rgb_array')
    if rgb_options is not None:
        render_kwargs['rgb_options'] = rgb_options

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
    np.savez_compressed(f'../data/{config_name}_imgs', data)

    print('Done.')


if __name__ == '__main__':
    for k in CONFIGS.keys():
        generate(k)
