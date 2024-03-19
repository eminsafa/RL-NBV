import os
import gymnasium as gym

from stable_baselines3 import (
    SAC,
    DDPG,
    TD3,
    PPO,
)

from trainer import Trainer
import rlnbv.environment


def get_env():
    return gym.make(
        "RoboRL-Navigator-Panda-Bullet",
        render_mode="rgb_array",
    )


def merge_csv(progress_file, merged_file):
    # Check if merged.csv exists and if it has content to determine if header needs to be skipped
    file_exists = os.path.isfile(merged_file)
    skip_header = file_exists and os.path.getsize(merged_file) > 0

    with open(progress_file, 'r') as progress:
        lines = progress.readlines()

    with open(merged_file, 'a') as merged:
        if not skip_header:
            merged.writelines(lines)  # Write with header if file is new or empty
        else:
            merged.writelines(lines[1:])  # Skip header otherwise


def train_model(env_id, total_loop, episodes_per_env, model_name="DDPG"):
    if model_name == 'DDPG':
        model = DDPG("MlpPolicy", env_id, verbose=0)
    elif model_name == 'SAC':
        model = SAC("MlpPolicy", env_id, verbose=1)
    elif model_name == 'TD3':
        model = TD3("MlpPolicy", env_id, verbose=1)
    elif model_name == 'PPO':
        model = PPO("MlpPolicy", env_id, verbose=1)

    episode_counts = 0
    for _ in range(total_loop):
        if episode_counts % episodes_per_env == 0:
            model.set_env(get_env())

        trainer = Trainer(model=model, target_step=100, log_interval=2, directory_path=path)
        trainer.train()
        logs_path = os.path.join(path, "logs")
        merge_csv(logs_path + '/progress.csv', logs_path + '/merged.csv')
        episode_counts += 1


raw_path = "/home/furkanduman/dev/RL-NBV/models/roborl-navigator/"

for model_name in ['DDPG', 'TD3', 'SAC']:
    path = raw_path + model_name
    train_model("RoboRL-Navigator-Panda-Bullet", 25, 100, model_name)


