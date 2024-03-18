import gymnasium as gym

from stable_baselines3 import (
    SAC,
    DDPG,
    TD3
)

from trainer import Trainer
import rlnbv.environment


def get_env():
    return gym.make(
        "RoboRL-Navigator-Panda-Bullet",
        render_mode="rgb_array",
    )


import os


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


# Usage
merge_csv('progress.csv', 'merged.csv')

raw_path = "/home/furkanduman/dev/RL-NBV/models/roborl-navigator/"
log_interval = 100
render_mode = "rgb_array"

for model_name in ['DDPG', 'TD3', 'SAC']:
    path = raw_path + model_name
    target_steps = 10
    for i in range(target_steps):
        if i == 0:
            if model_name == 'DDPG':
                model = DDPG(policy="MlpPolicy", env=get_env(), verbose=1)
            elif model_name == 'SAC':
                model = SAC(policy="MlpPolicy", env=get_env(), verbose=1)
            elif model_name == 'TD3':
                model = TD3(policy="MlpPolicy", env=get_env(), verbose=1)

            trainer = Trainer(model=model, target_step=500, log_interval=log_interval, directory_path=path)
        else:
            model.set_env(env=get_env())
            trainer = Trainer(model=model, target_step=500, log_interval=log_interval, directory_path=path)

        trainer.train()
        print(f"\n\n\t\t >>>>> Step {i+1}/{target_steps}")
        del trainer
