try:
    import google.colab
    import sys
    sys.path.insert(1, '/content/RL-NBV')
except:
    pass

import gymnasium as gym

from stable_baselines3 import (
    TD3,
    DDPG,
    SAC,
)

from trainer import Trainer
import rlnbv.environment


def get_env():
    return gym.make(
        "RoboRL-Navigator-Panda-Bullet",
        render_mode="rgb_array",
    )


target_step = 10_000
path = "/home/furkanduman/dev/RL-NBV/models/roborl-navigator/"
log_interval = 250

model = DDPG(policy="MlpPolicy", env=get_env(), verbose=1)
trainer = Trainer(model=model, target_step=target_step, log_interval=log_interval, directory_path=path+'DDPG_50K_failed')
trainer.train()

model = TD3(policy="MlpPolicy", env=get_env(), verbose=1)
trainer = Trainer(model=model, target_step=target_step, log_interval=log_interval, directory_path=path+'TD3_50K')
trainer.train()

model = SAC(policy="MlpPolicy", env=get_env(), verbose=1)
trainer = Trainer(model=model, target_step=target_step, log_interval=log_interval, directory_path=path+'SAC_50K')
trainer.train()



