import os.path

try:
    import google.colab
    import sys
    sys.path.insert(1, '/content/RL-NBV')
except:
    pass

import gymnasium as gym

from stable_baselines3 import (
    SAC,
)

from trainer import Trainer
import rlnbv.environment

path = "/home/furkanduman/dev/RL-NBV/models/roborl-navigator/PROD"
render_mode = "rgb_array"

for i in range(40):
    env = gym.make(
        "RoboRL-Navigator-Panda-Bullet",
        render_mode=render_mode,
    )

    if i == 0:
        model = SAC(policy="MlpPolicy", env=env, verbose=1)
        trainer = Trainer(model=model, target_step=500, log_interval=10, directory_path=path)
    else:
        env = gym.make(
            "RoboRL-Navigator-Panda-Bullet",
            render_mode=render_mode,
        )
        model.set_env(env=env)
        trainer = Trainer(model=model, target_step=500, log_interval=10, directory_path=path)

    env.close()
    trainer.train()

    del env
    del trainer
