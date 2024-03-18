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

env = gym.make(
    "RoboRL-Navigator-Panda-Bullet",
    render_mode="rgb_array",
)

if True:
    model = DDPG(policy="MlpPolicy", env=env, verbose=1)
    trainer = Trainer(model=model, target_step=50_000, log_interval=10)
else:
    model = SAC.load(
        '/home/furkanduman/dev/RL-NBV/models/roborl-navigator/FEB_27_1/model.zip',
        env=env,
    )
    trainer = Trainer(model=model, target_step=1_000, log_interval=10,
                      directory_path="/home/furkanduman/dev/RL-NBV/models/roborl-navigator/FEB_27_1")

trainer.train()
