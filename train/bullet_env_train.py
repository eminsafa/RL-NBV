try:
    import google.colab
    import sys
    sys.path.insert(1, '/content/RL-NBV')
except:
    pass

import gymnasium as gym

from stable_baselines3 import (
    TD3,
)

from trainer import Trainer
import rlnbv.environment

env = gym.make(
    "RoboRL-Navigator-Panda-Bullet",
    render_mode="rgb_array",
)

model = TD3(policy="MlpPolicy", env=env, verbose=1)

trainer = Trainer(model=model, target_step=1_000)

trainer.train()
