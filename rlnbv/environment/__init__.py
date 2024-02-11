from gymnasium.envs.registration import register
from .base_env import BaseEnv

register(
    id="RoboRL-Navigator-Panda-Bullet",
    entry_point="rlnbv.environment.env_panda_bullet:PandaBulletEnv",
    max_episode_steps=3,
)
