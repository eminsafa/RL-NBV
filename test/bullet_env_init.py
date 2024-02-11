import numpy as np

from rlnbv.environment.env_panda_bullet import PandaBulletEnv

"""
TEST Bullet Environment Initialization
"""

env = PandaBulletEnv(render_mode="rgb_array")

# env = PandaBulletEnv(render_mode="human")

env.reset()

action = np.ones(7)
observation, reward, terminated, truncated, info = env.step(action)
action = np.zeros(7)
for i in range(1_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if i % 50 == 0:
        env.reset()
        print("ENV reset")

