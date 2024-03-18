import numpy as np

from rlnbv.environment.env_panda_bullet import PandaBulletEnv

"""
TEST Bullet Environment Initialization
"""

env = PandaBulletEnv(render_mode="human")

# env = PandaBulletEnv(render_mode="human")

episode = 0
env.reset()
action = np.zeros(7)
sub_counter = 0
for i in range(10_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or terminated:
        sub_counter = 0
        env.reset()
    else:
        if i > 0 and sub_counter % 15 == 0:
            env.reset()
            episode += 1
            print(f"\n\t=== NEW EPISODE - {episode} ===\n")
    sub_counter += 1
