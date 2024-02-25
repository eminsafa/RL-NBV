import numpy as np

from rlnbv.environment.env_panda_bullet import PandaBulletEnv

"""
TEST Bullet Environment Initialization
"""

env = PandaBulletEnv(render_mode="human")

# env = PandaBulletEnv(render_mode="human")

env.reset()
action = np.zeros(7)
for i in range(1_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or terminated:
        env.reset()
    else:
        print("reward: ", reward)
        if i > 0 and i % 15 == 0:
            env.reset()
            print("\n\t=== NEW EPISODE ===\n")

