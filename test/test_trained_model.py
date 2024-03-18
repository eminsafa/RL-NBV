import time

from stable_baselines3 import (
    HerReplayBuffer,
    SAC,
)
import gymnasium as gym
import rlnbv.environment


env = gym.make(
    "RoboRL-Navigator-Panda-Bullet",
    render_mode="human",
)

model = SAC.load(
    '/home/furkanduman/dev/RL-NBV/models/roborl-navigator/FEB_27_22/model.zip',
    env=env,
)

observation = model.env.reset()
# Evaluate the agent

# for _ in range(50):
#     start_time = time.time()
#     action = model.predict(observation)
#     end_time = time.time()
#     planning_time = round((end_time - start_time) * 1000)
#     action = action[0]
#     observation, reward, terminated, info = model.env.step(action)
#     model.env.render()
#

episode = 0
sub_counter = 0
for i in range(10_000):
    action = model.predict(observation)
    observation, reward, terminated, info = model.env.step(action[0])
    if terminated or terminated:
        sub_counter = 0
        env.reset()
    else:
        if i > 0 and sub_counter % 15 == 0:
            env.reset()
            episode += 1
            print(f"\n\t=== NEW EPISODE - {episode} ===\n")
    sub_counter += 1
    model.env.render()





# observation = model.env.reset()

