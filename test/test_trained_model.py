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
    '/home/furkanduman/dev/RL-NBV/models/roborl-navigator/FEB_24_1/model.zip',
    env=env,
)

observation = model.env.reset()
# Evaluate the agent

for _ in range(50):
    start_time = time.time()
    action = model.predict(observation)
    end_time = time.time()
    planning_time = round((end_time - start_time) * 1000)
    action = action[0]
    print(f"Action: {action[0][0]}")
    observation, reward, terminated, info = model.env.step(action)
    print(f"Reward: {reward}\n")
    model.env.render()




# observation = model.env.reset()

