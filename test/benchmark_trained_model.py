import time

from stable_baselines3 import (
    HerReplayBuffer,
    SAC,
)
import gymnasium as gym
# import rlnbv.environment
from rlnbv.environment.env_panda_bullet import PandaBulletEnv

# env = PandaBulletEnv(
#     render_mode="rgb_array",
# )

env = gym.make(
    "RoboRL-Navigator-Panda-Bullet",
    render_mode="rgb_array",
)

model = SAC.load(
    '/home/furkanduman/dev/RL-NBV/models/roborl-navigator/archive/PROD/model.zip',
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
results = []

for i in range(10_000):
    action = model.predict(observation)
    _, prev_view_count = env.task.get_view_array()
    observation, reward, terminated, info = model.env.step(action[0])
    new_unique_view_count = env.last_success_count
    print(f"Prev. View count: {prev_view_count} - New View count: {new_unique_view_count} - {i}")
    results.append((prev_view_count, new_unique_view_count, observation[0][1]))
    observation = model.env.reset()
    model.env.render()
    if i % 100 == 0 and i > 0:
        print("\n>>> RESET ENV ----- \n")
        del env
        time.sleep(3)
        env = gym.make(
            "RoboRL-Navigator-Panda-Bullet",
            render_mode="rgb_array",
        )
        model.set_env(env)
        model.env.reset()

file = open("results.txt", "w")
for result in results:
    file.write(f"{result[0]}\t{result[1]}\t{result[2]}\n")

# observation = model.env.reset()

