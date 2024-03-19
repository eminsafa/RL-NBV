from stable_baselines3 import SAC


def get_action(distance: float, radius: float) -> float:
    return SAC.load('/home/furkanduman/dev/RL-NBV/models/roborl-navigator/archive/PROD/model.zip').predict([[distance, radius]])[0][0][0]


for r in [round(0 + i * 0.001, 2) for i in range(int((0.1 - 0.09) / 0.001) + 1)]:
    print(get_action(0.6, r))

