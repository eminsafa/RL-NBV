import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file into a DataFrame
models = {
    "TD3": {'i': 1, "marker": "o", "color": "blue"},
    "SAC": {"i": 2, "marker": "s", "color": "red"},
    # "DDPG": {"i": 2, "marker": "^", "color": "green"},
}
plt.figure(figsize=(12, 6))

for model, d in models.items():
    i = d['i']
    plt.subplot(1, 2, i)

    data = []
    file = open(f'/home/furkanduman/dev/RL-NBV/models/roborl-navigator/{model}/logs/merged.csv')
    index = 0
    for i in file.readlines():
        index += 1
        if index < 25:
            continue

        if index > 250:
            break

        sep = i.split(',')
        try:
            r = float(sep[0])
            l = float(sep[1])
        except Exception as e:
            print(e)
            continue
        data.append(min(r, l))

    #success_rate = list(data['rollout/ep_rew_mean'])
    #timesteps = data['time/total_timesteps']
    plt.plot(data, color=models[model]['color'])
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward Mean')
    plt.title(f'{model} Reward Mean Over Time')
    ##plt.xlim(0, 100)  # Set the X-axis limits to be from 1 to 5
    plt.ylim(-60, 0)
    plt.grid(True)

# Show the plot

plt.show()