import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file into a DataFrame
models = {
    "TD3": {'i': 1, "marker": "o", "color": "blue"},
    "SAC": {"i": 3, "marker": "s", "color": "red"},
    "DDPG": {"i": 2, "marker": "^", "color": "green"},
}
plt.figure(figsize=(14, 6))

for model, d in models.items():
    i = d['i']
    plt.subplot(1, 3, i)
    data = pd.read_csv(
        f'/home/furkanduman/dev/RL-NBV/models/roborl-navigator/DDPG/logs/progress.csv'
    )
    # Extract the 'rollout/success_rate' column

    success_rate = data['rollout/ep_rew_mean']
    timesteps = data['time/total_timesteps']
    plt.plot(timesteps, success_rate, linestyle='-', color=d['color'])
    plt.xlabel('Steps (K)')
    plt.ylabel('Success Rate')
    plt.title(f'{model} Success Rate Over Time')
    ##plt.xlim(0, 100)  # Set the X-axis limits to be from 1 to 5
    #plt.ylim(0, 1.05)
    plt.grid(True)

# Show the plot

plt.show()