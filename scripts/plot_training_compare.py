import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('logs/figs', exist_ok=True)

no_path = 'logs/training_logs_no_ude_500_seed42.csv'
ude_path = 'logs/training_logs_ude_500_seed42.csv'

no = pd.read_csv(no_path)
ude = pd.read_csv(ude_path)

# plot rewards
plt.figure(figsize=(8,4))
plt.plot(no['episode'], no['total_reward'], label='no_ude')
plt.plot(ude['episode'], ude['total_reward'], label='ude')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.title('Reward curve (500 episodes)')
plt.tight_layout()
plt.savefig('logs/figs/reward_compare_500_seed42.png')
plt.close()

# plot explored_ratio
plt.figure(figsize=(8,4))
plt.plot(no['episode'], no['explored_ratio'], label='no_ude')
plt.plot(ude['episode'], ude['explored_ratio'], label='ude')
plt.xlabel('Episode')
plt.ylabel('Explored Ratio')
plt.legend()
plt.title('Explored Ratio (500 episodes)')
plt.tight_layout()
plt.savefig('logs/figs/explored_ratio_compare_500_seed42.png')
plt.close()

# save a small CSV with mean explored_ratio over last 50 episodes
summary = {
    'no_ude_mean_explored_last50': no['explored_ratio'].tail(50).mean(),
    'ude_mean_explored_last50': ude['explored_ratio'].tail(50).mean(),
}
import json
with open('logs/figs/explored_summary_500_seed42.json','w') as f:
    json.dump(summary, f, indent=2)

print('Plots saved to logs/figs/ and summary to logs/figs/explored_summary_500_seed42.json')
