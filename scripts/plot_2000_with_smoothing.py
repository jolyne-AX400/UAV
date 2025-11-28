import os, csv, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('logs/figs', exist_ok=True)

runs = [
    ('ude', 'logs/training_logs_ude_2500_seed42.csv', 'logs/adj_stats_ude_2500_seed42.csv'),
    ('no_ude', 'logs/training_logs_no_ude_2500_seed42.csv', 'logs/adj_stats_no_ude_2500_seed42.csv')
]
max_ep = 2000

def moving_average(x, w=20):
    res = []
    n = len(x)
    for i in range(n):
        # window centered/backwards average: use last w values up to i
        start = max(0, i - w + 1)
        window = [v for v in x[start:i+1] if not math.isnan(v)]
        if window:
            res.append(sum(window)/len(window))
        else:
            res.append(float('nan'))
    return res

summary = {}
for name, tlog, alog in runs:
    eps = []
    rewards = []
    explored = []
    if os.path.exists(tlog):
        with open(tlog, newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                if i > max_ep:
                    break
                tr = row.get('total_reward') or row.get('reward')
                er = row.get('explored_ratio')
                try:
                    rewards.append(float(tr))
                except Exception:
                    rewards.append(float('nan'))
                try:
                    explored.append(float(er))
                except Exception:
                    explored.append(float('nan'))
                eps.append(i)
    summary[name] = {'eps': eps, 'rewards': rewards, 'explored': explored}

    # aggregate adj_mean per episode from alog
    per_ep_adj = {}
    if os.path.exists(alog):
        with open(alog, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(float(row.get('episode', 0)))
                except Exception:
                    continue
                if ep > max_ep:
                    continue
                try:
                    adjm = float(row.get('adj_mean', 0))
                except Exception:
                    continue
                per_ep_adj.setdefault(ep, []).append(adjm)
    adj_means = []
    for i in range(1, max_ep+1):
        vals = per_ep_adj.get(i, [])
        if vals:
            adj_means.append(statistics.mean(vals))
        else:
            adj_means.append(float('nan'))
    summary[name]['adj_means'] = adj_means

# Plotting helper to draw raw MA(20) and smooth MA(100)
def plot_metric(name, key, ylabel, fname):
    eps = summary[name]['eps'] if key != 'adj' else list(range(1, max_ep+1))
    vals = summary[name][key if key!='adj' else 'adj_means']
    # ensure length max_ep
    if len(vals) < max_ep:
        vals = vals + [float('nan')] * (max_ep - len(vals))
    ma20 = moving_average(vals, w=20)
    ma100 = moving_average(vals, w=100)

    plt.figure(figsize=(10,5))
    x = range(1, max_ep+1)
    plt.plot(x, ma20, label='MA 20', color='C0', linewidth=1)
    plt.plot(x, ma100, label='MA 100 (smoothed)', color='C1', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(f'{name} - {ylabel} (first {max_ep} episodes)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# Create plots for both runs and metrics
for name, _, _ in runs:
    plot_metric(name, 'rewards', 'Total reward (MA)', f'logs/figs/{name}_reward_2000_ma20_ma100.png')
    plot_metric(name, 'explored', 'Explored ratio (MA)', f'logs/figs/{name}_explored_2000_ma20_ma100.png')
    plot_metric(name, 'adj', 'Adj mean', f'logs/figs/{name}_adjmean_2000_ma20_ma100.png')

print('Saved plots for first 2000 episodes with MA20 and MA100 to logs/figs/')
