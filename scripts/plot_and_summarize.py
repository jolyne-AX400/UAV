import csv
import os
import statistics
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('logs/figs', exist_ok=True)

runs = [
    ('ude', 'logs/training_logs_ude_2500_seed42.csv', 'logs/adj_stats_ude_2500_seed42.csv'),
    ('no_ude', 'logs/training_logs_no_ude_2500_seed42.csv', 'logs/adj_stats_no_ude_2500_seed42.csv')
]

summary = {}

# Read training logs
for name, tlog, alog in runs:
    summary[name] = {}
    eps = []
    rewards = []
    explored = []
    if os.path.exists(tlog):
        with open(tlog, newline='') as f:
            reader = csv.DictReader(f)
            for i,row in enumerate(reader, start=1):
                # try multiple possible column names
                tr = row.get('total_reward') or row.get('reward') or row.get('total_reward:')
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
    summary[name]['eps'] = eps
    summary[name]['rewards'] = rewards
    summary[name]['explored'] = explored

# Read adj stats and aggregate per episode mean(adj_mean)
for name, tlog, alog in runs:
    per_ep_adj = {}
    if os.path.exists(alog):
        with open(alog, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(float(row.get('episode', 0)))
                except Exception:
                    continue
                try:
                    adjm = float(row.get('adj_mean', 0))
                except Exception:
                    continue
                per_ep_adj.setdefault(ep, []).append(adjm)
    # build list aligned with episode indices in training log (if present)
    max_ep = max(per_ep_adj.keys()) if per_ep_adj else 0
    adj_means = []
    if 'eps' in summary[name] and summary[name]['eps']:
        N = len(summary[name]['eps'])
        for i in range(1, N+1):
            vals = per_ep_adj.get(i, [])
            if vals:
                adj_means.append(statistics.mean(vals))
            else:
                adj_means.append(float('nan'))
    else:
        # fallback: use per_ep_adj sorted
        for ep in range(1, max_ep+1):
            vals = per_ep_adj.get(ep, [])
            if vals:
                adj_means.append(statistics.mean(vals))
            else:
                adj_means.append(float('nan'))
    summary[name]['adj_means'] = adj_means

# Helper: moving average
def moving_average(x, w=20):
    res = []
    s = 0.0
    q = []
    for val in x:
        if math.isnan(val):
            q.append(val)
            res.append(float('nan'))
            continue
        q.append(val)
        s += val
        if len(q) > w:
            old = q.pop(0)
            if not math.isnan(old):
                s -= old
        # compute average over non-nan in q
        valid = [v for v in q if not math.isnan(v)]
        if valid:
            res.append(sum(valid)/len(valid))
        else:
            res.append(float('nan'))
    return res

# Plot rewards
plt.figure(figsize=(10,5))
for name in ['ude','no_ude']:
    eps = summary[name]['eps']
    rewards = summary[name]['rewards']
    if eps and any([not math.isnan(r) for r in rewards]):
        ma = moving_average(rewards, w=20)
        plt.plot(eps, ma, label=name)
plt.xlabel('Episode')
plt.ylabel('Total reward (20-ep MA)')
plt.title('Reward (20-ep moving avg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/figs/reward_compare_2500.png')
plt.close()

# Plot explored_ratio
plt.figure(figsize=(10,5))
for name in ['ude','no_ude']:
    eps = summary[name]['eps']
    explored = summary[name]['explored']
    if eps and any([not math.isnan(e) for e in explored]):
        ma = moving_average(explored, w=20)
        plt.plot(eps, ma, label=name)
plt.xlabel('Episode')
plt.ylabel('Explored ratio (20-ep MA)')
plt.title('Explored Ratio (20-ep moving avg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/figs/explored_compare_2500.png')
plt.close()

# Plot adj_mean
plt.figure(figsize=(10,5))
for name in ['ude','no_ude']:
    adj = summary[name]['adj_means']
    if adj and any([not math.isnan(a) for a in adj]):
        ma = moving_average(adj, w=20)
        plt.plot(range(1, len(adj)+1), ma, label=name)
plt.xlabel('Episode')
plt.ylabel('Adj mean (20-ep MA)')
plt.title('Adj Mean per Episode (20-ep moving avg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/figs/adj_mean_compare_2500.png')
plt.close()

# Write summary text
with open('logs/summary_2500_runs.txt', 'w') as out:
    out.write('Summary of 2500-episode runs\n')
    out.write('Files generated: logs/figs/reward_compare_2500.png, logs/figs/explored_compare_2500.png, logs/figs/adj_mean_compare_2500.png\n\n')
    for name in ['ude','no_ude']:
        out.write('--- {} ---\n'.format(name))
        rewards = [r for r in summary[name]['rewards'] if not math.isnan(r)]
        explored = [e for e in summary[name]['explored'] if not math.isnan(e)]
        adj = [a for a in summary[name]['adj_means'] if not math.isnan(a)]
        if rewards:
            out.write('episodes: {}\n'.format(len(rewards)))
            out.write('mean total_reward: {:.2f}\n'.format(statistics.mean(rewards)))
            out.write('median total_reward: {:.2f}\n'.format(statistics.median(rewards)))
        else:
            out.write('no reward data\n')
        if explored:
            out.write('mean explored_ratio: {:.4f}\n'.format(statistics.mean(explored)))
        else:
            out.write('no explored_ratio data\n')
        if adj:
            out.write('mean adj_mean: {:.4f}\n'.format(statistics.mean(adj)))
        else:
            out.write('no adj data\n')
        out.write('\n')

print('Figures saved to logs/figs and summary saved to logs/summary_2500_runs.txt')
