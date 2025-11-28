import csv, os, statistics

pairs = [
    ('ude', 'logs/training_logs_ude_2500_seed42.csv', 'logs/adj_stats_ude_2500_seed42.csv'),
    ('no_ude', 'logs/training_logs_no_ude_2500_seed42.csv', 'logs/adj_stats_no_ude_2500_seed42.csv')
]

for name, tlog, alog in pairs:
    print('---', name, '---')
    if os.path.exists(tlog):
        with open(tlog, newline='') as f:
            reader = csv.DictReader(f)
            rewards = []
            explored = []
            for row in reader:
                try:
                    rewards.append(float(row.get('total_reward', row.get('reward', 0))))
                except Exception:
                    pass
                try:
                    explored.append(float(row.get('explored_ratio', 0)))
                except Exception:
                    pass
        if rewards:
            print('episodes:', len(rewards))
            print('mean total_reward: {:.2f}'.format(statistics.mean(rewards)))
            print('median total_reward: {:.2f}'.format(statistics.median(rewards)))
        else:
            print('no reward data')
        if explored:
            print('mean explored_ratio: {:.4f}'.format(statistics.mean(explored)))
        else:
            print('no explored_ratio data')
    else:
        print(f'missing training log: {tlog}')

    if os.path.exists(alog):
        with open(alog, newline='') as f:
            reader = csv.DictReader(f)
            adj_means = []
            for row in reader:
                try:
                    adj_means.append(float(row.get('adj_mean', 0)))
                except Exception:
                    pass
        if adj_means:
            print('adj entries:', len(adj_means))
            print('mean adj_mean: {:.4f}'.format(statistics.mean(adj_means)))
        else:
            print('no adj_mean data')
    else:
        print(f'missing adj stats: {alog}')

    print()
