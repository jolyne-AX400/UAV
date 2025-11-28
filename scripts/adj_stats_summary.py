import csv, statistics
from collections import defaultdict
rows=defaultdict(list)
with open('logs/adj_stats.csv') as f:
    r=csv.reader(f)
    next(r)
    for ep,step,mean,diff in r:
        ep=int(ep)
        rows[ep].append((float(mean),float(diff)))
for ep in sorted(rows):
    means=[m for m,d in rows[ep]]
    diffs=[d for m,d in rows[ep]]
    print(f'EP {ep}: mean(adj) mean={statistics.mean(means):.6f}, std={statistics.pstdev(means):.6f}, mean(diff)={statistics.mean(diffs):.6f}')
