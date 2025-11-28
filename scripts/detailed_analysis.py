import csv, os, math, statistics
from math import isnan
from scipy import stats

pairs = [
    ('ude', 'logs/training_logs_ude_2500_seed42.csv', 'logs/adj_stats_ude_2500_seed42.csv'),
    ('no_ude', 'logs/training_logs_no_ude_2500_seed42.csv', 'logs/adj_stats_no_ude_2500_seed42.csv')
]

def read_training(tlog, max_ep=None):
    rewards=[]
    explored=[]
    if not os.path.exists(tlog):
        return rewards, explored
    with open(tlog, newline='') as f:
        reader = csv.DictReader(f)
        for i,row in enumerate(reader, start=1):
            if max_ep and i>max_ep: break
            tr = row.get('total_reward') or row.get('reward')
            er = row.get('explored_ratio')
            try:
                rewards.append(float(tr))
            except:
                rewards.append(float('nan'))
            try:
                explored.append(float(er))
            except:
                explored.append(float('nan'))
    return rewards, explored

def read_adj_per_episode(alog, max_ep=None):
    per_ep={} 
    if not os.path.exists(alog): return []
    with open(alog, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(float(row.get('episode',0)))
            except:
                continue
            if max_ep and ep>max_ep: continue
            try:
                adjm = float(row.get('adj_mean',0))
            except:
                continue
            per_ep.setdefault(ep,[]).append(adjm)
    max_ep = max(per_ep.keys()) if per_ep else 0
    res = []
    for i in range(1, max_ep+1):
        vals = per_ep.get(i,[])
        if vals:
            res.append(statistics.mean(vals))
        else:
            res.append(float('nan'))
    return res

print('Detailed analysis for runs (2500 episodes)')
print('-------------------------------------------')
for name,tlog,alog in pairs:
    print('\nRun:', name)
    rewards, explored = read_training(tlog)
    adj = read_adj_per_episode(alog)
    N = len(rewards)
    print('episodes in training log:', N)
    valid_rewards = [r for r in rewards if not isnan(r)]
    if not valid_rewards:
        print('No reward data')
        continue
    print('overall mean reward: {:.2f}, std: {:.2f}'.format(statistics.mean(valid_rewards), statistics.pstdev(valid_rewards)))
    # segments
    def seg_stats(arr, start, end):
        seg = [x for x in arr[start:end] if not isnan(x)]
        if not seg: return None
        return (statistics.mean(seg), statistics.pstdev(seg))
    first500 = seg_stats(rewards, 0, min(500, N))
    mid = seg_stats(rewards, max(0, N//2 - 250), N//2 + 250 if N>=500 else N)
    last500 = seg_stats(rewards, max(0, N-500), N)
    if first500: print('first 500 mean,std: {:.2f}, {:.2f}'.format(first500[0], first500[1]))
    if mid: print('middle 500 mean,std: {:.2f}, {:.2f}'.format(mid[0], mid[1]))
    if last500: print('last 500 mean,std: {:.2f}, {:.2f}'.format(last500[0], last500[1]))
    # explored
    valid_explored = [x for x in explored if not isnan(x)]
    if valid_explored:
        print('mean explored_ratio overall: {:.4f}'.format(statistics.mean(valid_explored)))
    # adj mean
    valid_adj = [x for x in adj if not isnan(x)]
    if valid_adj:
        print('mean adj_mean overall: {:.4f}'.format(statistics.mean(valid_adj)))
    # correlation between adj_mean and explored per episode (align lengths)
    L = min(len(adj), len(explored))
    if L>10:
        xs = [adj[i] for i in range(L) if not isnan(adj[i]) and not isnan(explored[i])]
        ys = [explored[i] for i in range(L) if not isnan(adj[i]) and not isnan(explored[i])]
        if len(xs)>5:
            r, p = stats.pearsonr(xs, ys)
            print('pearson corr adj_mean vs explored_ratio: r={:.3f}, p={:.3e}'.format(r,p))
        else:
            print('not enough paired data for correlation')
    else:
        print('not enough adj/explored overlap for correlation')

# Simple between-run comparison on last 500 episodes mean reward
r1,_ = read_training(pairs[0][1])
r2,_ = read_training(pairs[1][1])
if len(r1)>=500 and len(r2)>=500:
    s1 = r1[-500:]
    s2 = r2[-500:]
    s1v = [x for x in s1 if not isnan(x)]
    s2v = [x for x in s2 if not isnan(x)]
    print('\nBetween-run comparison (last 500 episodes):')
    print('ude last500 mean,std: {:.2f}, {:.2f}'.format(statistics.mean(s1v), statistics.pstdev(s1v)))
    print('no_ude last500 mean,std: {:.2f}, {:.2f}'.format(statistics.mean(s2v), statistics.pstdev(s2v)))
    # t-test
    try:
        tstat, pval = stats.ttest_ind(s1v, s2v, equal_var=False)
        print('t-test (unequal var): t={:.3f}, p={:.3e}'.format(tstat, pval))
    except Exception as e:
        print('t-test failed:', e)
else:
    print('Not enough data for between-run last500 comparison')
