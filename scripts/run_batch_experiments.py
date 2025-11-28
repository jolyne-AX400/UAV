"""Batch-run experiments: run multiple configurations sequentially and save logs/adj_stats per run.
Usage: python scripts/run_batch_experiments.py
This script launches `train.py` with different flags, captures stdout to logs/run_<tag>_<seed>.log,
and renames logs/adj_stats.csv to logs/adj_stats_<tag>_<seed>.csv after each run.
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / 'train.py'
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)

# Experiment grid
configs = [
    {'tag': 'no_ude', 'use_ude': False},
    {'tag': 'ude', 'use_ude': True},
]
seeds = [42, 43]

# training params (moderate length to permit reasonably full exploration)
episodes = 100
steps_per_episode = 200

# other args
top_k = 3
adj_threshold = 0.5
alpha_sparse = 0.01

def run_one(cfg, seed):
    tag = cfg['tag']
    use_ude = cfg['use_ude']
    log_file = LOGS / f'run_{tag}_seed{seed}.log'
    # build command
    cmd = [sys.executable, str(TRAIN_PY),
           '--episodes', str(episodes),
           '--steps-per-episode', str(steps_per_episode),
           '--top-k', str(top_k),
           '--adj-threshold', str(adj_threshold),
           '--alpha-sparse', str(alpha_sparse),
           '--seed', str(seed),
           '--log-interval', '1',
           '--save-interval', '100'
    ]
    if use_ude:
        cmd.append('--use-ude')

    print('Running:', ' '.join(cmd))
    with open(log_file, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), universal_newlines=True)
        # stream output to file and console
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end='')
        proc.wait()
    # after run, rename adj_stats.csv if exists
    adj_src = LOGS / 'adj_stats.csv'
    if adj_src.exists():
        adj_dst = LOGS / f'adj_stats_{tag}_seed{seed}.csv'
        if adj_dst.exists():
            # avoid overwrite
            adj_dst = LOGS / f'adj_stats_{tag}_seed{seed}_dup.csv'
        adj_src.rename(adj_dst)
        print('Saved adj stats to', adj_dst)
    else:
        print('Warning: adj_stats.csv not found after run', tag, seed)


if __name__ == '__main__':
    for cfg in configs:
        for seed in seeds:
            run_one(cfg, seed)

    print('All experiments finished. Collected CSVs in logs/')
