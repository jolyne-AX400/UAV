import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(csv_path):
    df = pd.read_csv(csv_path)
    # group by episode
    grouped = df.groupby('episode')
    summary = grouped.agg(adj_mean_mean=('adj_mean','mean'),
                          adj_mean_std=('adj_mean','std'),
                          adj_diff_mean=('adj_diff','mean'),
                          adj_diff_std=('adj_diff','std'),
                          steps=('step','count'))
    return df, summary


def save_summary_txt(summary, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary.to_string())


def plot_adj_mean(summary, tag, out_dir):
    plt.figure(figsize=(8,4))
    plt.errorbar(summary.index, summary['adj_mean_mean'], yerr=summary['adj_mean_std'].fillna(0), fmt='-o')
    plt.xlabel('Episode')
    plt.ylabel('Mean(adj_mean)')
    plt.title(f'Adj Mean per Episode ({tag})')
    plt.grid(True)
    p = os.path.join(out_dir, f'adj_mean_{tag}.png')
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p


def plot_adj_diff(summary, tag, out_dir):
    plt.figure(figsize=(8,4))
    plt.plot(summary.index, summary['adj_diff_mean'], '-o')
    plt.xlabel('Episode')
    plt.ylabel('Mean(adj_diff)')
    plt.title(f'Adj Diff per Episode ({tag})')
    plt.grid(True)
    p = os.path.join(out_dir, f'adj_diff_{tag}.png')
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p


def plot_step_heatmap(df, tag, out_dir, max_steps=200):
    # create pivot table: rows=episode, cols=step -> adj_mean
    # to avoid huge images, limit steps to first max_steps
    pivot = df.pivot(index='episode', columns='step', values='adj_mean')
    # reindex columns to 0..max_steps-1
    cols = [c for c in pivot.columns if c < max_steps]
    data = pivot[cols].values
    plt.figure(figsize=(10, max(2, data.shape[0]*0.2)))
    plt.imshow(data, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='adj_mean')
    plt.xlabel('Step')
    plt.ylabel('Episode')
    plt.title(f'Adj Mean Heatmap ({tag})')
    p = os.path.join(out_dir, f'adj_heatmap_{tag}.png')
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p


def find_spikes(df, top_k=20):
    # return top_k rows by adj_diff
    top = df.nlargest(top_k, 'adj_diff')
    return top[['episode','step','adj_mean','adj_diff']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to adj_stats csv')
    parser.add_argument('--tag', default='run', help='tag for outputs')
    parser.add_argument('--out', default='logs/figs', help='output directory')
    parser.add_argument('--top-k', type=int, default=20, help='top k spikes to save')
    args = parser.parse_args()

    csv = args.input
    tag = args.tag
    out = args.out
    os.makedirs(out, exist_ok=True)
    base_out = os.path.join('logs', f'adj_summary_{tag}.txt')

    try:
        df, summary = summarize(csv)
    except Exception as e:
        print('Failed to read or summarize CSV:', e)
        sys.exit(2)

    save_summary_txt(summary, base_out)
    plots = []
    plots.append(plot_adj_mean(summary, tag, out))
    plots.append(plot_adj_diff(summary, tag, out))
    try:
        plots.append(plot_step_heatmap(df, tag, out))
    except Exception:
        # if pivot fails (e.g., steps too many or irregular), skip
        pass

    spikes = find_spikes(df, top_k=args.top_k)
    spikes_out = os.path.join('logs', f'adj_spikes_{tag}.csv')
    spikes.to_csv(spikes_out, index=False)

    print('Summary saved to', base_out)
    print('Spikes saved to', spikes_out)
    print('Plots saved:', ', '.join(plots))

if __name__ == '__main__':
    main()
