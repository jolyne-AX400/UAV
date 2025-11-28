import pandas as pd
import numpy as np
import os
import sys

from math import sqrt

def summarize_df(df):
    g = df.groupby('episode')
    s = g.agg(adj_mean_mean=('adj_mean','mean'), adj_mean_std=('adj_mean','std'),
              adj_diff_mean=('adj_diff','mean'), adj_diff_std=('adj_diff','std'), steps=('step','count'))
    return s


def paired_ttest(x, y):
    # x,y are numpy arrays of same length
    d = x - y
    n = len(d)
    mean_d = d.mean()
    std_d = d.std(ddof=1)
    se = std_d / sqrt(n)
    if se == 0:
        return float('nan'), float('nan'), n-1
    t = mean_d / se
    # try to compute p-value using scipy if available
    try:
        from scipy import stats
        p = stats.t.sf(abs(t), df=n-1) * 2
    except Exception:
        p = float('nan')
    return t, p, n-1


def main():
    # allow custom paths via argv (fallback to default names)
    a_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/adj_stats_no_ude.csv'
    b_path = sys.argv[2] if len(sys.argv) > 2 else 'logs/adj_stats_ude.csv'
    if not os.path.exists(a_path) or not os.path.exists(b_path):
        print('Need both files:', a_path, b_path)
        sys.exit(2)

    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    sa = summarize_df(a)
    sb = summarize_df(b)

    # align on episodes present in both
    common = sa.index.intersection(sb.index)
    sa_c = sa.loc[common]
    sb_c = sb.loc[common]

    # compute paired tests on adj_mean_mean and adj_diff_mean
    adj_mean_x = sa_c['adj_mean_mean'].values
    adj_mean_y = sb_c['adj_mean_mean'].values
    adj_diff_x = sa_c['adj_diff_mean'].values
    adj_diff_y = sb_c['adj_diff_mean'].values

    t1, p1, df1 = paired_ttest(adj_mean_x, adj_mean_y)
    t2, p2, df2 = paired_ttest(adj_diff_x, adj_diff_y)

    out_lines = []
    out_lines.append('Paired comparison for episodes: %d episodes (aligned)' % (len(common)))

    # summary statistics
    out_lines.append('\nSummary statistics (per-episode means across steps):')
    out_lines.append('Metric, no_ude_mean, no_ude_std, ude_mean, ude_std, mean_diff (no_ude - ude)')
    out_lines.append('adj_mean, %.6f, %.6f, %.6f, %.6f, %.6f' % (
        sa_c['adj_mean_mean'].mean(), sa_c['adj_mean_mean'].std(),
        sb_c['adj_mean_mean'].mean(), sb_c['adj_mean_mean'].std(),
        sa_c['adj_mean_mean'].mean() - sb_c['adj_mean_mean'].mean()
    ))
    out_lines.append('adj_diff, %.6f, %.6f, %.6f, %.6f, %.6f' % (
        sa_c['adj_diff_mean'].mean(), sa_c['adj_diff_mean'].std(),
        sb_c['adj_diff_mean'].mean(), sb_c['adj_diff_mean'].std(),
        sa_c['adj_diff_mean'].mean() - sb_c['adj_diff_mean'].mean()
    ))

    # show first 10 episodes as example pairs
    out_lines.append('\nFirst 10 episode pairs (adj_mean):')
    df_pair_mean = pd.DataFrame({'episode': common, 'no_ude': sa_c['adj_mean_mean'].values, 'ude': sb_c['adj_mean_mean'].values})
    out_lines.append(df_pair_mean.head(10).to_string(index=False))

    out_lines.append('\nFirst 10 episode pairs (adj_diff):')
    df_pair_diff = pd.DataFrame({'episode': common, 'no_ude': sa_c['adj_diff_mean'].values, 'ude': sb_c['adj_diff_mean'].values})
    out_lines.append(df_pair_diff.head(10).to_string(index=False))

    out_lines.append('\nStat tests (paired t-test across episodes):')
    out_lines.append('adj_mean: t=%.4f, p=%s, df=%d' % (t1, ('%.4e' % p1) if not np.isnan(p1) else 'nan', df1))
    out_lines.append('adj_diff: t=%.4f, p=%s, df=%d' % (t2, ('%.4e' % p2) if not np.isnan(p2) else 'nan', df2))

    out_text = '\n'.join(out_lines)
    with open('logs/adj_compare.txt', 'w', encoding='utf-8') as f:
        f.write(out_text)

    print('Comparison saved to logs/adj_compare.txt')

if __name__ == '__main__':
    main()
