"""
论文出图脚本（离线，依赖 matplotlib + scipy，不依赖 wandb）。

读取 report.dump_record 落盘的结果（history.csv 与 records/*.json），生成：
1. 学习曲线（胜率/得分 vs step，EMA 平滑 + 多种子 mean±std 阴影带）
2. 得分分布（小提琴图）
3. 出牌类型分布（堆叠条形图）
4. 显著性检验（bootstrap 95% CI + Welch t 检验 + Mann-Whitney U）

用法示例：
  python -m rlcard.games.tractors.eval.plot_results \
      --results_dir ./log/eval_results --out_dir ./figures --fmt pdf
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps

from ..env.utils import __PLAY_ROLES__, __WRONG__
from ..model.play_model import NUM_STRATEGY, _STRATEGY_NAMES

# 全局出版级样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 110,
    'savefig.bbox': 'tight',
})

_OFFICIAL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# ---------------------------------------------------------------- load
def load_history(history_csv):
    """读取 history.csv -> dict[tag] = list[row]，按 step 排序。"""
    rows = []
    with open(history_csv, newline='') as f:
        for r in csv.DictReader(f):
            r['step'] = int(float(r['step']))
            r['num_games'] = int(r.get('num_games', 0))
            r['idler_mean_win_score'] = float(r.get('idler_mean_win_score', 0.0))
            for k in ('hrl_as_banker_winrate', 'hrl_as_idler_winrate', 'hrl_lead_rate'):
                if r.get(k) not in (None, ''):
                    r[k] = float(r[k])
            rows.append(r)
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r['tag'], []).append(r)
    for tag in by_tag:
        by_tag[tag].sort(key=lambda r: r['step'])
    return by_tag


def load_records(records_dir):
    """读取 records/*.json -> list[dict]（排除 *_full.json / *_adp.npz）。"""
    recs = []
    for path in sorted(glob.glob(os.path.join(records_dir, '*.json'))):
        if path.endswith('_full.json'):
            continue
        with open(path) as f:
            recs.append(json.load(f))
    return recs


def _records_by_tag(records):
    """按 tag 聚合 records -> dict[tag] = list[record]。"""
    by_tag = {}
    for rec in records:
        by_tag.setdefault(rec.get('tag'), []).append(rec)
    return by_tag


def _ema(x, alpha=0.9):
    y = np.zeros_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1 - alpha) * x[i]
    return y


def _aggregate_by_step(rows, metric):
    """同一 tag 下按 step 聚合为 mean/std（多种子时打阴影带）。"""
    steps = sorted(set(r['step'] for r in rows))
    mean, std = [], []
    for s in steps:
        vals = [r[metric] for r in rows if r['step'] == s]
        mean.append(np.mean(vals))
        std.append(np.std(vals))
    return steps, np.array(mean), np.array(std)


# ---------------------------------------------------------------- figures
def plot_learning_curve(history, metrics=None, save_path=None, ema_alpha=0.9):
    """学习曲线：每个 tag 画 EMA 平滑曲线，多种子则加 mean±std 阴影。"""
    if metrics is None:
        metrics = ['hrl_as_banker_winrate', 'hrl_as_idler_winrate', 'idler_mean_win_score']

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, (tag, rows) in enumerate(history.items()):
            steps, mean, std = _aggregate_by_step(rows, metric)
            smoothed = _ema(mean, ema_alpha)
            color = _OFFICIAL_COLORS[i % len(_OFFICIAL_COLORS)]
            ax.plot(steps, smoothed, color=color, label=tag, lw=1.6)
            if std.max() > 1e-9:
                ax.fill_between(steps, smoothed - std, smoothed + std,
                                color=color, alpha=0.2)
        ax.set_xlabel('step')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_score_distribution(records, key='hrl_level_scores', save_path=None):
    """得分/升级分分布：小提琴图，每 tag 一个。"""
    tags, distributions = [], []
    for rec in records:
        data = rec.get(key, [])
        if len(data) == 0:
            continue
        tags.append(f"{rec.get('tag')} (n={len(data)})")
        distributions.append(np.asarray(data, dtype=np.float64))

    if not distributions:
        raise ValueError(f'没有可用于绘图的数据: {key}')

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(distributions) + 2), 4.5))
    ax.violinplot(distributions, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(tags) + 1))
    ax.set_xticklabels(tags, rotation=20, ha='right')
    ax.set_ylabel(key)
    ax.set_title(key)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_play_type_distribution(records, save_path=None):
    """出牌类型分布：堆叠条形图（归一化占比）。"""
    tags, mats = [], []
    for rec in records:
        pd = rec.get('play_type_distribution', {})
        if not pd:
            continue
        mat = np.array([pd[r] for r in __PLAY_ROLES__], dtype=np.float64)  # (role, type)
        if mat.sum() == 0:
            continue
        tags.append(rec.get('tag'))
        mats.append(mat.sum(axis=0))  # 全体角色合计 (type,)

    if not mats:
        raise ValueError('没有 play_type_distribution 数据')

    mat = np.vstack(mats)  # (tag, type)
    mat = mat / mat.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tags))
    bottom = np.zeros(len(tags))
    labels = ['NONE', 'SINGLE', 'PAIR', 'TRACTOR', 'SUSPECT', 'DISCARD'][:__WRONG__]
    for t in range(__WRONG__):
        ax.bar(x, mat[:, t], bottom=bottom, width=0.6, label=labels[t],
               color=_OFFICIAL_COLORS[t % len(_OFFICIAL_COLORS)])
        bottom += mat[:, t]
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha='right')
    ax.set_ylabel('proportion')
    ax.set_title('play type distribution')
    ax.legend(ncol=min(__WRONG__, 3), loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_final_level_info(records, save_path=None):
    """最终级数差：只统计 hrl 队（座位 0/2）所在阵营的级数差，按 tag 聚合为柱状图。

    final_level_info 为终局结算 reward（牌局结束时的级数差），
    'banker/banker_op' 表示 hrl 队充当庄家方时的 +end_score，
    'banker_down/banker_up' 表示 hrl 队充当闲家方时的 +end_score。
    """
    by_tag = _records_by_tag(records)
    tags, b_mean, b_std, i_mean, i_std = [], [], [], [], []

    for tag, recs in by_tag.items():
        b = [rec.get('final_level_info', {}).get('banker', 0) for rec in recs]
        i = [rec.get('final_level_info', {}).get('banker_down', 0) for rec in recs]
        if not b and not i:
            continue
        tags.append(tag)
        b_mean.append(float(np.mean(b)) if b else 0.0)
        b_std.append(float(np.std(b)) if b else 0.0)
        i_mean.append(float(np.mean(i)) if i else 0.0)
        i_std.append(float(np.std(i)) if i else 0.0)

    if not tags:
        raise ValueError('没有 final_level_info 数据')

    x = np.arange(len(tags))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(tags) + 2), 4.5))
    ax.bar(x - width / 2, b_mean, width, yerr=b_std, capsize=3,
           label='banker team', color=_OFFICIAL_COLORS[0])
    ax.bar(x + width / 2, i_mean, width, yerr=i_std, capsize=3,
           label='idler team', color=_OFFICIAL_COLORS[1])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha='right')
    ax.set_ylabel('final level distance')
    ax.set_title('final level distance')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_adp(records, save_path=None):
    """adp（平均决策收益）按角色、按 tag 聚合为柱状图。

    adp 为过程 reward（_get_step_reward）在每局内的均值，角色维度对应 4 个座位。
    """
    by_tag = _records_by_tag(records)
    tags = []
    role_means = {r: [] for r in __PLAY_ROLES__}
    role_stds = {r: [] for r in __PLAY_ROLES__}

    for tag, recs in by_tag.items():
        vals_by_role, any_data = {}, False
        for r in __PLAY_ROLES__:
            vals = [rec.get('adp', {}).get(r, {}).get('mean') for rec in recs]
            vals = [v for v in vals if v is not None]
            vals_by_role[r] = vals
            if vals:
                any_data = True
        if not any_data:
            continue
        tags.append(tag)
        for r in __PLAY_ROLES__:
            v = vals_by_role[r]
            role_means[r].append(float(np.mean(v)) if v else 0.0)
            role_stds[r].append(float(np.std(v)) if v else 0.0)

    if not tags:
        raise ValueError('没有 adp 数据')

    x = np.arange(len(tags))
    n_roles = len(__PLAY_ROLES__)
    width = 0.8 / n_roles
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(tags) + 2), 4.5))
    for i, r in enumerate(__PLAY_ROLES__):
        offset = (i - (n_roles - 1) / 2) * width
        ax.bar(x + offset, role_means[r], width, yerr=role_stds[r], capsize=3,
               label=r, color=_OFFICIAL_COLORS[i % len(_OFFICIAL_COLORS)])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha='right')
    ax.set_ylabel('adp (mean step reward)')
    ax.set_title('average decision payoff (adp)')
    ax.legend(ncol=min(n_roles, 2))
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_idler_mean_win_score(records, save_path=None):
    """闲家场均得分：按 tag 聚合为柱状图，误差线为不同 record 间的标准差。"""
    by_tag = _records_by_tag(records)
    tags, mean, std = [], [], []

    for tag, recs in by_tag.items():
        vals = [rec.get('idler_mean_win_score') for rec in recs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        tags.append(tag)
        mean.append(float(np.mean(vals)))
        std.append(float(np.std(vals)))

    if not tags:
        raise ValueError('没有 idler_mean_win_score 数据')

    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(tags) + 2), 4.5))
    ax.bar(x, mean, 0.6, yerr=std, capsize=3, color=_OFFICIAL_COLORS[2])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha='right')
    ax.set_ylabel('idler mean win score')
    ax.set_title('idler mean win score')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_strategy_distribution(savelog_dir, out_dir, fmt='pdf'):
    """策略分布离线下图：读取 dmc.py 落盘的 strategy_dist/*.npz。

    每个角色生成两张图：
      1. strategy_dist_<role>：分别显示学习策略与事后动作标签的分布；两者无固定一一对应关系。
      2. strategy_sel_ts_<role>：采样频率随训练帧数的时间序列（判断学习演化）。
    """
    dist_dir = os.path.join(savelog_dir, 'strategy_dist')
    n_colors = len(_OFFICIAL_COLORS)

    for role in __PLAY_ROLES__:
        path = os.path.join(dist_dir, '%s.npz' % role)
        if not os.path.exists(path):
            print(f'跳过 {role}: 未找到 {path}')
            continue
        d = np.load(path)
        sel = d['sel']            # [T, 7]
        tgt = d['tgt']            # [T, 7]
        frames = d['frames']      # [T]

        sel_mean = sel.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)

        x = np.arange(NUM_STRATEGY)
        option_names = [f'g{i + 1}' for i in range(NUM_STRATEGY)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        axes[0].bar(x, sel_mean, color=_OFFICIAL_COLORS[0])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(option_names)
        axes[0].set_title(f'{role} learned strategy selection')
        axes[1].bar(x, tgt_mean, color=_OFFICIAL_COLORS[1])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(_STRATEGY_NAMES, rotation=20, ha='right')
        axes[1].set_title(f'{role} post-hoc action labels (diagnostic)')
        for ax in axes:
            ax.set_ylabel('mean frequency over training')
            ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'strategy_dist_{role}.{fmt}'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i in range(NUM_STRATEGY):
            ax.plot(frames, sel[:, i], label=option_names[i], lw=1.4,
                    color=_OFFICIAL_COLORS[i % n_colors])
        ax.set_xlabel('frames')
        ax.set_ylabel('selection freq')
        ax.set_title(f'{role} strategy selection over training')
        ax.legend(ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'strategy_sel_ts_{role}.{fmt}'))
        plt.close(fig)

        print(f'{role}: {len(frames)} 个采样点已出图')

    print(f'策略分布图已输出到: {out_dir}')


# ---------------------------------------------------------------- stats
def bootstrap_ci(data, stat=np.mean, n_resamples=10000, confidence_level=0.95):
    data = np.asarray(data, dtype=np.float64)
    res = sps.bootstrap((data,), stat, n_resamples=n_resamples,
                        confidence_level=confidence_level, vectorized=False)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def compare_distributions(a, b):
    """两组样本的显著性检验。返回可打印的 dict。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = {
        'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'ci_a': bootstrap_ci(a), 'ci_b': bootstrap_ci(b),
    }
    if len(a) >= 2 and len(b) >= 2:
        out['welch_t_p'] = float(sps.ttest_ind(a, b, equal_var=False).pvalue)
        out['mannwhitney_p'] = float(sps.mannwhitneyu(a, b, alternative='two-sided').pvalue)
    return out


def print_significance(records, key='hrl_level_scores'):
    """对每对 tag 做显著性检验并打印。"""
    groups = {rec.get('tag'): np.asarray(rec.get(key, []), dtype=np.float64)
              for rec in records if len(rec.get(key, [])) > 0}
    tags = list(groups.keys())
    print(f'\n== 显著性检验: {key} ==')
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            res = compare_distributions(groups[a], groups[b])
            print(f'{a} vs {b}:')
            print(f"  mean {a}={res['mean_a']:.3f} (CI {res['ci_a'][0]:.3f}~{res['ci_a'][1]:.3f}) "
                  f"vs {b}={res['mean_b']:.3f} (CI {res['ci_b'][0]:.3f}~{res['ci_b'][1]:.3f})")
            if 'welch_t_p' in res:
                print(f"  Welch t p={res['welch_t_p']:.4g}, Mann-Whitney U p={res['mannwhitney_p']:.4g}")


def showplt():
    ap = argparse.ArgumentParser(description='生成论文出图')
    ap.add_argument('--results_dir', default='./log/eval_results',
                    help='report.dump_record 的输出目录（含 history.csv 与 records/）')
    ap.add_argument('--out_dir', default='./figures', help='图形输出目录')
    ap.add_argument('--fmt', default='pdf', choices=['png', 'pdf', 'svg'],
                    help='输出格式')
    ap.add_argument('--ema_alpha', default=0.9, type=float, help='学习曲线 EMA 平滑系数')
    ap.add_argument('--savelog_dir', default=None,
                    help='训练 savelog 目录（含 strategy_dist/*.npz），用于离线下图策略分布')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ext = args.fmt

    hist = load_history(os.path.join(args.results_dir, 'history.csv'))
    records = load_records(os.path.join(args.results_dir, 'records'))

    fig = plot_learning_curve(hist, ema_alpha=args.ema_alpha)
    fig.savefig(os.path.join(args.out_dir, f'learning_curve.{ext}'))
    plt.close(fig)

    recs_by_tag = {}
    for rec in records:
        recs_by_tag.setdefault(rec.get('tag'), []).append(rec)

    if records:
        # 每 tag 取最新一个 checkpoint 的明细做分布图
        latest = [max(v, key=lambda r: r.get('step', 0)) for v in recs_by_tag.values()]

        fig = plot_score_distribution(latest, key='hrl_level_scores')
        fig.savefig(os.path.join(args.out_dir, f'score_distribution.{ext}'))
        plt.close(fig)

        fig = plot_play_type_distribution(latest)
        fig.savefig(os.path.join(args.out_dir, f'play_type_distribution.{ext}'))
        plt.close(fig)

        fig = plot_final_level_info(records)
        fig.savefig(os.path.join(args.out_dir, f'final_level_info.{ext}'))
        plt.close(fig)

        fig = plot_adp(records)
        fig.savefig(os.path.join(args.out_dir, f'adp.{ext}'))
        plt.close(fig)

        fig = plot_idler_mean_win_score(records)
        fig.savefig(os.path.join(args.out_dir, f'idler_mean_win_score.{ext}'))
        plt.close(fig)

        print_significance(records, key='hrl_level_scores')

    if args.savelog_dir:
        plot_strategy_distribution(args.savelog_dir, args.out_dir, ext)

    print(f'图已输出到: {args.out_dir}')


if __name__ == '__main__':
    showplt()
