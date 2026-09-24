"""
评估结果落盘与可视化上报。

职责：
1. 把 evaluate() 返回的统计量规整成 record，并落盘（JSON 明细 + 追加 history.csv 标量 + adp npz）。
2. 可选上报 wandb：标量、直方图、牌型分布表格。

wandb 未安装或未开启 use_wandb 时静默跳过，不影响落盘与训练主流程。
"""
import csv
import json
import os

import numpy as np

from ..env.utils import __PLAY_ROLES__, __WRONG__
from ..model.play_model import NUM_STRATEGY

# history.csv 的固定列顺序
_HISTORY_FIELDS = [
    'tag', 'step', 'num_games',
    'hrl_as_banker_winrate', 'hrl_as_idler_winrate',
    'idler_mean_win_score',
    'hrl_lead_rate',
]


def _dump_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not serializable: {type(o)}')


def build_record(stats, tag, step):
    """把 evaluate() 返回的 stats 规整成统一的 record 结构。"""
    play_dist = {}
    for r in __PLAY_ROLES__:
        seq = stats.get('num_play_type_distribution', {}).get(r, [])
        if len(seq) == 0:
            play_dist[r] = [0] * __WRONG__
        else:
            play_dist[r] = np.asarray(seq, dtype=np.float64).sum(axis=0).astype(int).tolist()

    # 策略导向分布：hrl 队（座位 0/2）每局 7 维计数，跨局求和
    strat_seq = stats.get('hrl_strategy_distribution', [])
    if len(strat_seq) == 0:
        strategy_dist = [0] * NUM_STRATEGY
    else:
        strategy_dist = np.asarray(strat_seq, dtype=np.float64).sum(axis=0).astype(int).tolist()

    # adp 压缩成均值/方差（完整序列单独存 npz）
    adp = stats.get('adp', {})
    adp_summary = {}
    for r in __PLAY_ROLES__:
        arr = np.asarray(adp.get(r, []), dtype=np.float64)
        adp_summary[r] = {
            'mean': float(arr.mean()) if arr.size else 0.0,
            'std': float(arr.std()) if arr.size else 0.0,
            'n': int(arr.size),
        }

    record = {
        'tag': tag,
        'step': step,
        'num_games': int(stats.get('num_games', 0)),
        'hrl_as_banker_winrate': float(stats.get('hrl_banker_winrate', 0.0)),
        'hrl_as_idler_winrate': float(stats.get('hrl_idler_winrate', 0.0)),
        'idler_mean_win_score': float(stats.get('idler_mean_win_score', 0.0)),
        'hrl_level_scores': [float(x) for x in stats.get('hrl_level_scores', [])],
        'num_lead_hrl': [int(x) for x in stats.get('num_lead_hrl', [])],
        'num_lead_total': [int(x) for x in stats.get('num_lead_total', [])],
        'hrl_lead_counts': int(stats.get('hrl_lead_counts', 0)),
        'hrl_lead_rate': float(stats.get('hrl_lead_rate', 0.0)),
        'play_type_distribution': play_dist,
        'strategy_distribution': strategy_dist,
        'bid_win': {r: [int(x) for x in stats.get('num_bid_win', {}).get(r, [])] for r in __PLAY_ROLES__},
        'final_level_info': {r: int(stats.get('final_level_info', {}).get(r, 0)) for r in __PLAY_ROLES__},
        'adp': adp_summary,
    }
    return record


def dump_record(record, save_dir):
    """落盘：明细 JSON、追加 history.csv、adp npz（若有）。"""
    os.makedirs(save_dir, exist_ok=True)

    detail_path = os.path.join(save_dir, 'records', f"{record['tag']}_{record['step']}.json")
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)
    _dump_json(record, detail_path)

    hist_path = os.path.join(save_dir, 'history.csv')
    write_header = not os.path.exists(hist_path)
    with open(hist_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: record[k] for k in _HISTORY_FIELDS})

    return detail_path


def dump_full_stats(stats, save_dir, tag, step):
    """落盘完整原始统计量（含未压缩的 adp、每局分布），供离线复现绘图。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'records', f"{tag}_{step}_full.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _dump_json(stats, path)

    adp = stats.get('adp', {})
    if any(len(adp.get(r, [])) for r in __PLAY_ROLES__):
        npz = os.path.join(save_dir, 'records', f"{tag}_{step}_adp.npz")
        np.savez(npz, **{r: np.asarray(adp.get(r, []), dtype=np.float64) for r in __PLAY_ROLES__})

    # 控牌原始数据：每局 hrl 队领出（回合首出）次数与该局总领出次数，供离线分析
    num_lead_hrl = stats.get('num_lead_hrl', [])
    num_lead_total = stats.get('num_lead_total', [])
    if len(num_lead_hrl):
        npz = os.path.join(save_dir, 'records', f"{tag}_{step}_lead.npz")
        np.savez(npz,
                 hrl=np.asarray(num_lead_hrl, dtype=np.int64),
                 total=np.asarray(num_lead_total, dtype=np.int64))
    return path


def init_wandb(flags, name=None):
    """初始化 wandb；未安装或未开启则返回 None。"""
    if not getattr(flags, 'use_wandb', False):
        return None
    try:
        import wandb
    except ImportError:
        print('[report] wandb 未安装，跳过 wandb 上报（落盘不受影响）')
        return None

    run = wandb.init(
        project=getattr(flags, 'wandb_project', 'tractor-zero'),
        entity=getattr(flags, 'wandb_entity', None) or None,
        name=name or getattr(flags, 'wandb_name', None) or None,
        config={
            k: getattr(flags, k)
            for k in ('batch_size', 'unroll_length', 'learning_rate_banker',
                      'learning_rate_idler', 'learning_rate_bid', 'learning_rate_cover')
            if hasattr(flags, k)
        },
        reinit=True,
    )
    return run


def log_record(run, record, prefix=''):
    """把 record 记录到 wandb：标量 + 直方图 + 牌型表格。"""
    if run is None:
        return
    import wandb

    step = record.get('step', 0)
    key = (lambda k: f'{prefix}/{k}') if prefix else (lambda k: k)

    logs = {
        key('hrl_as_banker_winrate'): record.get('hrl_as_banker_winrate', 0.0),
        key('hrl_as_idler_winrate'): record.get('hrl_as_idler_winrate', 0.0),
        key('idler_mean_win_score'): record.get('idler_mean_win_score', 0.0),
        key('hrl_lead_rate'): record.get('hrl_lead_rate', 0.0),
        key('num_games'): record['num_games'],
    }

    for r in __PLAY_ROLES__:
        a = record.get('adp', {}).get(r)
        if a:
            logs[key(f'adp_mean/{r}')] = a['mean']
            logs[key(f'adp_std/{r}')] = a['std']

    def _hist(values, name):
        if values is None or len(values) == 0:
            return
        # 直接传入原始 values，wandb 内部会自动处理分箱
        logs[key(name)] = wandb.Histogram(values)
        
    _hist(record.get('hrl_level_scores'), 'hrl_level_dist')

    run.log(logs, step=step)

    # 牌型分布表
    columns = ['role'] + [f'type_{i}' for i in range(__WRONG__)]
    data = [[r] + record['play_type_distribution'][r] for r in __PLAY_ROLES__]
    run.log({key('play_type_distribution'): wandb.Table(columns=columns, data=data)}, step=step)

    # 策略导向分布表（hrl 队，座位 0/2）
    strat_columns = ['side'] + [f'g{i+1}' for i in range(NUM_STRATEGY)]
    strat_data = [['hrl'] + record['strategy_distribution']]
    run.log({key('strategy_distribution'): wandb.Table(columns=strat_columns, data=strat_data)}, step=step)


def finish_wandb(run):
    if run is not None:
        run.finish()