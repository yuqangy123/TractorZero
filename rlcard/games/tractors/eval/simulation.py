import builtins
import multiprocessing as mp
import os.path
import pickle
import copy, time
from .env_eval import EnvironmentEval
from ..env.env import Env
from ..env.utils import *
from ..model.play_model import NUM_STRATEGY
import torch


output_to_file = False
output_list = []


# def print(*args, **kwargs):
#     builtins.print(*args, **kwargs)
#     if output_to_file:
#         end = "\n"
#         if kwargs.get("end") is not None:
#             end = kwargs.get("end")
#         output_list.append(" ".join(args) + end)


# 庄家方（庄家 + 对家）与闲家方（下家 + 上家）
__BANKER_TEAM__ = ['banker', 'banker_op']
__IDLER_TEAM__ = ['banker_down', 'banker_up']


def _torch_device(device):
    return 'cpu' if device == 'cpu' else 'cuda:' + str(device)


# 角色 -> 座位偏移：座位 = (本局庄家座位 + 偏移) % 4。
# banker/banker_down/banker_op/banker_up 依次是庄家及其下家、对家、上家。
_ROLE_TO_SEAT_OFFSET = {'banker': 0, 'banker_down': 1, 'banker_op': 2, 'banker_up': 3}
# 座位 0/2 固定为 eval 模型方（互为对家），座位 1/3 固定为对手方。
__EVAL_SEATS__ = (0, 2)


def _load_seat_agents(eval_models, opp_model, device):
    """座位 0/2 固定使用 eval 模型（HRLAgent，按当局角色切换对应角色模型）；
    座位 1/3 固定使用对手模型（RandomAgent/RuleAgent，与角色无关）。

    HRLAgent 推理无状态，同角色模型只加载一个实例，两个 eval 座位共享。
    返回 (eval_agents, opp_agents)，均为 {role: agent}。
    """
    from ..agents.hrl_agent import HRLAgent
    eval_agents = HRLAgent(device=device)
    
    for role in list(__PLAY_ROLES__) + ['bid', 'cover']:
        path = eval_models[role]
        if isinstance(path, list):
            path = path[0]  # 简化：多模型集成时仅取首个（ensemble 未实现）
        
        eval_agents.loadModel(role, path, _torch_device(device))
        

    opp_agents = None
    if opp_model['banker'] == 'random':
        from ..agents.random_agent import RandomAgent
        opp_agents = RandomAgent()
    else:
        from ..agents.rule_agent import RuleAgent
        opp_agents = RuleAgent()
            
    return eval_agents, opp_agents




def _split_eval_deals(eval_data, n):
    """把 {level: [deal,...]} 按进程数 n 等分：逐 level 交错切片，保证每份覆盖各级别发牌。"""
    if not eval_data or n <= 1:
        return [eval_data] * n
    splits = [{} for _ in range(n)]
    for level, deals in eval_data.items():
        for i in range(n):
            splits[i][level] = deals[i::n]
    return splits


def mp_simulate(eval_models, opp_model, device, q, flags, eval_deals=None):
    game = Env()
    if hasattr(flags, 'print_game_log'):
        game.print_game_log(True)

    # 注入父进程切分好的固定发牌数据（每个进程消费各自的一份，不再重复同一份）
    if eval_deals:
        game._env.set_eval_deals(eval_deals)

    env = EnvironmentEval(game, device)

    # 座位 0/2 -> eval 模型；座位 1/3 -> 对手模型。
    eval_agents, opp_agents = _load_seat_agents(eval_models, opp_model, device)
    
    #统计数据
    adp = {rule:[] for rule in __PLAY_ROLES__}#只统计 hrl 座位角色
    total_win_score = {rule:0 for rule in __PLAY_ROLES__}#hrl 队当闲家时的局分（只统计 hrl）
    hrl_level_scores = []#hrl 队（座位 0/2）每局级差分布
    num_play_type_distribution = {rule:[] for rule in __PLAY_ROLES__}#每局出牌类型分布（只统计 hrl）
    hrl_strategy_distribution = []#hrl 队（座位 0/2）每局策略导向分布（7 维计数）
    num_bid_win = {rule:[] for rule in __PLAY_ROLES__}#每局叫牌胜利情况（只统计 hrl 叫主方）
    final_level_info = {rule:0 for rule in __PLAY_ROLES__}#最终局的级数
    # 控牌统计：每局 hrl 队领出（回合首出）次数与该局总领出次数
    num_lead_hrl = []
    num_lead_total = []
    # 座位固定视角：hrl 队（座位 0/2）当局充当庄家方/闲家方的局数与其胜局数
    games_as_banker = 0
    wins_as_banker = 0
    games_as_idler = 0
    wins_as_idler = 0
    

    def start_game():
        position, obs, env_output = env.initial(flags=flags)
        infosets = env._get_infosets()
        play_type_distribution = {rule:[0 for _ in range(__WRONG__)] for rule in __PLAY_ROLES__}
        strategy_distribution = {rule:[0 for _ in range(NUM_STRATEGY)] for rule in __PLAY_ROLES__}
        # 每局各角色领出（回合首出）次数
        lead_distribution = {rule:0 for rule in __PLAY_ROLES__}
        # 回合重置后遇到的第一个出牌角色即本回合领出者
        pending_lead = True
        
        while True:
            if position in __PLAY_ROLES__:
                if pending_lead:
                    lead_distribution[position] += 1
                    pending_lead = False
                # 座位固定分配：角色 -> 绝对座位的映射依赖本局庄家座位。
                seat = (game._env.getBanker() + _ROLE_TO_SEAT_OFFSET[position]) % 4
                hrl_play = seat in __EVAL_SEATS__ 
                use_agent = eval_agents if hrl_play else opp_agents
                with torch.no_grad():
                    agent_output = use_agent.play(position, obs['z'], obs['x'], obs['z_batch'], obs['x_batch'], env_output['legal_actions'], env_output['legal_types'], infosets, flags=flags)       
                
                action_type_index = agent_output['action_type_index']
                action = agent_output['action']
                pred_action_type = env_output['legal_types'][action_type_index]
                action = [matrix2cards(env_output['legal_actions'][pred_action_type][action]), pred_action_type]
                if hrl_play:
                    play_type_distribution[position][pred_action_type] += 1
                    _strategy = agent_output.get('strategy')
                    if _strategy is not None:
                        idx = torch.argmax(_strategy).item()
                        strategy_distribution[position][idx] += 1
                
            elif position == 'bid':
                bid_seat = game._env.getPlayerPosition()
                hrl_play = bid_seat in __EVAL_SEATS__ 
                use_agent = eval_agents if hrl_play else opp_agents
                with torch.no_grad():
                    agent_output = use_agent.bid(obs['z_batch'], obs['x_batch'], infosets, flags=flags)
                action_index = agent_output['action'].cpu().detach().item()
                legal_actions = env_output['legal_actions']
                action = [legal_actions[action_index][0]]#0不叫，1-4花色，5无主
                
            elif position == 'cover':
                cover_seat = game._env.getBanker()
                hrl_play = cover_seat in __EVAL_SEATS__ 
                use_agent = eval_agents if hrl_play else opp_agents
                with torch.no_grad():
                    agent_output = use_agent.cover(obs['z_batch'], obs['x_batch'], infosets, flags=flags)
                logits = agent_output['action'].cpu().detach()
                legal_actions = env_output['legal_actions']
                _legal_actions = legal_actions.flatten()
                f_action = _legal_actions * logits[0]
                _, indices = torch.topk(f_action, k=8)
                action = torch.zeros_like(_legal_actions)
                action[indices] = 1
                action = action.reshape(legal_actions.shape)
                action = [matrix2cards(action)]
               
            else:
                raise ValueError(f"unkown position:{position}")
            
            position, obs, env_output = env.step(action)
            infosets = env._get_infosets()
                        
            #统计单步数据
            if 'step_reward' in env_output:
                step_reward = env_output['step_reward']
                # 本墩所属牌的庄家：done 时银行家可能已轮转，用上局庄家
                if env_output.get('done'):
                    banker_seat = game._env.getLastBanker()
                else:
                    banker_seat = game._env.getBanker()
                for r in __PLAY_ROLES__:
                    if (banker_seat + _ROLE_TO_SEAT_OFFSET[r]) % 4 in __EVAL_SEATS__:
                        adp[r].append(step_reward[r])
                
            #roundend后step一次切换状态
            if env_output['stage'] == 'roundend':
                position, obs, env_output = env.step(action)
                pending_lead = True  # 本回合收尾，下一回合/下一局的首个出牌即为领出
                
            #统计该局数据
            if 'game_reward' in env_output:
                game_score = env_output['game_score']
                game_reward = env_output['game_reward']
                banker_seat = game._env.getLastBanker()
                # hrl 队 = 座位 0/2；取 hrl 座位角色的级差（同队两座位级差相同）
                seat_to_role = {(banker_seat + off) % 4: role
                                for role, off in _ROLE_TO_SEAT_OFFSET.items()}
                hrl_level_scores.append(game_reward[seat_to_role[0]])

                # 座位固定视角：本局庄家座位决定 hrl 队（座位 0/2）充当庄家方还是闲家方。
                hrl_as_banker = banker_seat in __EVAL_SEATS__
                # 局分只统计 hrl 队当闲家时的得分（game_score 即闲家方得分）
                if not hrl_as_banker:
                    total_win_score['banker_down'] += game_score
                    total_win_score['banker_up'] += game_score
                banker_team_win = game_reward['banker'] > 0
                hrl_win = banker_team_win if hrl_as_banker else (not banker_team_win)
                nonlocal games_as_banker, wins_as_banker, games_as_idler, wins_as_idler
                if hrl_as_banker:
                    games_as_banker += 1
                    wins_as_banker += int(hrl_win)
                else:
                    games_as_idler += 1
                    wins_as_idler += int(hrl_win)

                # 最终局级数差：先清零（避免 hist 残留上局 hrl 扮演过的角色），只记录本局 hrl 座位角色
                for r in __PLAY_ROLES__:
                    final_level_info[r] = 0
                for r in __PLAY_ROLES__:
                    if (banker_seat + _ROLE_TO_SEAT_OFFSET[r]) % 4 in __EVAL_SEATS__:
                        final_level_info[r] = game_reward[r]

                # 控牌：把按角色累计的领出次数折算到座位，仅统计 hrl 队（座位 0/2）
                num_lead_total.append(sum(lead_distribution.values()))
                num_lead_hrl.append(sum(cnt for r, cnt in lead_distribution.items()
                                        if (banker_seat + _ROLE_TO_SEAT_OFFSET[r]) % 4 in __EVAL_SEATS__))
                # 策略导向：把按角色累计的 7 维策略计数折算到座位，仅统计 hrl 队
                hrl_strategy = [0 for _ in range(NUM_STRATEGY)]
                for r in __PLAY_ROLES__:
                    if (banker_seat + _ROLE_TO_SEAT_OFFSET[r]) % 4 in __EVAL_SEATS__:
                        for i in range(NUM_STRATEGY):
                            hrl_strategy[i] += strategy_distribution[r][i]
                hrl_strategy_distribution.append(hrl_strategy)
                
                #todo
                if flags.log_print:
                    print(f'hrlwin?:{hrl_win}({'banker' if hrl_as_banker else 'idler'}), {'banker' if game_reward['banker'] > 0 else 'idler'} win, level: {infosets.level}')
                
                bid_rule = env._get_last_bid_rule()
                bid_seat = (banker_seat + _ROLE_TO_SEAT_OFFSET[bid_rule]) % 4
                # 只统计 hrl 座位叫主方的叫牌胜负
                if bid_seat in __EVAL_SEATS__:
                    bid_mult = 1 if (bid_rule == 'banker' or bid_rule == 'banker_op') else -1
                    num_bid_win[bid_rule].append(bid_mult*(1 if game_reward['banker'] > 0 else -1))
                    
                
                for r in __PLAY_ROLES__:
                    # 出牌类型分布同样只统计 hrl 座位角色
                    if (banker_seat + _ROLE_TO_SEAT_OFFSET[r]) % 4 in __EVAL_SEATS__:
                        num_play_type_distribution[r].append(play_type_distribution[r])
                    play_type_distribution[r] = [0 for _ in range(__WRONG__)]
                    strategy_distribution[r] = [0 for _ in range(NUM_STRATEGY)]
                    lead_distribution[r] = 0
            
            if env_output['stage'] == 'finalend':
                break

    start_game()

    torch.cuda.empty_cache()
    
    q.put((
        adp,
        total_win_score,
        hrl_level_scores,
        num_play_type_distribution,
        hrl_strategy_distribution,
        num_bid_win,
        final_level_info,
        num_lead_hrl,
        num_lead_total,
        games_as_banker,
        wins_as_banker,
        games_as_idler,
        wins_as_idler,
    ))
    
  
#评估模型
def evaluate(eval_models, opp_model, device, flags, tag='eval', step=0, eval_deals_list=None):    
    ctx = mp.get_context('spawn')
    # 用 Queue 而非 SimpleQueue：SimpleQueue 是有限容量 pipe，多个进程大数据 put 会在 join 前阻塞
    q = ctx.Queue()
    processes = []
    
    # 并行跑多个完整 series（2->A），聚合小局统计以收窄 winrate 置信区间。
    # 分批启动：同时运行的进程数受 eval_concurrency 限制，避免评估进程把训练占用的 GPU 显存打爆。
    n_series = int(getattr(flags, 'eval_series', 4))
    concurrency = max(1, int(getattr(flags, 'eval_concurrency', 1)))
    print(f'共需评估{n_series}轮, 开{n_series}个进程，并同时进行{concurrency}个任务')

    adp = {r: [] for r in __PLAY_ROLES__}
    total_win_score = {r: 0 for r in __PLAY_ROLES__}
    hrl_level_scores = []
    num_play_type_distribution = {r: [] for r in __PLAY_ROLES__}
    hrl_strategy_distribution = []
    num_bid_win = {r: [] for r in __PLAY_ROLES__}
    final_level_info = {r: 0 for r in __PLAY_ROLES__}
    num_lead_hrl = []
    num_lead_total = []
    # 座位固定视角：hrl 队（座位 0/2）充当庄家方/闲家方的局数与胜局数
    games_as_banker = 0
    wins_as_banker = 0
    games_as_idler = 0
    wins_as_idler = 0

    def _accumulate(item):
        (adp_i, total_win_score_i, hrl_level_scores_i,
         num_play_type_distribution_i, hrl_strategy_distribution_i,
         num_bid_win_i, final_level_info_i, num_lead_hrl_i, num_lead_total_i,
         games_as_banker_i, wins_as_banker_i, games_as_idler_i, wins_as_idler_i) = item
        nonlocal games_as_banker, wins_as_banker, games_as_idler, wins_as_idler
        for r in __PLAY_ROLES__:
            adp[r].extend(adp_i[r])
            total_win_score[r] += total_win_score_i[r]
            num_play_type_distribution[r].extend(num_play_type_distribution_i[r])
            num_bid_win[r].extend(num_bid_win_i[r])
            final_level_info[r] = final_level_info_i[r]  # 保留最后一次的最终局级数差
        hrl_level_scores.extend(hrl_level_scores_i)
        hrl_strategy_distribution.extend(hrl_strategy_distribution_i)
        num_lead_hrl.extend(num_lead_hrl_i)
        num_lead_total.extend(num_lead_total_i)
        games_as_banker += games_as_banker_i
        wins_as_banker += wins_as_banker_i
        games_as_idler += games_as_idler_i
        wins_as_idler += wins_as_idler_i

    started = 0
    while started < n_series:
        batch = min(concurrency, n_series - started)
        print(f'开始评估第{started}轮，同时进行{batch}个任务')
        for _ in range(batch):
            chunk = eval_deals_list[started] if eval_deals_list else None
            p = ctx.Process(
                    target=mp_simulate,
                    args=(eval_models, opp_model, device, q, flags, chunk))
            p.start()
            processes.append(p)
            
        for p in processes[len(processes) - batch:]:
            p.join()
        # 关键：每批 join 后立即取走本批结果，防止队列积压。
        # 否则父进程攒到最后才 get，多个进程的大数据量会把底层 pipe 写满，
        # 进程退出时 feeder/flush 仍会阻塞（SimpleQueue/Queue 都受 pipe 容量限制）。
        for _ in range(batch):
            _accumulate(q.get(timeout=600))
        print(f'结束评估第{started}轮')
        started += batch
    
    print(f"结束评估模型:{tag}")

    #hrl 队总对局数
    num_games = games_as_banker + games_as_idler

    #hrl 队当闲家时的场均局分（只统计 hrl 当闲家的局）
    idler_mean_win_score = (total_win_score['banker_down'] / games_as_idler
                            if games_as_idler else 0.0)

    # 座位固定视角：hrl 队（座位 0/2）充当庄家方 / 闲家方时的胜率
    hrl_banker_winrate = wins_as_banker / games_as_banker if games_as_banker else 0.0
    hrl_idler_winrate = wins_as_idler / games_as_idler if games_as_idler else 0.0

    # 控牌率：hrl 队（座位 0/2）领出（回合首出）占比，仅统计 hrl 模型
    hrl_lead_counts = sum(num_lead_hrl)
    total_lead_counts = sum(num_lead_total)
    hrl_lead_rate = hrl_lead_counts / total_lead_counts if total_lead_counts else 0.0

    stats = {
        'adp': adp,
        'num_games': int(num_games),
        'idler_mean_win_score': idler_mean_win_score,
        'hrl_level_scores': [float(x) for x in hrl_level_scores],
        'num_play_type_distribution': num_play_type_distribution,
        'hrl_strategy_distribution': [list(x) for x in hrl_strategy_distribution],
        'num_bid_win': num_bid_win,
        'final_level_info': final_level_info,
        # hrl 队（座位 0/2）按当局身份统计的胜负
        'hrl_games_as_banker': int(games_as_banker),
        'hrl_wins_as_banker': int(wins_as_banker),
        'hrl_banker_winrate': float(hrl_banker_winrate),
        'hrl_games_as_idler': int(games_as_idler),
        'hrl_wins_as_idler': int(wins_as_idler),
        'hrl_idler_winrate': float(hrl_idler_winrate),
        'num_lead_hrl': [int(x) for x in num_lead_hrl],
        'num_lead_total': [int(x) for x in num_lead_total],
        'hrl_lead_counts': int(hrl_lead_counts),
        'hrl_lead_rate': float(hrl_lead_rate),
    }

    # 结果落盘（供离线绘图复现）
    from . import report
    save_dir = os.path.join(getattr(flags, 'savelog', './log'), 'eval_results')
    record = report.build_record(stats, tag=tag, step=step)
    report.dump_record(record, save_dir)
    report.dump_full_stats(stats, save_dir, tag=tag, step=step)
    print('数据已落地')
    return stats
    

def evaluateTrainingModel(modelist, eval_frame, writer, flags):
    device = getattr(flags, 'evaluate_device', 'cpu')
    from . import report

    if not flags.log_print:
        run = report.init_wandb(flags)

    eval_model = modelist['eval_model']

    # 固定发牌数据在父进程加载一次，按 series 数量等分后分发给各评估进程，
    # 保证每个 series 用不同的发牌（而不是每个进程重复同一份）。
    eval_deals_list = None
    eval_data_path = getattr(flags, 'eval_data', None)
    if eval_data_path:
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f'eval_data not found: {eval_data_path}')
        with open(eval_data_path, 'rb') as f:
            eval_deals = pickle.load(f)
        n_series = max(1, int(getattr(flags, 'eval_series', 4)))
        eval_deals_list = _split_eval_deals(eval_deals, n_series)

    # 座位固定对局：座位 0/2 恒为 eval 模型，座位 1/3 恒为对手；与 random/rule/历史最优对战。
    for opp_model_name, opp_model in modelist['opp_model'].items():
        tag = f'hrl_vs_{opp_model_name}'
        print(f"开始评估模型:{tag}")
        
        stats = evaluate(eval_model, opp_model, device, flags, tag=tag, step=eval_frame,
                         eval_deals_list=eval_deals_list)
        
        record = report.build_record(stats, tag=tag, step=eval_frame)

        # hrl 队充当庄家方 / 闲家方时的胜率（座位固定，身份随叫庄结果而定）。
        hrl_banker_winrate = record['hrl_as_banker_winrate']
        hrl_idler_winrate = record['hrl_as_idler_winrate']

        if not flags.log_print:
            writer.add_scalar(f'winrate/{tag}/hrl_as_banker', hrl_banker_winrate, eval_frame)
            writer.add_scalar(f'winrate/{tag}/hrl_as_idler', hrl_idler_winrate, eval_frame)

            report.log_record(run, record, prefix=tag)

        print(f'[{tag}] hrl_as_banker_winrate={hrl_banker_winrate:.4f}, '
              f'hrl_as_idler_winrate={hrl_idler_winrate:.4f}, '
              f'idler_mean_win_score={record["idler_mean_win_score"]:.3f}, '
              f'games={record["num_games"]}, frame={eval_frame}')

    if not flags.log_print:
        report.finish_wandb(run)
                
        
