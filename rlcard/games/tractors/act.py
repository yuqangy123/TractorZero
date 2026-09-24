from rlcard.games.tractors.env.env import Env
# from .env.env import Env
from .env.env_utils import Environment#*
from .env.utils import *
from .model.play_model import compute_strategy_targets
import torch
import traceback
import numpy as np
import os
import time

from rlcard.optimizer.radam import RAdam

def get_batch_play(b_queues, flags, lock):
    b_queue = b_queues
    buffer = []
    # start_t = timer()
    while len(buffer) < flags.batch_size:
        buffer.append(b_queue.get())
    batch = {
        key: torch.stack([m[key] for m in buffer], dim=1)
        for key in ["target_adp", "target_wp", "obs_action_type_x", "obs_action_type_z",
                    "obs_z", "obs_x", "obs_strategy_z", "obs_strategy_x",
                    "strategy_action", "strategy_target"]
    }
    del buffer
    return batch

def get_batch_bid(b_queues, flags, lock):
    b_queue = b_queues
    buffer = []
    # start_t = timer()
    while len(buffer) < flags.batch_size:
        buffer.append(b_queue.get())
    batch = {
        key: torch.stack([m[key] for m in buffer], dim=1)
        for key in ["target_wp", "obs_z", "obs_x"]
    }
    del buffer
    return batch

def get_batch_cover(b_queues, flags, lock):
    b_queue = b_queues
    buffer = []
    # start_t = timer()
    while len(buffer) < flags.batch_size:
        buffer.append(b_queue.get())
    batch = {
        key: torch.stack([m[key] for m in buffer], dim=1)
        for key in ["target_wp", "cover_action_mask",
                    "obs_z", "obs_x"]
    }
    del buffer
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = {'banker':'learning_rate_banker', 'banker_op':'learning_rate_idler', 'banker_down':'learning_rate_idler',\
        'banker_up':'learning_rate_idler', 'bid':'learning_rate_bid', 'cover':'learning_rate_cover'}
    optimizers = {}
    for position, learn_rate in positions.items():
        optimizer = RAdam(
            learner_model.parameters(position),
            lr=getattr(flags, learn_rate),
            eps=flags.epsilon)
        optimizers[position] = optimizer
    return optimizers

def create_env(flags):
    return Env(flags.objective)

def act(actor_index, device, batch_queues, model, banker_win_counter, idler_win_counter, exp_epsilon_shared, flags=None):
    positions = ['banker', 'banker_op', 'banker_up', 'banker_down', 'bid', 'cover']
        
    try:
        T = flags.unroll_length
        print(f'Device {str(device)} Actor {actor_index} started.')

        env = create_env(flags)
        env = Environment(env, device)

        # done_buf = {p: [] for p in positions}
        # episode_return_buf = {p: [] for p in positions}
        
        
        
        obs_z_buf = {p: [] for p in positions}
        obs_x_buf = {p: [] for p in positions}
        obs_action_type_z_buf = {p: [] for p in ['banker', 'banker_op', 'banker_up', 'banker_down']}
        obs_action_type_x_buf = {p: [] for p in ['banker', 'banker_op', 'banker_up', 'banker_down']}
        # 策略头的状态级输入（不含牌型候选），与推理时同一表示
        obs_strategy_z_buf = {p: [] for p in ['banker', 'banker_op', 'banker_up', 'banker_down']}
        obs_strategy_x_buf = {p: [] for p in ['banker', 'banker_op', 'banker_up', 'banker_down']}
        # 必须记录行为策略（含 epsilon 探索），不能由 learner 的当前 argmax 替换。
        strategy_action_buf = {p: [] for p in __PLAY_ROLES__}
                
        # play_action_type_mask_buf = {p: [] for p in positions}
        target_adp_buf = {p: [] for p in positions}
        target_wp_buf = {p: [] for p in positions}
        strategy_target_buf = {p: [] for p in ['banker', 'banker_op', 'banker_up', 'banker_down']}
        
        # bid_return_buf = {"bid": []}
        bid_action_mask = None
        bid_obs_z = None
        bid_obs_x = None
        
        
        cover_public_score_buf = {"cover": []}
        cover_return_buf = {"cover": []}
        cover_action_mask_buf = {"cover": []}
        cover_action_mask = None
        cover_obs_z = None
        cover_obs_x = None
        size = {p: 0 for p in positions}
        
        position, obs, env_output = env.initial(model, device, flags=flags)
        
        while True:
            flags.exp_epsilon = exp_epsilon_shared.value
            while True:
                if position in __PLAY_ROLES__:
                    _ti = time.perf_counter()
                    with torch.no_grad():
                        agent_output = model.play(position, obs['z'], obs['x'], obs['z_batch'], obs['x_batch'], env_output['legal_actions'], env_output['legal_types'], flags=flags)       
                    
                    action_type_index = agent_output['action_type_index']
                    action = agent_output['action']
                    # 策略头的状态级输入（不含牌型候选），与推理时同一表示
                    obs_strategy_z_buf[position].append(obs['z'].detach().cpu())
                    obs_strategy_x_buf[position].append(obs['x'].detach().cpu())
                    strategy_action_buf[position].append(agent_output['strategy'].detach().cpu())
                    pred_action_type = env_output['legal_types'][action_type_index]
                    
                    obs_action_type_z_buf[position].append(obs['z_batch'][action_type_index].detach().cpu())
                    obs_action_type_x_buf[position].append(obs['x_batch'][action_type_index].detach().cpu())
                    
                    obs_z_buf[position].append(obs['z_batch'][action_type_index].detach().cpu())
                    obs_x = torch.cat((torch.from_numpy(env_output['legal_actions'][pred_action_type][action]), obs['x_batch'][action_type_index]), axis=0)
                    obs_x_buf[position].append(obs_x)
                    size[position] += 1
                    
                    action = [matrix2cards(env_output['legal_actions'][pred_action_type][action]), pred_action_type]
                    
                elif position == 'bid':
                    _ti = time.perf_counter()
                    with torch.no_grad():
                        agent_output = model.bid(obs['z_batch'], obs['x_batch'], flags=flags)
                    action_index = agent_output['action'].cpu().detach().item()
                    legal_actions = env_output['legal_actions']
                    action = [legal_actions[action_index][0]]#0不叫，1-4花色，5无主
                    bid_cards = legal_actions[action_index][1]
                    bid_obs_z = obs['z']
                    bid_obs_x = torch.cat((torch.from_numpy(cards2matrix(bid_cards)), obs['x']), dim=0).to(torch.int8)
                    
                elif position == 'cover':
                    _ti = time.perf_counter()
                    with torch.no_grad():
                        agent_output = model.cover(obs['z_batch'], obs['x_batch'], flags=flags)
                    logits = agent_output['action'].cpu().detach()
                    legal_actions = env_output['legal_actions']
                    _legal_actions = legal_actions.flatten()
                    f_action = _legal_actions * logits[0]
                    values, indices = torch.topk(f_action, k=8)
                    action = torch.zeros_like(_legal_actions)
                    action[indices] = 1
                    action = action.reshape(legal_actions.shape)
                    cover_action_mask = action  # 被埋的8张牌 one-hot [2,4,15]，作为回归目标掩码
                    action = [matrix2cards(action)]
                    cover_obs_z = obs['z']
                    cover_obs_x = obs['x']
                    
                    
                else:
                    raise ValueError(f"unkown position:{position}")
                
                position, obs, env_output = env.step(action)
                
                if 'step_reward' in env_output:
                    step_reward = env_output['step_reward']
                    # 本墩元数据已在环境 reset 之前冻结（env_utils），据此按动作特征生成互斥策略标签
                    step_meta = env_output.get('step_meta')
                    for rule in __PLAY_ROLES__:
                        target_adp_buf[rule].append(step_reward[rule])
                        strategy_target_buf[rule].append(torch.tensor(
                            compute_strategy_targets(step_meta, rule), device='cpu'))
                    
                #roundend后step一次切换状态
                if env_output['stage'] == 'roundend':
                    position, obs, env_output = env.step(action)
                    
                if 'game_reward' in env_output:
                    game_reward = env_output['game_reward']

                    # 先登记本局 bid/cover 样本，再统一补 reward：
                    # 若顺序颠倒，补 reward 时 size 尚未计入本局样本，diff 会少算 1，
                    # 导致 bid/cover 观测配到下一局的 game_reward（整体错位一局）。
                    obs_z_buf['bid'].append(bid_obs_z)
                    obs_x_buf['bid'].append(bid_obs_x)
                    size['bid'] += 1

                    # cover_public_score_buf['cover'].append(game_reward['cover_public_score'])
                    # cover_return_buf['cover'].append(game_reward['cover'])
                    cover_action_mask_buf['cover'].append(cover_action_mask)
                    obs_z_buf['cover'].append(cover_obs_z)
                    obs_x_buf['cover'].append(cover_obs_x)
                    size['cover'] += 1

                    for p in positions:
                        diff = size[p] - len(target_wp_buf[p])
                        if diff > 0:
                            # done_buf[p].extend([False for _ in range(diff - 1)])
                            # done_buf[p].append(True)
                            wp_return = game_reward[p]
                            # 出牌角色的终局 reward 为级数差 ±{1,2,3}，统一除以 3 归一化到 [-1,1]，
                            # 同时作为上层 Q(s, strategy) 的无折扣终局 Monte Carlo 回报。
                            #放在组装点归一化的原因：评估逻辑依赖`game_reward` 的 原始级数差 ：`final_level_info` 在 report.py:81 用`int(...)` 取整、 plot_results.py 直接当级数差画图。若在源头`/3` ，会被`int()` 截断成 0，破坏评估结果。
                            if p in __PLAY_ROLES__: wp_return = wp_return / 3.0
                            target_wp_buf[p].extend([wp_return for _ in range(diff)])
                    
                    
                    #统计双方胜率
                    if game_reward['banker'] > 0: 
                        with banker_win_counter.get_lock():
                            banker_win_counter.value += 1
                    else:
                        with idler_win_counter.get_lock():
                            idler_win_counter.value += 1
                    break
            
            fit_buff = False
            for p in __PLAY_ROLES__:
                if size[p] > T:
                    fit_buff = True
                    batch_queues[p].put({
                        # "done": torch.stack(
                        #     [torch.tensor(ndarr, device="cpu") for ndarr in done_buf[rule][:T]]),
                        "target_adp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_adp_buf[p][:T]]),
                        "target_wp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_wp_buf[p][:T]]),
                        "obs_action_type_x": torch.stack(obs_action_type_x_buf[p][:T]),
                        "obs_action_type_z": torch.stack(obs_action_type_z_buf[p][:T]),
                        # "play_action_type_mask": torch.stack(
                        #     [torch.tensor(ndarr, device="cpu") for ndarr in play_action_type_mask_buf[p][:T]]),                        
                        "obs_z": torch.stack(
                            obs_z_buf[p][:T]),
                        "obs_x": torch.stack(
                            obs_x_buf[p][:T]),
                        "obs_strategy_z": torch.stack(obs_strategy_z_buf[p][:T]),
                        "obs_strategy_x": torch.stack(obs_strategy_x_buf[p][:T]),
                        "strategy_action": torch.stack(strategy_action_buf[p][:T]),
                        "strategy_target": torch.stack(strategy_target_buf[p][:T]),
                    })
                    
                    target_adp_buf[p] = target_adp_buf[p][T:]
                    target_wp_buf[p] = target_wp_buf[p][T:]
                    obs_action_type_x_buf[p] = obs_action_type_x_buf[p][T:]
                    obs_action_type_z_buf[p] = obs_action_type_z_buf[p][T:]
                    # play_action_type_mask_buf[p] = play_action_type_mask_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    obs_x_buf[p] = obs_x_buf[p][T:]
                    obs_strategy_z_buf[p] = obs_strategy_z_buf[p][T:]
                    obs_strategy_x_buf[p] = obs_strategy_x_buf[p][T:]
                    strategy_action_buf[p] = strategy_action_buf[p][T:]
                    strategy_target_buf[p] = strategy_target_buf[p][T:]
                    size[p] -= T
            
            if fit_buff:        
                # 定期释放 CUDA 缓存并打印显存占用（区分缓存碎片化 vs 真实泄露）
                if flags.training_device != 'cpu' and torch.cuda.is_available():
                    _dev = torch.device('cuda:' + str(flags.training_device))
                    _alloc = torch.cuda.memory_allocated(_dev)
                    _resvd = torch.cuda.memory_reserved(_dev)
                    
                    # print(f'act{actor_index} pid:{os.getpid() }, _alloc: {_alloc/1e9:.2f}GB,  _resvd: {_resvd/1e9:.2}GB')
                                                
                    #如果_resvd 大于 _alloc 3G，说明有缓存碎片化问题
                    if _resvd > 3 * 1e9:
                        print(f'actor{actor_index} Warning: _resvd is {_resvd}, possible memory leak')
                        torch.cuda.empty_cache()
                    
            for p in ['cover']:
                if size[p] > T:
                    batch_queues[p].put({
                        "target_wp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_wp_buf[p][:T]]),
                        "cover_action_mask": torch.stack(cover_action_mask_buf[p][:T]),
                        # "cover_public_score": torch.stack(
                        #     [torch.tensor(ndarr, device="cpu") for ndarr in cover_public_score_buf[p][:T]]),
                        # "cover_return": torch.stack(
                        #     [torch.tensor(ndarr, device="cpu") for ndarr in cover_return_buf[p][:T]]),
                        "obs_z": 
                            torch.stack(obs_z_buf[p][:T]),
                        "obs_x": 
                            torch.stack(obs_x_buf[p][:T]),
                    })
                    
                    target_wp_buf[p] = target_wp_buf[p][T:]
                    cover_action_mask_buf[p] = cover_action_mask_buf[p][T:]
                    # cover_public_score_buf[p] = cover_public_score_buf[p][T:]
                    # cover_return_buf[p] = cover_return_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    obs_x_buf[p] = obs_x_buf[p][T:]
                    size[p] -= T
                    
            for p in ['bid']:
                if size[p] > T:
                    batch_queues[p].put({
                        # "bid_return": torch.stack(
                        #     [torch.tensor(ndarr, device="cpu") for ndarr in bid_return_buf[p][:T]]),
                        "target_wp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_wp_buf[p][:T]]),
                        "obs_z": 
                            torch.stack(obs_z_buf[p][:T]),
                        "obs_x": 
                            torch.stack(obs_x_buf[p][:T]),
                    })
                    target_wp_buf[p] = target_wp_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    obs_x_buf[p] = obs_x_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    except Exception as e:
        print('Exception in worker process %i', actor_index)
        traceback.print_exc()        
        raise e    
    print('act over')
