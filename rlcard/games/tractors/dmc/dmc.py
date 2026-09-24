import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from rlcard.games.tractors.env.utils import *

from ..model.models import Model
from ..model.play_model import NUM_STRATEGY
from rlcard.games.tractors.act import *
from torch.utils.tensorboard import SummaryWriter

__timer = timeit.default_timer

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def compute_loss_(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2)
    return loss


def learn_bid(position, actor_models, model, batch, optimizer, scaler, flags, lock):
    """Performs a learning (optimization) step."""
    # print("Learn for ", position)
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
        device_type = 'cuda'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
    
    obs_x = torch.flatten(batch["obs_x"], 0, 1).float().to(device)
    obs_z = torch.flatten(batch['obs_z'], 0, 1).float().to(device)
    target_wp = torch.flatten(batch['target_wp'].to(device), 0, 1)#得分
        
    with lock:
        model.to(device)
        optimizer.zero_grad()
        def computeLoss():
            output = model.forward(obs_z, obs_x, return_value=True)        
            values = output['values'][0]  # BidModel.forward 返回 values=(output,) 单元素元组
            bid_target = (target_wp > 0).float().unsqueeze(-1)
            loss = compute_loss(values, bid_target)
            return loss
        
        if scaler:
            with torch.amp.autocast(device_type=device_type):
                loss = computeLoss()
        else:
            loss = computeLoss()
        
        stats = {
            'loss_' + position: loss.item(),
        }
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()            
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
            optimizer.step()
        
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def learn_cover(position, actor_models, model, batch, optimizer, scaler, flags, lock):
    """Performs a learning (optimization) step."""
    # print("Learn for ", position)
    
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
        device_type = 'cuda'
    else:
        device = torch.device('cpu')        
        device_type = 'cpu'

    obs_x = torch.flatten(batch["obs_x"], 0, 1).float().to(device)
    obs_z = torch.flatten(batch['obs_z'], 0, 1).float().to(device)
    target_wp = torch.flatten(batch['target_wp'].to(device), 0, 1)#得分
    cover_action_mask = torch.flatten(batch['cover_action_mask'].to(device), 0, 1)
    
    with lock:
        model.to(device)
        optimizer.zero_grad()
        
        def computeLoss():
            output = model.forward(obs_z, obs_x, return_value=True)
            cover_cards_logits = output['values'][0]  # (output,) -> tensor [N, 120]

            # 只对被埋的 8 张牌位用 game_reward['cover'](±1) 作为回归目标，其余牌位自蒸馏（无梯度）。
            # CoverModel 输出是 sigmoid 后的 [0,1] 概率，target 必须同步缩放到 [0,1]：
            # 直接用 ±1 做回归，输局（-1）永远不可达，loss 恒定 ≥1，训练信号失效。
            mask = cover_action_mask.reshape(cover_action_mask.shape[0], -1).float()  # [N, 120]
            target_logits = cover_cards_logits.detach().clone()
            cover_target = (target_wp > 0).float().unsqueeze(-1)
            
            target_logits = torch.where(mask == 1, cover_target.unsqueeze(-1), target_logits)
            loss = compute_loss(cover_cards_logits, target_logits)
            return loss
        
        if scaler:
            with torch.amp.autocast(device_type=device_type):
                loss = computeLoss()
        else:
            loss = computeLoss()

        stats = {
            'loss_' + position: loss.item(),
        }
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()            
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
            optimizer.step()
        
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats
    
def learn_play(position, actor_models, model, batch, optimizer, scaler, flags, lock):
    """DMC: 上层学习 Q(s, g)，下层学习采样策略 g 条件下的牌型/动作价值。"""
    # print("Learn for ", position)
    
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
        device_type = "cuda"
    else:
        device = torch.device('cpu')
        device_type = "cpu" 
    
    obs_x = torch.flatten(batch["obs_x"], 0, 1).float().to(device)
    obs_z = torch.flatten(batch['obs_z'], 0, 1).float().to(device)
    
    obs_action_type_x = torch.flatten(batch["obs_action_type_x"], 0, 1).float().to(device)
    obs_action_type_z = torch.flatten(batch['obs_action_type_z'], 0, 1).float().to(device)
        
    target_adp = torch.flatten(batch['target_adp'].to(device), 0, 1)#得分
    target_wp = torch.flatten(batch['target_wp'].to(device), 0, 1)
    # 策略头使用状态级输入（不含牌型候选），与推理时同一表示
    obs_strategy_z = torch.flatten(batch['obs_strategy_z'].to(device), 0, 1).float()
    obs_strategy_x = torch.flatten(batch['obs_strategy_x'].to(device), 0, 1).float()
    strategy_action = torch.flatten(batch['strategy_action'].to(device), 0, 1).float()
    target_strategy = torch.flatten(batch['strategy_target'].to(device), 0, 1).float()
    
    with lock:
        # # 检查并移动模型和优化器状态到正确设备
        # current_device = next(model.parameters()).device
        # print('current_device: ', current_device)
        # print('train device: ', device)
        # if current_device != device:
        #     model.to(device)
        #     # 优化器状态也移动到同一设备
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.to(device)
        #                 print('state[k] to: ', device)
        
        #learn_action_play(self, position, z, x, action_type, legal_actions)
        # win_rate, win, lose = model.learn_play(position, obs_z, obs_x, obs_action_type_x, obs_action_type_z)['values']

        model.to(device)
        optimizer.zero_grad()


        def computeLoss():            
            strategy_output = model.forward_strategy(obs_strategy_z, obs_strategy_x, return_value=True)
            strategy_logits = strategy_output['values']

            tp_output = model.forward_tp(obs_action_type_z, obs_action_type_x, strategy_action, return_value=True)
            act_output = model.forward_act(obs_z, obs_x, strategy_action, return_value=True)
            
            tp_output = tp_output['values']
            act_output = act_output['values']
            
            tp_win_rate = tp_output[0]
            tp_win = tp_output[1]
            tp_lose = tp_output[2]
            act_win_rate = act_output[0]
            act_win = act_output[1]
            act_lose = act_output[2]
            
            # 共享target：win_rate 头回归真正的胜率概率 1{wp>0}
            # （BCEWithLogits 内置 sigmoid；推理侧用 sigmoid(win_rate) 作 win/lose 混合权重）
            win_target = (target_wp > 0).float().unsqueeze(-1)  # [N,1]，与 tp/act_win_rate 的 [N,1] 对齐
            tp_loss1 = F.binary_cross_entropy_with_logits(tp_win_rate, win_target)
            
            target_win = torch.where(target_wp > 0, target_wp, torch.zeros_like(target_wp)).to(device)
            target_lose = -torch.where(target_wp < 0, target_wp, torch.zeros_like(target_wp)).to(device)

            # win/lose 头回归目标 = 终局级差 target_wp（主，保留 ±1/2/3 级数区分）
            #                + 本墩即时得分 target_adp × step_reward_weight（细粒度奖励信号）。
            # 推理 argmax(out) 主要优化终局胜率/级数差，同时受当前出牌的即时得分影响。
            # combined_target = target_wp + flags.step_reward_weight * target_adp

            tp_l_w = compute_loss_(tp_win, target_adp) * target_win
            tp_l_l = compute_loss_(tp_lose, target_adp) * target_lose
            tp_loss2 = (tp_l_w.sum() / (target_win > 0).sum().clamp(min=1)
                        + tp_l_l.sum() / (target_lose > 0).sum().clamp(min=1))
            tp_loss = tp_loss1 + tp_loss2
            
            act_loss1 = F.binary_cross_entropy_with_logits(act_win_rate, win_target)
            act_l_w = compute_loss_(act_win, target_adp) * target_win
            act_l_l = compute_loss_(act_lose, target_adp) * target_lose
            act_loss2 = (act_l_w.sum() / (target_win > 0).sum().clamp(min=1)
                        + act_l_l.sum() / (target_lose > 0).sum().clamp(min=1))
            act_loss = act_loss1 + act_loss2
            
                        
            # 交叉熵优化strategy
            target_idx = target_strategy.argmax(dim=-1)
            strategy_target_loss = F.cross_entropy(strategy_logits, target_idx)
            # 无额外网络的策略价值监督：把"标签策略的 logit"tanh 后回归该局最终得分 target_wp。
            # 用 target_idx（而非模型预测）索引，与 CE 同向避免确认偏差；
            # Huber 损失缓解 tanh 饱和区梯度消失。
            _tgt_logit = strategy_logits.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)
            strategy_value_loss = F.huber_loss(torch.tanh(_tgt_logit), target_wp)
            # # 用 KL 散度：让 probs 逼近 target_strategy
            # strategy_target_loss = F.kl_div(
            #     F.log_softmax(strategy_logits, dim=-1),
            #     target_strategy,
            #     reduction='batchmean'
            # )
            
            # 分别监控真实行为策略（含探索）与事后动作标签，不把二者视作监督配对。
            with torch.no_grad():
                _sel = strategy_action.mean(dim=0)
                _tgt = target_strategy.mean(dim=0)

            loss = (tp_loss + act_loss
                    + strategy_target_loss * flags.strategy_target_loss_weight
                    + strategy_value_loss * flags.strategy_value_loss_weight)
            return tp_loss, act_loss, strategy_target_loss, strategy_value_loss, loss, _sel, _tgt
        
        if scaler:
            with torch.amp.autocast(device_type=device_type):
                tp_loss, act_loss, strategy_target_loss, strategy_value_loss, loss, _sel, _tgt = computeLoss()
        else:
            tp_loss, act_loss, strategy_target_loss, strategy_value_loss, loss, _sel, _tgt = computeLoss()
            
        stats = {
            'loss_tp_loss_' + position: tp_loss.item(),
            'loss_act_loss_' + position: act_loss.item(),
            'loss_strategy_target_' + position: strategy_target_loss.item(),
            'loss_strategy_value_' + position: strategy_value_loss.item(),
        }
        for i in range(NUM_STRATEGY):
            stats['strategy_sel_%s_%d' % (position, i)] = float(_sel[i].item())
            stats['strategy_tgt_%s_%d' % (position, i)] = float(_tgt[i].item())
        
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
            optimizer.step()
        
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    
    #自动混合精度训练
    if flags.use_mixed_precision:
        from torch.cuda.amp import  GradScaler
        print('使用混合精度训练')
        
    
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    positions = __POSITIONS__
    
    # Initialize queues
    actor_processes = []
    # mp.set_start_method("fork", force=True)
    ctx = mp.get_context('spawn')
    batch_queues = {position: ctx.SimpleQueue() for position in positions}

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = []
    for position in positions:
        stat_keys += [
            # 'mean_episode_return_' + position,
            'loss_tp_loss_' + position,
            'loss_act_loss_' + position,
            'loss_strategy_target_' + position,
            'loss_strategy_value_' + position,
        ]
    
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {position:0 for position in positions}
    position_train_frame = {position:0 for position in positions}

    # 性能定位：从 learner 线程视角累计「等待数据(get_batch)」与「计算(learn)」耗时，
    # 用于判断 fps 下降是 actor 产数据慢(空等)还是 learner 算得慢(计算)。
    perf = {'get_batch': 0.0, 'learn': 0.0, 'n': 0}

    # 策略分布监控：跨 step 累积行为策略 / 事后标签频率（落盘 strategy_dist/*.npz）。
    strategy_log = {p: {'frames': [], 'sel': [], 'tgt': []} for p in __PLAY_ROLES__}

    def _dump_strategy_log():
        dist_dir = os.path.join(flags.savelog, 'strategy_dist')
        os.makedirs(dist_dir, exist_ok=True)
        for p in __PLAY_ROLES__:
            st = strategy_log[p]
            if not st['frames']:
                continue
            np.savez(
                os.path.join(dist_dir, '%s.npz' % p),
                frames=np.asarray(st['frames'], dtype=np.int64),
                sel=np.asarray(st['sel'], dtype=np.float32),
                tgt=np.asarray(st['tgt'], dtype=np.float32),
            )
            st['frames'].clear()
            st['sel'].clear()
            st['tgt'].clear()

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in positions:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
        stats = checkpoint_states["stats"]

        if not 'mean_episode_return_bidding' in stats:
            stats.update({"mean_episode_return_bidding": 0})
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        if not "bidding" in position_frames:
            position_frames.update({"bidding": 0})
        print(f"Resuming preempted job, current stats:\n{stats}")

    
    # Starting actor processes
    banker_win_counter = ctx.Value('i', 0)
    idler_win_counter = ctx.Value('i', 0)
    exp_epsilon_shared = ctx.Value('d', flags.exp_epsilon)
    
    def update_exp():
        #调整exp_epsilon
        new_exp_epsilon = (flags.exp_epsilon_maximum - 
                            (flags.exp_epsilon_maximum - flags.exp_epsilon_minimum) * (frames / flags.total_frames)
                            )
        flags.exp_epsilon = new_exp_epsilon
        exp_epsilon_shared.value = new_exp_epsilon
        
        print(f'exp_epsilon={new_exp_epsilon:.4f}')
    update_exp()
    
    for device in device_iterator:
        if device == 0:
            for i in range(flags.num_actors):
                actor = ctx.Process(
                    target=act,
                    #i, device, batch_queues, banker_win_counter, idler_win_countermodel, flags
                    args=(i, device, batch_queues, models[device], banker_win_counter, idler_win_counter, exp_epsilon_shared, flags))
                # actor.setDaemon(True)
                actor.start()
                actor_processes.append(actor)
        elif device == 1:
            for i in range(flags.num_actors2):
                actor = ctx.Process(
                    target=act,
                    #i, device, batch_queues, banker_win_counter, idler_win_countermodel, flags
                    args=(i, device, batch_queues, models[device], banker_win_counter, idler_win_counter, exp_epsilon_shared, flags))
                # actor.setDaemon(True)
                actor.start()
                actor_processes.append(actor)

    learn_calls = {'bid': learn_bid, 'cover': learn_cover, 'banker': learn_play, 'banker_op': learn_play, 'banker_down': learn_play, 'banker_up': learn_play}
    get_batch_calls = {'bid': get_batch_bid, 'cover': get_batch_cover, 'banker': get_batch_play, 'banker_op': get_batch_play, 'banker_down': get_batch_play, 'banker_up': get_batch_play}
    
    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, position_train_frame, stats
        # pid = threading.get_ident()
        scaler = GradScaler() if flags.use_mixed_precision else None
        
        while frames < flags.total_frames:
            _t0 = __timer()
            batch = get_batch_calls[position](batch_queues[position], flags, local_lock)
            _t1 = __timer()
            _stats = learn_calls[position](position, models, learner_model.get_model(position), batch,
                           optimizers[position], scaler, flags, position_lock)
            _t2 = __timer()
            
            # 策略分布：从返回中取出（不进入全局打印 stats），单独累积与落盘
            strat_sel = strat_tgt = None
            if position in __PLAY_ROLES__:
                strat_sel = [_stats.pop('strategy_sel_%s_%d' % (position, i), 0.0) for i in range(NUM_STRATEGY)]
                strat_tgt = [_stats.pop('strategy_tgt_%s_%d' % (position, i), 0.0) for i in range(NUM_STRATEGY)]
            # b_t = __timer()
            # print(f'batch_learn({pid}):{start_t:.4f}, {a_t:.4f}, {b_t:.4f}, use time:{(b_t-a_t):.4f}')
            with lock:
                perf['get_batch'] += _t1 - _t0
                perf['learn'] += _t2 - _t1
                perf['n'] += 1
                for k in _stats:
                    stats[k] = _stats[k]
                if strat_sel is not None:
                    strategy_log[position]['frames'].append(frames)
                    strategy_log[position]['sel'].append(strat_sel)
                    strategy_log[position]['tgt'].append(strat_tgt)
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys if k in stats})
                frames += T * B
                position_frames[position] += T * B
                
                #loss变化
                position_train_frame[position] += 1
                if position_train_frame[position]%200 == 0 and position in __PLAY_ROLES__:
                    save_position = position
                    save_position_frames = position_frames[save_position]
                    save_sel = list(strat_sel)
                    save_tgt = list(strat_tgt)
                    def write_loss_thread():
                        writer = SummaryWriter(f'{flags.savelog}/tractors_loss/{save_position}')
                        writer.add_scalar(f'loss_tp_loss', stats[f'loss_tp_loss_{position}'], save_position_frames)
                        writer.add_scalar(f'loss_act_loss', stats[f'loss_act_loss_{position}'], save_position_frames)
                        writer.add_scalar('loss_strategy_target', stats[f'loss_strategy_target_{position}'], save_position_frames)
                        writer.add_scalar('loss_strategy_value', stats[f'loss_strategy_value_{position}'], save_position_frames)
                        # sel 是学习策略 ID；tgt 是事后动作类别，两者无固定语义对应。
                        for i in range(NUM_STRATEGY):
                            writer.add_scalar(f'strategy_sel/g{i+1}', save_sel[i], save_position_frames)
                            writer.add_scalar(f'strategy_tgt/g{i+1}', save_tgt[i], save_position_frames)
                        writer.close()
                    threading.Thread(
                        target=write_loss_thread, name='write_loss',
                        args=()).start()

    locks = {}
    for device in device_iterator:
        locks[device] = {position: threading.Lock() for position in positions}
    position_locks = {position: threading.Lock() for position in positions}#{'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock(), 'bidding': threading.Lock()}

    threads = []
    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in positions:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,position,locks[device][position],position_locks[position]))
                thread.start()
                threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        print('Saving checkpoint to %s', checkpointpath)
        parent_dir = os.path.dirname(checkpointpath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            

        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},  # {{"general": _models["landlord"].state_dict()}
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},  # {"general": optimizers["landlord"].state_dict()}
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in positions: # ['landlord', 'landlord_up', 'landlord_down']
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)
    # checkpoint(frames)
    
    fps_log = []
    try:
        last_checkpoint_time = __timer()
        while frames < flags.total_frames:
            start_frames = frames
            _perf_start = (perf['get_batch'], perf['learn'], perf['n'])
            position_start_frames = {k: position_frames[k] for k in position_frames}
            position_start_train_frames = {k: position_train_frame[k] for k in position_train_frame}
            
            start_time = __timer()
            time.sleep(30)

            checkpoint_frames = 0
            if __timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint_frames = frames
                checkpoint(checkpoint_frames)
                last_checkpoint_time = __timer()

            # 定期落盘策略分布原始数据（供离线下图）
            _dump_strategy_log()

            # 定期释放 CUDA 缓存并打印显存占用（区分缓存碎片化 vs 真实泄露）
            if flags.training_device != 'cpu' and torch.cuda.is_available():
                _dev = torch.device('cuda:' + str(flags.training_device))
                _alloc = torch.cuda.memory_allocated(_dev)
                _resvd = torch.cuda.memory_reserved(_dev)
                print('GPU mem (cuda:%s): allocated=%.2fGB reserved=%.2fGB'
                      % (flags.training_device, _alloc / 1e9, _resvd / 1e9))
                
                #如果_alloc 超过16.5G ， 清理一下缓存碎片
                if _resvd - _alloc > 5. * 1e9:
                    print('Warning: _resvd > _alloc, possible memory leak')
                    torch.cuda.empty_cache()

            end_time = __timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            # learner 侧耗时分解：等待数据(get_batch) vs 计算(learn)
            _pget = perf['get_batch'] - _perf_start[0]
            _plearn = perf['learn'] - _perf_start[1]
            _pn = perf['n'] - _perf_start[2]
            _ptot = _pget + _plearn
            if _pn > 0:
                print('[perf] batches=%d | wait(get_batch)=%.1fms/b  compute(learn)=%.1fms/b | wait占比=%.1f%%'
                      % (_pn, _pget / _pn * 1000, _plearn / _pn * 1000,
                         100.0 * _pget / _ptot if _ptot > 0 else 0.0))

            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            position_learn_fps = {k: (position_train_frame[k] - position_start_train_frames[k]) for k in
                            position_train_frame}
            
            #'banker', 'banker_op', 'banker_down', 'banker_up', 'bid', 'cover'
            print('After %i (L:%i O:%i U:%i D:%i B:%i C:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f O:%.1f U:%.1f D:%.1f B:%.1f C:%.1f) learn_frames:(L:%.1f O:%.1f U:%.1f D:%.1f B:%.1f C:%.1f) winner_count:(B:%i, I:%i) Stats:\n%s' % (
                     frames,
                     position_frames['banker'],
                     position_frames['banker_op'],
                     position_frames['banker_down'],
                     position_frames['banker_up'],
                     position_frames['bid'],
                     position_frames['cover'],
                     fps,
                     fps_avg,
                     position_fps['banker'],
                     position_fps['banker_op'],
                     position_fps['banker_down'],
                     position_fps['banker_up'],
                     position_fps['bid'],
                     position_fps['cover'],
                     position_learn_fps['banker'],
                     position_learn_fps['banker_op'],
                     position_learn_fps['banker_down'],
                     position_learn_fps['banker_up'],
                     position_learn_fps['bid'],
                     position_learn_fps['cover'],
                     banker_win_counter.value,
                     idler_win_counter.value,
                     pprint.pformat(stats)))
            if checkpoint_frames > 0:
                def eval_thread():
                    # 评估依赖按需加载，独立调用 learner 时不需要加载评估/绘图模块。
                    from rlcard.games.tractors.eval.evaluate_training import evaluate_training_models
                    flags.log_print = False
                    evaluate_training_models(checkpoint_frames, flags)
                threading.Thread(
                    target=eval_thread, name='eval_thread',
                    args=()).start()
            update_exp()
            
                   

    except KeyboardInterrupt:
        _dump_strategy_log()
        return
    else:
        for thread in threads:
            thread.join()
        print('Learning finished after %d frames.', frames)

    _dump_strategy_log()
    checkpoint(frames)
