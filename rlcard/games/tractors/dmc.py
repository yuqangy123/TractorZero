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

from rlcard.games.tractors.models import Model
# from .file_writer import FileWriter

from rlcard.games.tractors.train_act import *

from torch.utils.tensorboard import SummaryWriter

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down', 'bidding']}


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def compute_loss_(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2)
    return loss


def learn_bid(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    print("Learn", position)
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    
    obs_x = batch["obs_x"]
    obs_x = torch.flatten(obs_x, 0, 1).to(device)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    
    bid_target = torch.flatten(batch['bid_return'].to(device), 0, 1)
    bid_action_mask = torch.flatten(batch['bid_action_mask'].to(device), 0, 1)
    bid_score = torch.flatten(batch['bid_score'].to(device), 0, 1)
    bid_suit = torch.flatten(batch['bid_suit'].to(device), 0, 1)
    
    with lock:
        model.to(device)
        output = model.learn_bid('bid', obs_z, obs_x, bid_action_mask)
        values = output['values']
        score, suit = values[0], values[1]
        score = score.gather(1, bid_score.unsqueeze(1)).squeeze(1)
        suit = suit.gather(1, bid_suit.unsqueeze(1)).squeeze(1)
        
        #叫分，花色分数
        loss_score = compute_loss(score, bid_target)
        suit_score = compute_loss(suit, bid_target)
        loss = loss_score + suit_score
        
        stats = {
            'mean_episode_return_' + position: bid_target.mean().item(),
            'loss_' + position: loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def learn_cover(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    print("Learn", position)
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    
    obs_x = batch["obs_x"]
    obs_x = torch.flatten(obs_x, 0, 1).to(device)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    
    cover_return = torch.flatten(batch['cover_return'].to(device), 0, 1)# 1 or -1
    cover_action_mask = torch.flatten(batch['cover_action_mask'].to(device), 0, 1)
    cover_cards_mtx_target = torch.flatten(batch['cover_action'].to(device), 0, 1)
    cover_public_score = torch.flatten(batch['cover_public_score'].to(device), 0, 1) # -1 <= x <= 1
    
    with lock:
        model.to(device)
        output = model.learn_bid('cover', obs_z, obs_x, cover_action_mask)
        cover_cards_mtx = output['values'][0]
        
        cover_public_score_ = cover_public_score * cover_return
        cover_cards_mtx_target_ = cover_cards_mtx_target * cover_public_score_
        
        loss = compute_loss_(cover_cards_mtx, cover_cards_mtx_target_)
        
        stats = {
            'mean_episode_return_' + position: cover_public_score_.item(),
            'loss_' + position: loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats
    
def learn_play(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    print("Learn", position)
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    
    obs_x = batch["obs_x"]
    obs_x = torch.flatten(obs_x, 0, 1).to(device)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    
    target_adp = torch.flatten(batch['target_adp'].to(device), 0, 1)#得分
    target_wp = torch.flatten(batch['target_wp'].to(device), 0, 1)
    
    play_action_type = torch.flatten(batch['play_action_type'].to(device), 0, 1)
    play_action_type_mask = torch.flatten(batch['play_action_type_mask'].to(device), 0, 1)
    play_action = torch.flatten(batch['play_action'].to(device), 0, 1)
    
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
        model.to(device)
        #learn_action_play(self, position, z, x, action_type, legal_actions)
        win_rate, win, lose = model.learn_play('position', obs_z, obs_x, play_action_type, play_action)['values']
        loss1 = compute_loss(win_rate, target_wp)
        l_w = compute_loss_(win, target_adp) * (1. + target_wp) / 2.
        l_l = compute_loss_(lose, target_adp) * (1. - target_wp) / 2.
        loss2 = l_w.mean() + l_l.mean()
        loss = loss1 + loss2
            

        stats = {
            'mean_episode_return_' + position: sum(mean_episode_return_buf[position])/len(mean_episode_return_buf[position]),
            'loss_' + position: loss.item(),
        }
        
        optimizer.zero_grad()
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
        model = Model(device="cpu")
        model.share_memory()
        model.eval()
        models[device] = model

    positions = ['banker', 'banker_down', 'banker_up', 'bid', 'cover']
    
    # Initialize queues
    actor_processes = []
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
            'mean_episode_return_' + position,
            'loss_' + position,
        ]
    
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {position:0 for position in positions}
    position_train_frame = {position:0 for position in positions}

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
        if not 'loss_bidding' in stats:
            stats.update({"loss_bidding": 0})
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        if not "bidding" in position_frames:
            position_frames.update({"bidding": 0})
        print(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = mp.Process(
                target=act,
                args=(i, device, batch_queues, models[device], flags))
            # actor.setDaemon(True)
            actor.start()
            actor_processes.append(actor)

    learn_calls = {'bid': learn_bid, 'cover': learn_cover, 'banker': learn_play, 'banker_down': learn_play, 'banker_up': learn_play}
    get_batch_calls = {'bid': get_batch_bid, 'cover': get_batch_cover, 'banker': get_batch_play, 'banker_down': get_batch_play, 'banker_up': get_batch_play}
    
    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, position_train_frame, stats
        # pid = threading.get_ident()
        
        while frames < flags.total_frames:
            # start_t = __timer()
            batch = get_batch_calls[position](batch_queues[device][position], position, flags, local_lock)
            # a_t = __timer()
            # if 'landlord' == position:#test code 不训练地主
            #     continue
            
            _stats = learn_calls[position](position, models, learner_model.get_model(position), batch,
                           optimizers[position], flags, position_lock)
            # b_t = __timer()
            # print(f'batch_learn({pid}):{start_t:.4f}, {a_t:.4f}, {b_t:.4f}, use time:{(b_t-a_t):.4f}')
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys if k in stats})
                frames += T * B
                position_frames[position] += T * B
                
                #模型参数变化
                position_train_frame[position] += 1
                if position_train_frame[position]%1000 == 0:
                    save_position = position
                    save_position_frames = position_frames[save_position]
                    def write_model_param_thread():
                        writer = SummaryWriter(f'{flags.savelog}/mahjong_parameters/{save_position}')
                        learn_model = learner_model.get_model(save_position)
                        for name, param in learn_model.named_parameters():
                            # 记录权重
                            writer.add_histogram(f'weights/{name}', param.data, save_position_frames)
                            # 记录梯度（如果存在）
                            if param.grad is not None:
                                writer.add_histogram(f'gradients/{name}', param.grad, save_position_frames)
                    threading.Thread(
                        target=write_model_param_thread, name='write_model_param',
                        args=()).start()


    threads = []
    locks = {}
    for device in device_iterator:
        for position in positions:
            locks[device] = {position: threading.Lock() for position in positions}#'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock(), 'bidding': threading.Lock()}
    position_locks = {position: threading.Lock() for position in positions}#{'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock(), 'bidding': threading.Lock()}

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
                '%s/%s/%s' % (flags.savedir, flags.xpid, "general_"+position+'_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            print('After %i (L:%i D:%i U:%i B:%i C:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f D:%.1f U:%.1f B:%.1f C:%.1f) Stats:\n%s',
                     frames,
                     position_frames['banker'],
                     position_frames['banker_down'],
                     position_frames['banker_up'],
                     position_frames['bid'],
                     position_frames['cover'],
                     fps,
                     fps_avg,
                     position_fps['banker'],
                     position_fps['banker_down'],
                     position_fps['banker_up'],
                     position_fps['bid'],
                     position_fps['cover'],
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        print('Learning finished after %d frames.', frames)

    checkpoint(frames)
