import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np
import logging
import torch
from torch import multiprocessing as mp
from torch import nn
import traceback
from rlcard.utils.file_writer import FileWriter
from rlcard.games.tractors.models import tractorModel
from rlcard.games.tractors.models import tractorActor
from .env.tractors_env import run
# from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act
shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('tractorsZero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['bidding', 'conver', 'play']}

#创建共享经验池
# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
import typing
Buffers = typing.Dict[str, typing.List[torch.Tensor]]
def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    # def store_transition(self, reward: float, done: float, state: np.ndarray, action: float, log_prob: float, value: float):
    # self.buffer["rewards"][self.ptr] = reward
    # self.buffer["dones"][self.ptr] = done
    # self.buffer["states"][self.ptr] = state.cpu().detach().numpy()
    # self.buffer["actions"][self.ptr] = action.cpu().detach().numpy()
    # self.buffer["log_probs"][self.ptr] = log_prob.cpu().detach().numpy()
    # self.buffer["values"][self.ptr] = value.cpu().detach().numpy()

    T = flags.unroll_length

    

    play_buffer, cover_buffer = {}
    for device in device_iterator:
        play_buffer[device] = {}
        cover_buffer[device] = {}

        #叫牌buffer obs_x, bid_card, left_num
        # state_dim = 319
        # specs = dict(
        #     score=dict(size=(T,), dtype=torch.float32),
        #     states=dict(size=(T, state_dim), dtype=torch.int8),
        #     bid_card=dict(size=(T, 2,14,4), dtype=torch.int8),
        #     left_num=dict(size=(T,), dtype=torch.float32),
        # )
        # _buffers: Buffers = {key: [] for key in specs}
        # for _ in range(flags.num_buffers):
        #     for key in _buffers:
        #         if not device == "cpu":
        #             _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
        #         else:
        #             _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
        #         _buffers[key].append(_buffer)
        # buffer[device]['bid'] = _buffers

        #埋底牌 own_cards, partner_bid_cards, rival_bid_cards
        specs = dict(
            cover_cards=dict(size=(T,2*4*14), dtype=torch.float32),
            hand_cards=dict(size=(T, 2*4*14), dtype=torch.float),
            partner_bid_card=dict(size=(T,3,2*4*14), dtype=torch.float32),
            rival_bid_card=dict(size=(T,3,2*4*14), dtype=torch.float32),
            # rival_bid_seat=dict(size=(T, 4), dtype=torch.float32),
            level_cards=dict(size=(T,2*4*14), dtype=torch.float32),
            # seat=dict(size=(T, 4), dtype=torch.int8),
            # banker=dict(size=(T, 4), dtype=torch.int8),
            reward=dict(size=(T, ), dtype=torch.float32),
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                if not device == "cpu":
                    _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                else:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                _buffers[key].append(_buffer)
        cover_buffer[device] = _buffers

    
        #出牌buffer
        for position in ['banker', 'player']:#庄家 闲家
            state_dim = 319 if position == 'banker' else 430#状态维度
            specs = dict(
                #以下是场面信息，每一回合中对所有玩家相同
                history_play_card=dict(size=(T,15,4,2*4*14), dtype=torch.int8),
                history_play_seat=dict(size=(T,15,4,4), dtype=torch.int8),
                history_play_team=dict(size=(T,15,4,4), dtype=torch.int8),
                history_played_card=dict(size=(T,2*4*14), dtype=torch.int8),
                history_level_card=dict(size=(T,2*4*14), dtype=torch.int8),
                history_score_card=dict(size=(T,2*4*14), dtype=torch.int8),
                history_remain_score_card=dict(size=(T,2*4*14), dtype=torch.int8),
                history_banker=dict(size=(T, 2), dtype=torch.int8),
                history_public_card=dict(size=(T,2*4*14), dtype=torch.int8),

                #每一回合范围内的每一步的场面信息，
                round_play_card=dict(size=(T,4,2*4*14), dtype=torch.int8),
                round_play_seat=dict(size=(T,4), dtype=torch.int8),
                round_play_team=dict(size=(T,4), dtype=torch.int8),
                # point=dict(size=(T,15,), dtype=torch.int8),
                seat=dict(size=(T, 4, 4), dtype=torch.int8),
                team=dict(size=(T, 4, 2), dtype=torch.int8),
                hand_cards=dict(size=(T, 4, 2*4*14), dtype=torch.float),
                player_remain_card_num=dict(size=(T, 4, 25), dtype=torch.int8),
                action=dict(size=(T,4, 2*4*14), dtype=torch.int8),

                #奖励信息
                reward=dict(size=(T,4, ), dtype=torch.float32),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            play_buffer[device] = _buffers
    return play_buffer, cover_buffer

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch
    #使用PPOClip的generate_batches


    
def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

def learn(position,
          actor_models,
          model,
          batch,
        #   optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch[''][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
        
    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+position: loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def train(flags):
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
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
    actors = {}
    for device in device_iterator:
        model = tractorActor(device=device, flags=flags)
        model.share_memory()
        model.eval()
        actors[device] = model

    # Initialize buffers
    play_buffer, cover_buffer = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    play_free_queue = {}
    play_full_queue = {}
    cover_free_queue = {}
    cover_full_queue = {}
        
    for device in device_iterator:
        play_free_queue[device] = ctx.SimpleQueue()
        play_full_queue[device] = ctx.SimpleQueue()
        cover_free_queue[device] = ctx.SimpleQueue()
        cover_full_queue[device] = ctx.SimpleQueue()

    # Learner model for training
    learner_playModel = tractorActor(device=device, flags=flags)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'bid':0, 'cover':0, 'banker':0, 'player':0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ['cover', 'player']:
            assert learner_playModel.load_state_dict(k, checkpoint_states["model_state_dict"][k]), f'model {k} load checkpoint failed'
            assert learner_playModel.load_optim_checkpoint(k, checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                actors[device].load_state_dict(k, checkpoint_states["model_state_dict"][k])
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor_pro = ctx.Process(
                target=run,
                args=(i, device, actors[device], play_free_queue[device], play_full_queue[device], cover_free_queue[device], cover_full_queue[device], play_buffer[device], cover_buffer[device], flags))
            actor_pro.start()
            actor_processes.append(actor_pro)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            _stats = learn(position, actors, learner_playModel[position], batch, flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            play_free_queue[device].put(m)
            play_free_queue[device].put(m)
            cover_free_queue[device].put(m)
            cover_free_queue[device].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'cover': threading.Lock(), 'player': threading.Lock()}
    position_locks = {'cover': threading.Lock(), 'player': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,position,locks[device][position],position_locks[position]))
                thread.start()
                threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': {k: learner_playModel[k].state_dict() for k in learner_playModel},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
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
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
