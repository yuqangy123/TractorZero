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

from rlcard.games.tractors.models import tractorActor
from .env.tractors_env import run

from rlcard.games.tractors import learner_predictor_model


# from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act
shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('tractorsZero_trainer_belief')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['predictor']}

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
    T = flags.unroll_length

    belief_buffer = {}
    for device in device_iterator:
        belief_buffer[device] = {}

        #隐藏信息预测模型buffer
        specs = dict(
            history_play_card=dict(size=(T,15,4,2,4,15), dtype=torch.float32),
            history_play_seat=dict(size=(T,15,4,4), dtype=torch.float32),
            history_played_card=dict(size=(T,2,4,15), dtype=torch.float32),
            history_bid_card=dict(size=(T,2,2,4,15), dtype=torch.float32),
            history_bid_seat=dict(size=(T,2,4), dtype=torch.float32),
            round_play_card=dict(size=(T,4,2,4,15), dtype=torch.float32),
            round_play_seat=dict(size=(T,4,4), dtype=torch.float32),

            score_card=dict(size=(T,2,4,15), dtype=torch.float32),
            remain_score_card=dict(size=(T,2,4,15), dtype=torch.float32),
            my_seat=dict(size=(T,4), dtype=torch.float32),
            banker_seat=dict(size=(T,4), dtype=torch.float32),
            
            # label
            public_card=dict(size=(T,2,4,15), dtype=torch.float32),
            hand_card=dict(size=(T,4,2,4,15), dtype=torch.float32),
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                if not device == "cpu":
                    _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                else:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                _buffers[key].append(_buffer)
        belief_buffer[device] = _buffers

    return belief_buffer

# def get_batch(free_queue,
#               full_queue,
#               buffers,
#               flags,
#               lock):
#     """
#     This function will sample a batch from the buffers based
#     on the indices received from the full queue. It will also
#     free the indices by sending it to full_queue.
#     """
#     with lock:
#         indices = [full_queue.get() for _ in range(flags.batch_size)]
#     batch = {
#         key: torch.stack([buffers[key][m] for m in indices], dim=1)
#         for key in buffers
#     }
#     for m in indices:
#         free_queue.put(m)
#     return batch
#     #使用PPOClip的generate_batches


    
def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

def train(flags):
    if not hasattr(flags, 'actor_device_cpu') or flags.training_device != 'cpu':
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

    if hasattr(flags, 'actor_device_cpu'):
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    actors = {}
    for device in device_iterator:
        model = tractorActor(device=device, args=flags)
        model.share_memory()
        model.eval()
        actors[device] = model

    # Initialize buffers
    belief_buffer = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
        
    for device in device_iterator:
        free_queue[device] = ctx.SimpleQueue()
        full_queue[device] = ctx.SimpleQueue()
        

    # Learner model for training
    learner_actor = {'predictor':tractorActor(device=device, args=flags)}

    # model optimizer
    optimizers = {'predictor':learner_predictor_model.create_optimizer(lr=flags.lr, learner_model=learner_actor['predictor'].get_model('predictor'))}
    

    # Stat Keys
    stat_keys = [
        # 'mean_episode_return',
        'loss'
    ]
    stats = {'predictor':{k: 0 for k in stat_keys}}
    model_frames = {'predictor':0}
    frames = {'predictor':0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ['predictor']:
            assert learner_actor[k].get_model('predictor').load_state_dict(k, checkpoint_states["model_state_dict"][k]), f'model {k} load checkpoint failed'
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                actors[device].load_state_dict(k, checkpoint_states["model_state_dict"][k])
            stats[k] = checkpoint_states["stats"][k]
            frames[k] = checkpoint_states["frames"][k]
            model_frames = checkpoint_states["model_frames"][k]
            log.info(f"Resuming preempted {k} model job, current stats:\n{stats}")
    
    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device].put(m)

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor_pro = ctx.Process(
                target=run,
                args=(i, device, actors[device], free_queue[device], full_queue[device], belief_buffer[device], flags))
            actor_pro.start()
            actor_processes.append(actor_pro)

    from ctypes import c_int, c_float, c_double, c_bool
    learn_count = ctx.Value(c_int, 0)
    
    
    def batch_and_learn_predictor_model(i, device, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, model_frames, stats
        tid = threading.get_ident()
        print('batch_and_learn_predictor_model', tid)
        while frames['predictor'] < flags.total_frames:
            batch = learner_predictor_model.get_batch(free_queue[device], full_queue[device], belief_buffer[device], flags, local_lock)
            _stats = learner_predictor_model.learn(actors[device], learner_actor['predictor'], batch, optimizers['predictor'], device, flags, mean_episode_return_buf, position_lock, learn_count)

            with lock:
                for k in _stats:
                    stats['predictor'][k] = _stats[k]
                to_log = dict(frames=frames['predictor'])
                to_log.update({k: stats['predictor'][k] for k in stat_keys})
                plogger.log(to_log)
                frames['predictor'] += T * B
                model_frames['predictor'] += T * B
    

    threads = []
    data_locks = {}
    for device in device_iterator:
        data_locks[device] = {'predictor': threading.Lock()}
    model_learn_locks = {'predictor': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            # for position in ['landlord', 'landlord_up', 'landlord_down']:
            thread = threading.Thread(
                target=batch_and_learn_predictor_model, name='batch-and-learn-predictor-model-%d' % i, args=(i, device, data_locks[device]['predictor'], model_learn_locks['predictor']))
            thread.start()
            threads.append(thread)

            
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': {k: learner_actor[k].get_model('predictor').state_dict() for k in learner_actor},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames['predictor'],
            'model_frames': model_frames['predictor']
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['predictor']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
            torch.save(learner_actor[position].get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames['predictor'] < flags.total_frames:
            start_frames = frames['predictor']
            position_start_frames = {k: model_frames[k] for k in model_frames}
            start_time = timer()
            time.sleep(60)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames['predictor'] - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            #position_fps = {k:(model_frames[k]-position_start_frames[k])/(end_time-start_time) for k in model_frames}
            # log.info('After %i (predictor:%i) frames: @ %.1f fps (avg@ %.1f fps) (predictor:%.1f) Stats:%s\n',
            #          frames['predictor'],
            #          model_frames['predictor'],
            #          fps,
            #          fps_avg,
            #          position_fps['predictor'],
            #          pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
