import argparse

parser = argparse.ArgumentParser(description='TractorZero: PyTorch Tractor AI')

# General Settings
parser.add_argument('--xpid', default='TractorZero_I',
                    help='Experiment id (default: TractorZero_I)')
parser.add_argument('--save_interval', default=60, type=int,
                    help='Time interval (in minutes) at which to save the model')
parser.add_argument('--objective', default='adp', type=str, choices=['adp'],
                    help='Use ADP as reward (default: ADP)')

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
# parser.add_argument('--actor_device_cpu', default=True, type=bool,
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='1,0', type=str,#用哪块GPU执行该项目
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=2, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=22, type=int,#actor数量
                    help='The number of actors for device 1')
parser.add_argument('--num_actors2', default=12, type=int,#
                    help='The number of actors for device 2')
parser.add_argument('--training_device', default='0', type=str,#default='cpu'
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', default=True, type=bool, #action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', default=False, action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='tractors_checkpoints',
                    help='Root dir where experiment data will be saved')
parser.add_argument('--savelog', default='./log',
                    help='log dir data will be saved')

# Evaluation settings
parser.add_argument('--eval_data', default='eval_data.pkl', type=str,
                    help='Path to pre-generated fixed deal data (pkl). '
                         'None 时评估使用随机发牌。')
parser.add_argument('--evaluate_device', default='1', type=str,#1是第2块GPU，cpu是用cpu跑
                    help='The index of the GPU used for evaluate models.')
parser.add_argument('--eval_series', default=4, type=int,#4
                    help='每次评估并行运行的完整 series（2->A）数量，聚合统计收窄置信区间')
parser.add_argument('--eval_concurrency', default=1, type=int,
                    help='同时运行的评估进程上限（显存受限时保持 1；evaluate_device=cpu 时可调大并行提速）')

# wandb (optional)
parser.add_argument('--use_wandb', default=True,
                    help='enable wandb logging for evaluation')
parser.add_argument('--wandb_project', default='tractor-zero', type=str,
                    help='wandb project name')
parser.add_argument('--wandb_name', default=None, type=str,
                    help='wandb run name')
parser.add_argument('--wandb_entity', default=None, type=str,
                    help='wandb entity/team')


# Hyperparameters
parser.add_argument('--total_frames', default=400000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.22, type=float,
                    help='The probability for exploration')
parser.add_argument('--exp_epsilon_minimum', default=0.02, type=float,
                    help='The probability for exploration')
parser.add_argument('--exp_epsilon_maximum', default=0.22, type=float,
                    help='The probability for exploration')

parser.add_argument('--temperature', default=1., type=float)
parser.add_argument('--decay', default=1., type=float)

parser.add_argument('--batch_size', default=6, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,#需正偶数
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=1, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=20.0, type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate_banker', default=0.0001, type=float,#0.0001
                    help='Learning rate')
parser.add_argument('--learning_rate_idler', default=0.0001, type=float,#0.0001
                    help='Learning rate')
parser.add_argument('--learning_rate_bid', default=0.0001, type=float,#0.0001
                    help='Learning rate')
parser.add_argument('--learning_rate_cover', default=0.0001, type=float,#0.0001
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-8, type=float,
                    help='RMSProp epsilon')

#learning settings
parser.add_argument('--use_mixed_precision', default=True, action='store_true',
                    help='Enable mixed precision (AMP) training on CUDA')
parser.add_argument('--strategy_value_loss_weight', default=3.0, type=float,
                    help='strategy value(Huber on target_wp) loss weight')
parser.add_argument('--strategy_target_loss_weight', default=1.0, type=float,
                    help='strategy CE(target_strategy) loss weight，压低后与回归项同量级')
parser.add_argument('--step_reward_weight', default=0.4, type=float,
                    help='本墩即时得分 target_adp 叠加进 win/lose 头回归目标的权重，'
                         '主目标仍为终局级差 target_wp（保留 ±1/2/3 级数区分）')
