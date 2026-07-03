import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='TractorZero_I',
                    help='Experiment id (default: TractorZero_I)')
parser.add_argument('--save_interval', default=20, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='adp', type=str, choices=['adp'],
                    help='Use ADP as reward (default: ADP)')

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
# parser.add_argument('--actor_device_cpu', default=True, type=bool,
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='1', type=str,#用哪块GPU执行该项目
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=30, type=int,#actor数量
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
# parser.add_argument('--training_device', default='cpu', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', default=True, type=bool, #action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='tractors_checkpoints',
                    help='Root dir where experiment data will be saved')
parser.add_argument('--savelog', default='./log',
                    help='log dir data will be saved')


# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')

parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--bid_exp_epsilon', default=0.4, type=float,#bid模型随机探索系数
                    help='The probability for bidding exploration')
parser.add_argument('--temperature', default=1., type=float)
parser.add_argument('--decay', default=1., type=float)

parser.add_argument('--batch_size', default=30, type=int,
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
parser.add_argument('--learning_rate_banker', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--learning_rate_idler', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--learning_rate_bid', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--learning_rate_cover', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-8, type=float,
                    help='RMSProp epsilon')
parser.add_argument('--coop_loss_weight', default=0.1, type=float,
                    help='farmer cooperate loss weight')