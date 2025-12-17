import os
import argparse
# from tractorzero.dpo import parser, trainer
# from rlcard.games.tractors import train_play
from rlcard.games.tractors import train_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Tractors in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='tractors',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy'
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='0',
    )
    parser.add_argument(
        '--load_model',
        default=True,
        action='store_true',
        help='Load an existing model',
    )
    parser.add_argument(
        '--xpid',
        default='leduc_holdem',
        help='Experiment id (default: leduc_holdem)',
    )
    parser.add_argument(
        '--savedir',
        default='experiments/predictor_result',
        help='Root dir where experiment data will be saved'
    )
    parser.add_argument(
        '--save_interval',
        default=30,
        type=int,
        help='Time interval (in minutes) at which to save the model',
    )
    
    parser.add_argument(
        '--training_device',
        default="0",
        type=str,
        help='The index of the GPU used for training models',
    )
    parser.add_argument(
        '--gpu_devices',
        default="0",
        type=str,
    )
    
    
    parser.add_argument(
        '--lr',
        default=0.005,
        type=float,
    )
    parser.add_argument(
        '--num_actor_devices',
        default=1,
        type=int,
        help='The number of devices used for simulation',
    )




    parser.add_argument(
        '--num_threads',
        default=1,
        type=int,
        help='learn线程',
    )    
    parser.add_argument(
        '--num_actors',
        default=4,
        type=int,
        help='The number of actors for each simulation device',
    )



    parser.add_argument(
        '--unroll_length',
        default=160,
        type=int,
    )
    parser.add_argument(
        '--num_buffers',
        default=128,
        type=int,
    )
    parser.add_argument(
        '--batch_size',#batchszie要小于num_buffers，batchszie不够就会一直等待足够的num_buffers，num_buffers又会等待batchsize训练数据释放
        default=2,
        type=int,
    )



    

    parser.add_argument(
        '--total_frames',
        default=100000000000,
        type=int,
    )
    parser.add_argument(
        '--disable_checkpoint',
        default=False,
        type=bool,
    )
    parser.add_argument('--max_grad_norm', default=40., type=float,
        help='Max norm of gradients')
    

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train_predict.train(args)