import os
import argparse
from .simulation import evaluateTrainingModel
import timeit
from torch.utils.tensorboard import SummaryWriter




def evaluate_training_models(frames, flags):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    modelist = {
        'eval_model' : {'bid':f'tractors_checkpoints/TractorZero_I/bid_{frames}.ckpt',
                        'cover':f'tractors_checkpoints/TractorZero_I/cover_{frames}.ckpt',
                        'banker':f'tractors_checkpoints/TractorZero_I/banker_{frames}.ckpt',
                        'banker_down':f'tractors_checkpoints/TractorZero_I/banker_down_{frames}.ckpt',
                        'banker_op':f'tractors_checkpoints/TractorZero_I/banker_op_{frames}.ckpt',
                        'banker_up':f'tractors_checkpoints/TractorZero_I/banker_up_{frames}.ckpt',
                        },
        
        'opp_model' : { 
                        # 'random':{
                        #         'bid':'random', 'cover':'random', 'banker':'random', 'banker_down':'random', 'banker_op':'random', 'banker_up':'random'
                        #         },
                       'rule':{
                                'bid':'rule', 'cover':'rule', 'banker':'rule', 'banker_down':'rule', 'banker_op':'rule', 'banker_up':'rule'
                                },
                        },
                }
    
    print('start evaluate training model..\n')
    timer = timeit.default_timer
    start_t = timer()
    
    log_path = f'{flags.savelog}/train'
    writer = SummaryWriter(log_dir=log_path)
    
    evaluateTrainingModel(modelist, frames, writer, flags)
    
    writer.close()
    
    end_t = timer()
    print(f'evaluate training model finish, use time:{(end_t-start_t):.4f}\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dou Dizhu Evaluation')
    parser.add_argument('--savelog', default='./log',
                    help='log dir data will be saved')
    parser.add_argument('--use_wandb', action='store_true',
                    help='enable wandb logging')
    parser.add_argument('--wandb_project', default='tractor-zero',
                    help='wandb project name')
    parser.add_argument('--wandb_name', default=None,
                    help='wandb run name')
    parser.add_argument('--wandb_entity', default=None,
                    help='wandb entity/team')
    parser.add_argument('--eval_data', default=None, type=str,
                    help='Path to pre-generated fixed deal data (pkl).')
    args = parser.parse_args()
    args.log_print = True

    evaluate_training_models(0, args)