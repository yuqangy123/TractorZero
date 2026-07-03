from rlcard.games.tractors.dmc import parser
from rlcard.games.tractors.dmc.dmc import train
import os, platform
import torch
import random
import numpy as np

def clear_env_list():
    import requests
    print(requests.post('http://192.168.112.4:8999/clear_env'))
    
if __name__ == '__main__':
    # 设置 random 模块的随机数种子
    rseed = 78420
    random.seed(rseed)
    # 设置 NumPy 的随机数种子
    np.random.seed(rseed)
    # 设置 PyTorch 的随机数种子
    torch.manual_seed(rseed)
    # 为所有GPU设置随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rseed)
    
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    
    #如果是linxu 系统
    is_linux = platform.system() == 'Linux'
    if is_linux:
        #切换 PyTorch 的共享策略，通过修改一个配置就能绕过文件描述符的限制。
        #这个改动将 PyTorch 共享内存的方式从 file_descriptor 切换为 file_system。它会使用 /dev/shm 下的文件来管理共享内存，从而有效规避文件描述符数量的限制
        torch.multiprocessing.set_sharing_strategy('file_system')
        # clear_env_list()
        
    else:
        pass
        flags.num_actors = 1
        flags.num_threads = 1
        flags.num_actor_devices = 1
        flags.actor_device_cpu = True
        flags.training_device = 'cpu'
        
    train(flags)