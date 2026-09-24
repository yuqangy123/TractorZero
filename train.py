from rlcard.games.tractors.dmc import parser
import platform
import torch


def clear_env_list():
    import requests
    print(requests.post('http://192.168.112.4:8999/clear_env'))

def evaluateModel(flags):
    from rlcard.games.tractors.eval.evaluate_training import evaluate_training_models
    flags.log_print = True
    flags.evaluate_device = 0
    # flags.print_game_log = True
    evaluate_training_models(14997000, flags)

def showPlt(flags):
    from rlcard.games.tractors.eval.plot_results import showplt
    showplt()

def train(flags):
    from rlcard.games.tractors.dmc.dmc import train
    train(flags)
    
if __name__ == '__main__':
   
    flags = parser.parse_args()
    # os.environ["e"] = "1,0"#flags.gpu_devices
    
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
    # showPlt(flags)
    # evaluateModel(flags)