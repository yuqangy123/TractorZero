"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    
    z = torch.from_numpy(obs['z']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action']).to(device)
    legal_actions = torch.from_numpy(obs['legal_actions'])
    
    obs = {
           'z': z,
           'x': x_no_action,
           }
    return position, obs, legal_actions

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self, model, device, flags=None):
        obs = self.env.reset(model, device, flags=flags)
        initial_position, initial_obs, legal_actions = _format_observation(obs, self.device)
        
        self.episode_return = torch.zeros(1, 1)
        
        # initial_done = torch.ones(1, 1, dtype=torch.bool)
        return initial_position, initial_obs, dict(
            done=False,
            legal_actions=legal_actions,
            episode_return=self.episode_return,
        )

    def step(self, action, model, device, flags=None):
        obs, reward, done, = self.env.step(action)

        self.episode_return = reward
        episode_return = self.episode_return
        
        if done:
            obs = self.env.reset(model, device, flags=flags)
            self.episode_return = torch.zeros(1, 1)
        
        position, obs, legal_actions = _format_observation(obs, self.device)
        # reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return position, obs, dict(
            done=done,
            legal_actions=legal_actions,
            episode_return=episode_return,
        )
    

    def close(self):
        self.env.close()
