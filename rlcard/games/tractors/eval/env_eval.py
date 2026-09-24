"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
#改编自env_utils.py
import torch

def _format_observation(obs):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    # if not device == "cpu":
    #     device = 'cuda:' + str(device)
    # device = torch.device(device)
    
    z = torch.from_numpy(obs['z'])#.to(device)
    x = torch.from_numpy(obs['x'])#.to(device)
    z_batch = torch.from_numpy(obs['z_batch'])#.to(device)
    x_batch = torch.from_numpy(obs['x_batch'])#.to(device)
    
    if isinstance(obs['legal_actions'], list):
        legal_actions = obs['legal_actions']
    else:
        legal_actions = torch.from_numpy(obs['legal_actions'])
    
    
    obs = {
           'z': z,
           'x': x,
           'z_batch': z_batch,
            'x_batch': x_batch,
           }
    return position, obs, legal_actions

class EnvironmentEval:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device

    def initial(self, flags=None):
        obs = self.reset()
        initial_position, initial_obs, legal_actions = _format_observation(obs)
        self._flags = flags
        
        # initial_done = torch.ones(1, 1, dtype=torch.bool)
        return initial_position, initial_obs, dict(
            done=False,
            legal_actions = legal_actions,
            stage = self.env._stage,
        )

    def step(self, action):
        obs_ori,  done, = self.env.step(action)

        if obs_ori:
            position, obs, legal_actions = _format_observation(obs_ori)
        # reward = torch.tensor(reward).view(1, 1)
        # done = torch.tensor(done).view(1, 1)

        if done:
            step_reward = self._get_step_reward()
            game_reward = self._get_reward()
            game_score = self._get_game_score()
            #最终结算就评估完了一局
            if self.env._stage == "finalend":
                return 0, {}, dict(
                    done=done,
                    stage = self.env._stage,
                    step_reward = step_reward,
                    game_reward = game_reward,
                    game_score = game_score,
                )
            
            
            obs = self.reset()
            position, obs, legal_actions = _format_observation(obs)
            env_output = dict(
                done=done,
                legal_actions = legal_actions,
                stage = self.env._stage,
                step_reward = step_reward,
                game_reward = game_reward,
                game_score = game_score,
            )
        else:
            env_output = dict(
                done=done,
                legal_actions = legal_actions,
                stage = self.env._stage,
            )
            if env_output['stage'] == 'roundend':
                env_output['step_reward'] = self._get_step_reward()
            if position in ['banker', 'banker_op', 'banker_down', 'banker_up']:
                env_output['legal_types'] = obs_ori['legal_types']
            
        return position, obs, env_output
    
    def reset(self):
        return self.env.reset()
        
    def close(self):
        self.env.close()

    def _get_step_reward(self):
        return self.env._get_step_reward()
    
    def _get_reward(self):
        return self.env._get_reward()
        
    def _get_game_score(self):
        return self.env._get_game_score()
    
    def _get_infosets(self):
        return self.env._get_infosets()
    
    def _get_last_bid_rule(self):
        return self.env._get_last_bid_rule()
    
    
    