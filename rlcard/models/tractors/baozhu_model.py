from rlcard.models.model import Model
from rlcard.games.tractors.utils import *


class BaoZhu_Model(Model):
    def __init__(self):
       super().__init__()
       pass
       

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        raise NotImplementedError
