import numpy as np
import traceback

from rlcard.games.tractors.env.env import Env
from rlcard.games.tractors.env.utils import *


def show(tag, obs):
    print('===', tag, 'stage=', env._stage)
    for k in ['x', 'z', 'x_batch', 'z_batch', 'legal_types', 'legal_actions']:
        v = obs.get(k)
        if v is None:
            print('   ', k, 'None')
        elif hasattr(v, 'shape'):
            print('   ', k, 'shape=', tuple(v.shape), 'dtype=', v.dtype)
        else:
            try:
                print('   ', k, type(v).__name__, 'len=', len(v))
            except Exception:
                print('   ', k, type(v).__name__)


env = Env('adp')
last_stage = None
try:
    obs = env.reset(None, 'cpu')
    show('reset', obs)
except Exception:
    traceback.print_exc()

for step_i in range(800):
    stage = env._stage
    if stage != last_stage:
        last_stage = stage
        print('>> stage changed to', stage)
    try:
        if stage == 'bid':
            legal = env._bid_infoset.legal_actions
            item = legal[np.random.randint(len(legal))]
            suit = item[0] if isinstance(item, (list, tuple)) else item
            obs, done = env.step([suit])
        elif stage == 'cover':
            hold = env._cover_infoset.player_hand_cards
            act = list(np.random.choice(hold, 8, replace=False))
            obs, done = env.step([act])
            show('cover', obs)
        elif stage in ('ready', 'play', 'roundend'):
            info = env._game_infoset
            flat = []
            for tp, actions in enumerate(info.legal_actions):
                for a in actions:
                    flat.append((tp, a))
            if not flat:
                print('no legal play; break')
                break
            tp, a = flat[np.random.randint(len(flat))]
            obs, done = env.step([a])
            show('play', obs)
        else:
            print('unexpected stage', stage)
            break
        if done:
            print('game done at step', step_i)
            break
    except Exception:
        traceback.print_exc()
        break
