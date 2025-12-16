
# from rlcard.games.tractors.models.ppo_model import MLPActorCritic, MLPQNetwork
# from rlcard.tractors.utils import *
# from rlcard.tractors.models.tractors_model import TractorsModel as Model

from rlcard.games.tractors.models.bid_model import BidModel
from rlcard.games.tractors.models.cover_model import CoverModel
from rlcard.games.tractors.models.player_model import PPOClip as PlayModel
from rlcard.games.tractors.models.predictor_model import Predictor as PredictModel
from rlcard.games.tractors.utils import *
import torch as t

ActionNumber = 2
# torch.set_num_threads(8)
class TractorsActor():
    def __init__(self, device='cpu', args={}):
        self.__models = dict()
        # self.__models['bid'] = BidModel(args).to(device)
        # self.__models['cover'] = CoverModel(args).to(device)
        
        # #不区分team
        # # self.__models['banker'] = PlayModel(args).to(device)
        # self.__models['player'] = PlayModel(args).to(device)
        
        # self, hidden_dim, num_opps
        _predictmodel = PredictModel()
        _predictmodel.toDevice(device != 'cpu' and t.device('cuda:'+str(device)) or device)
        total_predictmodel_params = sum(p.numel() for p in _predictmodel.parameters())
        print("Total predictmodel parameters:", total_predictmodel_params)
        self.__models['predictor'] = _predictmodel
        pass


    # def bidMajorCard(self, deal_card, hold_card, bid_card, own_pos, called_list, major, level):
    # # def biddingMajor(self, get_card, hold_card, bid_card, own_seat, bid_history, major, level):
    #     '''
    #         owned 自己的位置
    #         called 首次报主的玩家位置
    #         snatched 反主的玩家位置
    #     '''
    #     hold_card = deal_card+hold_card
    #     hold_card = cards2matrix(hold_card, level, major)
    #     left_num = 25 - sum(hold_card)
        

    #     bid_card = cards2matrix(bid_card, level, major)

    #     #友方的牌
    #     partner_called = []
    #     for bid_pro in bid_history:
    #         if bid_pro[0] != own_seat and (bid_pro[0]+2)%__PLAYER_COUNT__ == own_seat:
    #             partner_called = bid_pro[1]
    #             break                
    #     partner_called = cards2matrix(partner_called, level)

    #     pre = self.__models.bidding(hold_card, bid_card, left_num)
    #     return pre > 0.5

    # def coverCard(self, public_card, hold_card, own_seat, bid_history, level, major):
    #     hold_cards = public_card+hold_card
    #     hold_cards_mat = cards2matrix(hold_cards, level, major)

    #     score_cards = cards2matrix([16,17,18,19,36,37,38,39,48,49,50,51,70,71,72,73,90,91,92,93,102,103,104,105], level, major)#分數牌
        
    #     partner_called, rival_called = [],[]
    #     for bid_pro in bid_history:
    #         if bid_pro[0] != own_seat:
    #             if (bid_pro[0]+2)%__PLAYER_COUNT__ == own_seat:
    #                 partner_called = bid_pro[1]
    #             else:
    #                 rival_called = bid_pro[1]
                
    #     partner_called_mat = cards2matrix(partner_called, level, major)
    #     rival_called_mat = cards2matrix(rival_called, level, major)
        
    #     cover_card = self.__models.cover(hold_cards_mat, partner_called_mat, rival_called_mat, hold_cards)
    #     cover_card = matrix2cards(cover_card)
    #     return cover_card

    def playCard(self, obs_x, actions_fixed, actions_discard, discard_num, team, banker_seat):
        fixed_cards, display_cards,_,_,_ = self.__models['player'].act(obs_x, actions_fixed, actions_discard, discard_num, team)

        out_cards = fixed_cards + display_cards
        out_cards = matrix2card(out_cards)
        return out_cards

    def predictCard(self, input_x, isTrain=None):
        # history_play_card=input_x['history_play_card']
        # history_play_seat=input_x['history_play_seat']
        # history_played_card=input_x['history_played_card']
        # history_bid_card=input_x['history_bid_card']
        # history_bid_seat=input_x['history_bid_seat']   
        # round_play_card=input_x['round_play_card']
        # round_play_seat=input_x['round_play_seat']
        # score_card=input_x['score_card']
        # remain_score_card=input_x['remain_score_card']
        # my_seat=input_x['my_seat']
        # mask_card_mat = history_played_card
        opp_probs, bottom_prob = self.__models['predictor'](input_x, isTrain)
        return opp_probs, bottom_prob
    def load_state_dict(self, model_name, dict):
        if model_name in self.__models:
            self.__models[model_name].load_state_dict(dict)
            return True
        return False
        
    def load_optim_checkpoint(self, model_name, dict):
        if model_name in self.__models:
            self.__models[model_name].load_optim_checkpoint(dict)
            return True
        return False
    
    def get_model(self, model_name):
        if model_name in self.__models:
            return self.__models[model_name]
    
    def share_memory(self):
        for name, model in self.__models.items():
            model.share_memory()

    def eval(self):
        for name, model in self.__models.items():
            model.eval()

    def parameters(self, model_name):
        if model_name in self.__models:
            return self.__models[model_name].parameters()
    
    
if __name__ == '__main__':
    pass