import enum
import random, os, signal, time
# from utils import *
import json
from collections import Counter, deque
import copy
import traceback
from itertools import combinations
import itertools
import numpy as np
# full_input = {'log':[]}
envs = dict()
from rlcard.games.tractors.env.utils import *

###############################################################
# 牌面表示：数字
# h:红桃 d:方片 s:黑桃 c:草花 
# (0-h1 1-d1 2-s1 3-c1) (4-h2 5-d2 6-s2 7-c2) ... 52-joker 53-Joker (54-h1 55-d1 56-s1 57-c1) ... 106-joker 107-Joker
# 请注意：10记为0
# 共2副108张
###############################################################
# __CARDSCALE__ = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
# __SUITSET__ = ['s','h','c','d']# h:红桃 d:方片 s:黑桃 c:草花 
# __MAJOR__ = ['jo', 'Jo']#小王 大王
# __POINT__ = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

# 预计算牌面字符 -> 索引的映射，避免热路径上 O(n) 的 list.index() 调用
__CARDSCALE_IDX__ = {c: i for i, c in enumerate(__CARDSCALE__)}
__SUITSET_IDX__ = {s: i for i, s in enumerate(__SUITSET__)}


def _poker2num_value(poker):
    """牌字符 -> 单副牌内编号 (0-53)，不含两副牌偏移。"""
    if poker[0] == "j":
        return 52
    if poker[0] == "J":
        return 53
    return __CARDSCALE_IDX__[poker[1]] * 4 + __SUITSET_IDX__[poker[0]]


class tractorGame():
    def __init__(self):
        self.errored = [[] for _ in range(__PLAYER_COUNT__)]
        self.Major = __MAJOR__.copy()
        self.get_score = 0 # 闲家得分，每次step前清空一下该值
        self.get_score_pok=[]# 得分牌，每次step前清空一下该值
        self.globalInfo = {"stage":"gameend"} # 未确定主花色前为空
        self.play_seq = []#玩家出牌序列  数组元素数据结构：[player, card, card_type]
        
        self.step_count = -1 # 一局里面的步数, -1是未在对局中
        self.player_hand_cards = [[] for _ in range(__PLAYER_COUNT__)] # 每位玩家手牌
        self.player_hand_decks = [[] for _ in range(__PLAYER_COUNT__)] # 每位玩家手牌
        self.player_played_cards = [[] for _ in range(__PLAYER_COUNT__)] # 各玩家已经出过的牌
        self.player_level = ['2' for _ in range(__PLAYER_COUNT__)] # 各玩家当前的级数
        self.total_score = [0 for _ in range(__PLAYER_COUNT__)] # 玩家的总分数
        self.pointorder = __POINT__.copy()
        self.logs = []
        self.erro_code = 0
        self._final_ended = False
        self._eval_deals = None      # 固定发牌数据 {level: [deal, ...]}
        self._eval_deal_idx = {}     # 每个级数已消费到第几局的游标
      
    def num2Poker(self, num): # num: int-[0,107]
        # Already a poker
        if type(num) is str and (num in self.Major or (num[0] in __SUITSET__ and num[1] in __CARDSCALE__)):
            return num
        # Locate in 1 single deck
        NumInDeck = num % 54
        # joker and Joker:
        if NumInDeck == 52:
            return "jo"
        if NumInDeck == 53:
            return "Jo"
        # Normal cards:
        pokernumber = __CARDSCALE__[NumInDeck // 4]
        pokersuit = __SUITSET__[NumInDeck % 4]
        return pokersuit + pokernumber

     # poker: str
     # deck: int list
    def Poker2Num(self, poker, deck):
        NumInDeck = _poker2num_value(poker)
        return NumInDeck if NumInDeck in deck else NumInDeck + 54
        
        
    #pokers是个二维数组，每个元素是个list，list是个扑克牌列表
    def PokerList2Num(self, pokers, deck): # poker: str
        numDesk = []
        # deck 为两副牌中的物理编号(0-107, 互不相同)；同一牌值最多出现两次(单副/双副)。
        # 用 set + 每动作内 consumed 计数替代原来的 _deck[:] 拷贝 + list.remove，避免 O(n) 查找与删除。
        deck_set = set(deck)
        for _pokers in pokers:
            rule_poker = []
            consumed = {}
            for poker in _pokers:
                NumInDeck = _poker2num_value(poker)
                if consumed.get(NumInDeck, 0) == 0:
                    p = NumInDeck if NumInDeck in deck_set else NumInDeck + 54
                else:
                    p = NumInDeck + 54
                consumed[NumInDeck] = consumed.get(NumInDeck, 0) + 1
                rule_poker.append(p)
            numDesk.append(rule_poker)
        return numDesk

    #pokers 是个扑克牌列表 环境编码
    def Pokers2Num(self, pokers, deck): # poker: str
        numDesk = []
        _deck = deck.copy()
        for poker in pokers:
            NumInDeck = _poker2num_value(poker)
            p = NumInDeck if NumInDeck in _deck else NumInDeck + 54
            numDesk.append(p)
            if p not in _deck:
                pass
            _deck.remove(p)
            
        return numDesk
    
    # 确定主牌
    def setMajor(self, major, level):
        self.Major = __MAJOR__.copy()
        major_ = str(major)
        self.pointorder = __POINT__.copy()
        if major_ != 'n': # 非无主
            self.Major = [major_+point for point in self.pointorder if point != level] + [suit + level for suit in __SUITSET__ if suit != major_] + [major_ + level] + self.Major
        else: # 无主
            self.Major = [suit + level for suit in __SUITSET__] + self.Major
        
        #数字牌
        self.MajorCards = self.Pokers2Num(self.Major, list(range(0, 108)))
        #两副牌
        self.MajorCards = self.MajorCards + [item + 54 for item in self.MajorCards]
        self.pointorder.remove(level)
    ###############################################################
    # 报错模块
    # 报错类型：
    # INVALID_POKERID 数字不在0~107中
    # NOT_YOUR_POKER 打出的牌不是自己的牌
    # INVALID_POKERTYPE 非法牌型
    # ILLEGAL_MOVE 错误行动（出牌、报主、反主）
    ###############################################################
    

    def setError(self, player, reason): # player: int-[0,3] reason: str
        if player == -1:
            raise ValueError("SYSTEM_ERROR")
        
        
        endingScores = [0]*__PLAYER_COUNT__
        for i in range(__PLAYER_COUNT__):
            if i == player:
                endingScores[i] = -3 # 出错会被额外扣分
            elif i == (player + 2) % __PLAYER_COUNT__:
                endingScores[i] = 0
            else:
                endingScores[i] = 1
        
        self.errored[player].append([reason, endingScores])
        self.erro_code = 1
        # print(json.dumps({
        #     "command": "finish",
        #     "content": endingScores,
        #     "display": {
        #         "currplayer": player,
        #         "score": endingScores,
        #         "error": self.errored
        #     }
        # }))
        
        
        #test code
        pass
        # for pid,_ in envs.items():
        #     if pid != os.getpid():
        #         os.kill(pid, signal.SIGTERM)

        # response = run(self)
        # self.step(response)

    ###############################################################
    # 牌型鉴定模块
    # 牌型（除甩牌外） 
    # 单牌 1张任意 single
    # 对子 2张相同花色相同数字 pair
    # 连对（拖拉机）大小上连续 tractor
    # 甩牌（怀疑且需要环境判定） suspect
    ###############################################################

    # return: pokertype(str)
    def checkPokerType(self, poker, level): #poker: list[int]
        if len(poker) == 0: # 空出牌不构成任何合法牌型，避免 poker[0] 越界
            return __SUSPECT__
        poker = [self.num2Poker(p) for p in poker] if type(poker[0])==int else poker
        if len(poker) == 1:
            return __SINGLE__ #一张牌必定为单牌
        if len(poker) == 2:
            if poker[0] == poker[1]:
                return __PAIR__ #同点数同花色才是对子
            else:
                return __SUSPECT__ #怀疑是甩牌
        if len(poker) % 2 == 0: #其他情况下只有偶数张牌可能是整牌型（连对）
        # 连对：每组两张；各组花色相同；各组点数在大小上连续(需排除大小王和级牌)
            count = Counter(poker)
            if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2 and len(poker) == 4:
                return __TRACTOR__
            elif "jo" in count.keys() or "Jo" in count.keys(): # 排除大小王
                return __SUSPECT__
            for v in count.values(): # 每组两张
                if v != 2:
                    return __SUSPECT__
            pointpos = []
            suit = list(count.keys())[0][0] # 花色相同
            for k in count.keys():
                if k[0] != suit or k[1] == level: # 排除级牌
                    return __SUSPECT__
                pointpos.append(self.pointorder.index(k[1])) # 点数在大小上连续
            pointpos.sort()
            for i in range(len(pointpos)-1):
                if pointpos[i+1] - pointpos[i] != 1:
                    return __SUSPECT__
            return __TRACTOR__ # 说明是拖拉机
        
        return __SUSPECT__

    # 甩牌判定功能函数
    # return: ExistBigger(True/False)
    # 给定一组常规牌型，鉴定其他三家是否有同花色的更大牌型
    def checkBigger(self, poker, own, currplayer, level, major):
    # poker: 给定牌型 list
    # own: 各家持牌 list
        tyPoker = self.checkPokerType(poker, level)
        poker = [self.num2Poker(p) for p in poker] if type(poker[0])==int else poker
        assert tyPoker != __SUSPECT__, "Type 'throw' should contain common types"
        own_pok = [[self.num2Poker(num) for num in hold] for hold in own]
        if poker[0] in self.Major: # 主牌型应用主牌压
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                major_pok = [pok for pok in hold if pok in self.Major]
                count = Counter(major_pok)
                if len(poker) <= 2:
                    if poker[0][1] == level and poker[0][0] != major: # 含有副级牌要单算
                        if major == 'n': # 无主
                            for k,v in count.items(): 
                                if (k == 'jo' or k == 'Jo') and v >= len(poker):
                                    return True
                        else:
                            for k,v in count.items():
                                if (k == 'jo' or k == 'Jo' or k == major + level) and v >= len(poker):
                                    return True
                    else: 
                        for k,v in count.items():
                            if self.Major.index(k) > self.Major.index(poker[0]) and v >= len(poker):
                                return True
                else: # 拖拉机
                    if "jo" in poker: # 必定是大小王连对
                        return False # 不可能被压
                    if len(poker) == 4 and "jo" in count.keys() and "Jo" in count.keys():
                        if count["jo"] == 2 and count["Jo"] == 2: # 大小王连对必压
                            return True
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if k != 'jo' and k != 'Jo' and k[1] != level and self.pointorder.index(k[1]) > self.pointorder.index(poker[-1][1]): # 大小王和级牌当然不会参与拖拉机
                                pos.append(self.pointorder.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False
        else: # 副牌甩牌
            suit = poker[0][0]
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                suit_pok = [pok for pok in hold if pok[0] == suit and pok[1] != level]
                count = Counter(suit_pok)
                if len(poker) <= 2:
                    for k, v in count.items():
                        if self.pointorder.index(k[1]) > self.pointorder.index(poker[0][1]) and v >= len(poker):
                            return True
                else:
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if self.pointorder.index(k[1]) > self.pointorder.index(poker[-1][1]):
                                pos.append(self.pointorder.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False

        return False

    # 甩牌是否可行
    # return: poker(最终实际出牌:list[str])、ilcnt(非法牌张数)
    # 如果甩牌成功，返回的是对甩牌的拆分(list[list])
    def checkThrow(self, poker, own, currplayer, level, major, check=False):
    # poker: 甩牌牌型 list[int]
    # own: 各家持牌 list
    # level & major: 级牌、主花色
        ilcnt = 0
        pok = [self.num2Poker(p) for p in poker] if type(poker[0])==int else poker
        outpok = []
        failpok = []
        count = Counter(pok)
        if check:
            if list(count.keys())[0] in self.Major: # 如果是主牌甩牌
                for p in count.keys():
                    if p not in self.Major:
                        self.setError(currplayer, "INVALID_POKERTYPE")
            else: # 是副牌
                suit = list(count.keys())[0][0] # 花色相同
                for k in count.keys():
                    if k[0] != suit:
                        self.setError(currplayer, "INVALID_POKERTYPE")
        # 优先检查整牌型（拖拉机）
        pos = []
        tractor = []
        suit = ''
        for k, v in count.items():
            if v == 2:
                if k != 'jo' and k != 'Jo' and k[1] != level: # 大小王和级牌当然不会参与拖拉机
                    pos.append(self.pointorder.index(k[1]))
                    suit = k[0]
        if len(pos) >= 2:
            pos.sort()
            tmp = []
            suc_flag = False
            for i in range(len(pos)-1):
                if pos[i+1]-pos[i] == 1:
                    if not suc_flag:
                        tmp = [suit + self.pointorder[pos[i]], suit + self.pointorder[pos[i]], suit + self.pointorder[pos[i+1]], suit + self.pointorder[pos[i+1]]]
                        del count[suit + self.pointorder[pos[i]]]
                        del count[suit + self.pointorder[pos[i+1]]] # 已计入拖拉机的，从牌组中删去
                        suc_flag = True
                    else:
                        tmp.extend([suit + self.pointorder[pos[i+1]], suit + self.pointorder[pos[i+1]]])
                        del count[suit + self.pointorder[pos[i+1]]]
                elif suc_flag:
                    tractor.append(tmp)
                    suc_flag = False
            if suc_flag:
                tractor.append(tmp)
        # 对牌型作基础的拆分 
        for k,v in count.items(): 
            outpok.append([k for i in range(v)])
        outpok.extend(tractor)

        if check:
            for poktype in outpok:
                if self.checkBigger(poktype, own, currplayer, level, major): # 甩牌失败
                    ilcnt += len(poktype)
                    failpok.append(poktype)  
        
        if ilcnt > 0:
            finalpok = []
            kmin = ""
            for poktype in failpok:
                getmark = poktype[-1] 
                if kmin == "":
                    finalpok = poktype
                    kmin = getmark
                elif kmin in self.Major: # 主牌甩牌
                    if self.Major.index(getmark) < self.Major.index(kmin):
                        finalpok = poktype
                        kmin = getmark
                else: # 副牌甩牌
                    if self.pointorder.index(getmark[1]) < self.pointorder.index(kmin[1]):
                        finalpok = poktype
                        kmin = getmark
            finalpok = [[finalpok[0]]]
        else: 
            finalpok = outpok

        return finalpok, ilcnt 

    ###############################################################
    # 发牌决定模块
    # 先确定好发给每个人的25张牌，再进行逐一发牌
    ###############################################################

    def set_eval_deals(self, deals):
        """注入固定发牌数据：{level: [{'allocation':..., 'publiccard':...}, ...]}"""
        self._eval_deals = deals
        self._eval_deal_idx = {level: 0 for level in deals}

    def _next_eval_deal(self, level):
        """按级数顺序取下一局预生成的发牌；数据缺失或用尽则返回 None（回退随机发牌）。"""
        if not self._eval_deals or level not in self._eval_deals:
            return None
        deals = self._eval_deals[level]
        idx = self._eval_deal_idx.get(level, 0)
        if idx >= len(deals):
            return None
        deal = deals[idx]
        self._eval_deal_idx[level] = idx + 1
        return deal

    def initGame(self):
        seedRandom = str(random.randint(0, 2147483000))
        full_input = {}

        if "initdata" not in full_input:
            full_input["initdata"] = {}
        # try:
        #     full_input["initdata"] = json.loads(full_input["initdata"])
        # except Exception:
        #     pass
        if type(full_input["initdata"]) is not dict:
            full_input["initdata"] = {}

        if "seed" in full_input["initdata"]:
            seedRandom = full_input["initdata"]["seed"] 
        
        random.seed(seedRandom)

        # 固定发牌注入：按当前级数取预生成的发牌（用于评估可复现）。
        # 首局 globalInfo 尚无 level，默认按 "2" 取。
        eval_deal = self._next_eval_deal(self.globalInfo.get("level", "2"))
        if eval_deal is not None:
            return full_input, seedRandom, eval_deal["allocation"], eval_deal["publiccard"]

        if "allocation" in full_input["initdata"]:
            allocation = full_input["initdata"]["allocation"]
        else:
            '''每局使用两副牌共108张，每人分得25张手牌，剩余8张为底牌，4人两两一队。'''
            allo = list(range(108))
            rng = np.random.default_rng()
            rng.shuffle(allo)
            allocation = []
            for i in range(__PLAYER_COUNT__):
                allocation.append(allo[i*__HAND_CARD_NUM__:(i+1)*__HAND_CARD_NUM__])
        if "publiccard" in full_input["initdata"]:
            publiccard = full_input["initdata"]["publiccard"]
        else:
            publiccard = allo[__PLAYER_COUNT__*__HAND_CARD_NUM__:__PLAYER_COUNT__*__HAND_CARD_NUM__ + 8]
        
        return full_input, seedRandom, allocation, publiccard
    
    def reset(self):
        # 刚开局
        if self.globalInfo["stage"] == "gameend":
            full_input, seedRandom, allocation, publiccard = self.initGame()
            initdata = {}
            initdata["allocation"] = allocation
            initdata["seed"] = seedRandom
            initdata["publiccard"] = publiccard
            
            if "first_round" not in self.globalInfo:
                self.globalInfo["first_round"] = True
                self.globalInfo["level"] = "2"
            else:
                self.globalInfo["first_round"] = False
                
                
            
            self.globalInfo["allocation"] = initdata["allocation"]
            self.globalInfo["seed"] = initdata["seed"]
            self.globalInfo["publiccard"] = initdata["publiccard"]
            self.globalInfo['playedcard'] = []#已出牌
            
            # 跨局状态重置：否则新一局会带入上一局的出牌记录/步数/错误
            self.player_played_cards = [[] for _ in range(__PLAYER_COUNT__)]
            self.play_seq = []
            self.step_count = -1
            self.errored = [[] for _ in range(__PLAYER_COUNT__)]
            self.erro_code = 0
            
            
            #初始化banking
            if "banking" not in self.globalInfo: # 没有规定摸牌方
                first = 0
                banking = {
                "called": [],
                "major": "",
                "banker": -1,
                'banker_last': 0,
                } 
            else:
                first = self.globalInfo["banking"]["banker"]
                banking = {
                "called": [],
                "major": "",
                "banker": first,
                'banker_last': self.globalInfo["banking"]["banker_last"],
                } 
                
            self.globalInfo["banking"] = banking
            self.globalInfo["playerpos"] = first
            self.globalInfo["first_dealer"] = first

            # 轮流发牌：每位玩家每次发一张牌，发完一张后进入叫主阶段
            self.player_hand_cards = [[] for _ in range(__PLAYER_COUNT__)]
            self.player_hand_decks = [[] for _ in range(__PLAYER_COUNT__)]
            self.globalInfo["dealt"] = 0
            self.globalInfo["deal_count"] = [0] * __PLAYER_COUNT__
            self.globalInfo["bid_seq"] = []
            
            # 先发第一张牌给 first，随后每次叫主后继续发下一张
            self._deal_to(first)
            self.globalInfo["stage"] = "bid"
            
        elif self.globalInfo["stage"] == "finalend":
            # 新大局必须在发牌前清除旧等级和庄家；普通 gameend 则保留升级结果。
            # 保留固定发牌数据及消费游标、日志开关等环境配置。
            self.globalInfo = {"stage": "gameend"}
            self.player_level = ['2' for _ in range(__PLAYER_COUNT__)]
            self.total_score = [0 for _ in range(__PLAYER_COUNT__)]
            self.Major = __MAJOR__.copy()
            self.MajorCards = []
            self.pointorder = __POINT__.copy()
            self.get_score = 0
            self.get_score_pok = []
            self._final_ended = False
            self.reset()
        
    ###############################################################
    # 报主和反主模块
    # 接收每回合发牌的报主和反主信息
    # 类似于地主和叫分，主花色和庄家会作为常规信息单独提供给玩家
    ###############################################################

    # return Banking
    def checkBanker(self, repo, level, currplayer, banking, first_round): 
    # repo: int(player's response) 
    # level: str(current level) 
    # currplayer: int(current playerid)
    # banking: dict_object (called, snatched, major, banker)
        newbanking = banking
        if len(repo) == 1: # 单张报主
            if len(banking["called"]) > 0: # 已报过主
                self.setError(currplayer, "ILLEGAL_MOVE")
            poker = self.num2Poker(repo[0])
            if poker[1] != level: # 不是级牌
                self.setError(currplayer, "ILLEGAL_MOVE")
            banking["called"].append(currplayer)
            newbanking["major"] = poker[0]
            if first_round:
                newbanking["banker"] = currplayer
                
            return newbanking
        if len(repo) == 2: # 对子反主
            if  len(banking["called"]) == 0 or len(banking["called"]) == __PLAYER_COUNT__-1: # 还未报主或已经反主
                self.setError(currplayer, "ILLEGAL_MOVE")
            poker = [self.num2Poker(repo[0]), self.num2Poker(repo[1])]
            if poker[0] != poker[1]: # 不是对子
                self.setError(currplayer, "ILLEGAL_MOVE")
            if poker[0][1] != level: # 不是级牌
                if poker[0] == "jo" or poker[0] == "Jo": # 是大小王
                    banking["called"].append(currplayer)
                    newbanking["major"] = "n"
                    if first_round:
                        newbanking["banker"] = currplayer
                        
                    return newbanking
                self.setError(currplayer, "ILLEGAL_MOVE")
            banking["called"].append(currplayer)
            newbanking["major"] = poker[0][0]
            if first_round:
                newbanking["banker"] = currplayer
            return newbanking
        return newbanking
        

    ###############################################################
    # 出牌裁判模块
    # 包含针对常规出牌和甩牌的裁判模块
    # self.checkLegalMove: 每名玩家行动后判定行动是否合法
    # self.checkWinner: 一轮行动结束后找该轮最大的玩家
    # * 行动前，会统一判断玩家出的牌是否在自己的手牌中（包括报主与反主）
    ###############################################################

    # 罚分
    def Punish(self, currplayer, banker, score):
        if (currplayer - banker) % 2 != 0: # 当前玩家不是庄家
            self.get_score -= score
        else: # 庄家罚分，加到闲家上
            self.get_score += score

    #获取牌堆中所有符合牌型
    def getTypePoke(self, own_pok, level, type = [__SINGLE__, __PAIR__, __TRACTOR__, __SUSPECT__]):        
        rule_poks = {tp:[] for tp in type}#poker格式
        
        # own_pok = ['d4','d4','d5','d5','d7','d7','d8','d8']#test
        
        count = Counter(own_pok)
        if __SINGLE__ in type:
            rule_poks[__SINGLE__].extend([[p] for p in own_pok])
            
        if __PAIR__ in type:
            for k, v in count.items():
                if v == 2:# and k != 'jo' and k != 'Jo':
                    rule_poks[__PAIR__].append([k,k])
                    
        if __TRACTOR__ in type:
            tr = self.parseTractorPoker(own_pok, level)
            rule_poks[__TRACTOR__].extend(tr)

        if __SUSPECT__ in type:
            tr = self.parseSuspectPoker(own_pok, level, self.globalInfo["banking"]["major"])
            if tr:
                max_len = max(len(pok) for pok in tr)
                tr = [pok for pok in tr if len(pok) >= max_len - 2]
            rule_poks[__SUSPECT__].extend(tr)

        return rule_poks
    # 检查是否有可应手牌型
    # return: Exist(True/False)
    def checkRes(self, poker, own, level): # poker: list[int]
        pok = [self.num2Poker(p) for p in poker]
        own_pok = [self.num2Poker(p) for p in own]
        if pok[0] in self.Major:
            major_pok = [pok for pok in own_pok if pok in self.Major]
            count = Counter(major_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else: # 拖拉机 
                pos = []
                for k, v in count.items():
                    if v == 2:
                        if k != 'jo' and k != 'Jo' and k[1] != level: # 大小王和级牌当然不会参与拖拉机
                            pos.append(self.pointorder.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        else:
            suit = pok[0][0]
            suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
            count = Counter(suit_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else:
                pos = []
                for k, v in count.items():
                    if v == 2:
                        pos.append(self.pointorder.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        return False

    #play_pok 不是甩牌牌型
    def checkResUnSuspect(self, play_pok, own_pok, level): # poker: list[int]
        poker_len = len(play_pok)
        suit = play_pok[0][0]
        all_hands = [[] for _ in range(__WRONG__)]
        
        # typoker = self.checkPokerType(play_pok, level)
        #出的是主牌
        if play_pok[0] in self.Major:
            major_pok = [pok for pok in own_pok if pok in self.Major]
            my_pok_count = Counter(major_pok)   

            #单张或对子         
            if poker_len <= 2:
                card_type = __SINGLE__ if poker_len == 1 else __PAIR__
                ret = []
                
                for k,v in my_pok_count.items():
                    if v >= poker_len:
                        ret.append([k]*poker_len)
                        
                if len(major_pok) < poker_len:
                    unmajor_pok = [pok for pok in own_pok if pok not in major_pok]
                    ret = [major_pok+list(pok) for pok in list(combinations(unmajor_pok, poker_len-len(major_pok)))]
                    card_type = __DISCARD__
                elif len(ret) == 0:              
                    ret.extend(list(combinations(major_pok, poker_len)))
                
                all_hands[card_type].extend(ret)
                return all_hands
                
            # 主牌拖拉机
            else: 
                # deck_Major = [pok for pok in own_pok if pok in self.Major]
                deck_Major = major_pok
                card_type = __TRACTOR__
                ret = self.parseTractorPoker(deck_Major, level, poker_len)
                
                if len(ret) > 0:
                    all_hands[card_type].extend(ret)
                    return all_hands
                
                #没有拖拉机，看有没有对子
                pairspok = [p for p,v in my_pok_count.items() if v == 2]
                #对子数>=出牌数
                if len(pairspok) >= poker_len//2:
                    combpairs = list(combinations(pairspok, poker_len//2))
                    for pairs_pok in combpairs: ret.append(pairs_pok*2)
                        
                #对子数不够，用对子+单牌组合
                else:
                    singlepok = [p for p,v in my_pok_count.items() if v == 1]                      
                    fixed_pok = pairspok*2

                    #(对子+单牌)的数量<出牌数，则将所有的牌加进来                        
                    if len(fixed_pok) + len(singlepok) < poker_len:
                        fixed_pok += singlepok
                        singlepok = [pok for pok in own_pok if pok not in deck_Major]
                        card_type = __DISCARD__
                        
                    ret = [fixed_pok + list(poks) for poks in combinations(singlepok, poker_len-len(fixed_pok))]
                
                all_hands[card_type].extend(ret)
                return all_hands
            
                            
                
        #出的是副牌
        else:
            suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
            # print('suit_pok:',suit_pok)
            my_pok_count = Counter(suit_pok)

            #单张或对子
            if poker_len <= 2:
                card_type = __SINGLE__ if poker_len == 1 else __PAIR__
                ret = []
                
                for k,v in my_pok_count.items():
                    if v >= poker_len:
                        ret.append([k]*poker_len)

                if len(suit_pok) < poker_len:
                    unsuit_pok = [pok for pok in own_pok if pok[0] != suit or pok[1] == level]
                    ret = [suit_pok+list(pok) for pok in list(combinations(unsuit_pok, poker_len-len(suit_pok)))]
                    card_type = __DISCARD__
                elif len(ret) == 0:
                    ret.extend(list(combinations(suit_pok, poker_len)))
                    
                all_hands[card_type].extend(ret)
                return all_hands
            
            #副牌拖拉机
            else:
                card_type = __TRACTOR__
                ret = self.parseTractorPoker(suit_pok, level, poker_len)
                
                if len(ret) > 0:                    
                    all_hands[card_type].extend(ret)
                    return all_hands
                
                #没有副拖拉机，看有没有主牌拖拉机
                deck_Major = [pok for pok in own_pok if pok in self.Major]
                ret = self.parseTractorPoker(deck_Major, level, poker_len)
                
                if len(ret) > 0:
                    all_hands[card_type].extend(ret)
                    return all_hands
                
                #其他牌
                ret = []
                pairspok = [p for p,v in my_pok_count.items() if v == 2]                    
                #对子数>=出牌数
                if len(pairspok) >= poker_len//2:
                    combpairs = list(combinations(pairspok, poker_len//2))
                    for pairs_pok in combpairs: ret.append(pairs_pok*2)
                    
                #对子数不够，用对子+单牌组合
                else:
                    singlepok = [p for p,v in my_pok_count.items() if v == 1]
                    fixed_pok = pairspok*2

                    #(对子+单牌)的数量<出牌数，则将所有的牌加进来
                    if len(fixed_pok) + len(singlepok) < poker_len:
                        fixed_pok += singlepok
                        singlepok = [pok for pok in own_pok if pok[0] != suit or pok[1] == level]
                        card_type = __DISCARD__
                    ret.extend([fixed_pok + list(poks) for poks in combinations(singlepok, poker_len-len(fixed_pok))])
                
                all_hands[card_type].extend(ret)
                return all_hands

    
    # return outpok(The actual move if the move is legal; If illegal, report error)
    def checkLegalMove(self, poker, level, major, currplayer, history, own, banker): # own: All players' hold before this move
    # poker: list[int] player's move
    # history: other players' moves in the current round: list[list]
        pok = poker
        hist = [[self.num2Poker(p) for p in move] for move in history]
        outpok = pok
        own_pok = [self.num2Poker(p) for p in own[currplayer]]
        if len(history) == 0: # The first move in a round
            # Player can only throw in the first round
            typoker = self.checkPokerType(poker, level)
            if typoker == __SUSPECT__:
                outpok_s, ilcnt = self.checkThrow(poker, own, currplayer, level, major, True)
                if ilcnt > 0:
                    self.Punish(currplayer, banker, ilcnt*10)
                outpok = [p for poktype in outpok_s for p in poktype] # 符合交互模式，把甩牌展开
        else:
            tyfirst = self.checkPokerType(history[0], level)
            if len(poker) != len(history[0]):
                self.setError(currplayer, "ILLEGAL_MOVE")
            if tyfirst == __SUSPECT__: # 这里own不一样了，但是可以不需要check
                outhis, ilcnt = self.checkThrow(history[0], own, currplayer, level, major, check=False)
                # 甩牌不可能失败，因此只存在主牌毙或者贴牌的情形，且不可能有应手
                # 这种情况下的非法行动：贴牌不当
                # outhis是已经拆分好的牌型(list[list])
                # flathis = [p for poktype in outhis for p in poktype]
                if outhis[0][0] in self.Major: 
                    major_pok = [p for p in pok if p in self.Major]
                    if len(major_pok) != len(poker): # 这种情况下，同花(主牌)必须已经贴完
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(major_hold):
                            self.setError(currplayer, "ILLEGAL_MOVE")
                    else: #全是主牌
                        outhis.sort(key=lambda x: len(x), reverse=True) # 牌型从大到小来看
                        major_hold = [p for p in own_pok if p in self.Major]
                        matching = True
                        if self.checkPokerType(outhis[0], level) == __TRACTOR__: # 拖拉机来喽
                            divider, _ = self.checkThrow(poker, [[]], currplayer, level, major, check=False)
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self.checkThrow(major_hold, [[]], currplayer, level, major, check=False)
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for poktype in outhis: # 可以使用这种方法的原因在于同一组花色/主牌可组成的牌型数量太少，不会出现多解
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching: # 不匹配，看手牌是否存在应手
                                res_ex = True
                                for chtype in outhis:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else: 
                                        res_ex = False
                                        break
                                if res_ex: # 存在应手，说明贴牌不当
                                    self.setError(currplayer, "ILLEGAL_MOVE")
                                else: # 存在应手，继续检查
                                    pair_own = sum([len(x) for x in own_divide if len(x) >= 2])
                                    pair_his = sum([len(x) for x in outhis if len(x) >= 2])
                                    pair_pok = sum([len(x) for x in divider if len(x) >= 2])
                                    if pair_pok < min(pair_own, pair_his):
                                        self.setError(currplayer, "ILLEGAL_MOVE")
                else:
                    suit = hist[0][0][0]
                    suit_pok = [p for p in pok if p not in self.Major and p[0] == suit]
                    if len(suit_pok) != len(poker): # 这种情况下，同花(主牌)必须已经贴完
                        suit_hold = [p for p in own_pok if p not in self.Major and p[0] == suit]
                        if len(suit_pok) != len(suit_hold):
                            self.setError(currplayer, "ILLEGAL_MOVE")
                    else: 
                        outhis.sort(key=lambda x: len(x), reverse=True) # 牌型从大到小来看
                        suit_hold = [p for p in own_pok if p not in self.Major and p[0] == suit]
                        matching = True
                        if self.checkPokerType(outhis[0], level) == __TRACTOR__: # 拖拉机来喽
                            divider, _ = self.checkThrow(poker, [[]], currplayer, level, major, check=False)
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self.checkThrow(suit_hold, [[]], currplayer, level, major, check=False)
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for poktype in outhis: # 可以使用这种方法的原因在于同一组花色/主牌可组成的牌型数量太少，不会出现多解
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching: # 不匹配，看手牌是否存在应手
                                res_ex = True
                                for chtype in outhis:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else: 
                                        res_ex = False
                                        break
                                if res_ex: # 存在应手，说明贴牌不当
                                    self.setError(currplayer, "ILLEGAL_MOVE")
                                else: # 存在应手，继续检查
                                    pair_own = sum([len(x) for x in own_divide if len(x) >= 2])
                                    pair_his = sum([len(x) for x in outhis if len(x) >= 2])
                                    pair_pok = sum([len(x) for x in divider if len(x) >= 2])
                                    if pair_pok < min(pair_own, pair_his):
                                        self.setError(currplayer, "ILLEGAL_MOVE")
                            # 到这里关于甩牌贴牌的问题基本上解决，是否存在反例还有待更详细的讨论

            else: # 常规牌型
            # 该情形下的非法行动：(1) 有可以应手的牌型但贴牌或用主牌毙 (2) 贴牌不当(有同花不贴/拖拉机有对子不贴)
                if self.checkRes(history[0], own[currplayer], level): #(1) 有应手但贴牌或毙
                    if self.checkPokerType(poker, level) != tyfirst:
                        self.setError(currplayer,"ILLEGAL_MOVE")
                    if hist[0][0] in self.Major and pok[0] not in self.Major:
                        self.setError(currplayer,"ILLEGAL_MOVE")
                    if hist[0][0] not in self.Major and (pok[0] in self.Major or pok[0][0] != hist[0][0][0]):
                        self.setError(currplayer, "ILLEGAL_MOVE") 
                elif self.checkPokerType(poker, level) != tyfirst: #(2) 贴牌不当: 有同花不贴完/同花色不跟整牌型
                    own_pok = [self.num2Poker(p) for p in own[currplayer]]
                    if hist[0][0] in self.Major:
                        major_pok = [p for p in pok if p in self.Major]
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(poker): # 这种情况下，同花(主牌)必须已经贴完
                            if len(major_pok) != len(major_hold):
                                self.setError(currplayer, "ILLEGAL_MOVE")
                        else: # 完全是主牌
                            count = Counter(major_hold)
                            if tyfirst == __PAIR__:
                                for v in count.values():
                                    if v == 2:
                                        self.setError(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == __TRACTOR__:
                                trpairs = len(history[0])//2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if pkpairs < trpairs and pkpairs < hdpairs: # 并不是所有对子都用上了
                                    self.setError(currplayer, "ILLEGAL_MOVE")

                    else: 
                        suit = hist[0][0][0]
                        suit_pok = [p for p in pok if p[0] == suit and p not in self.Major]
                        suit_hold = [p for p in own_pok if p[0] == suit and p not in self.Major]
                        if len(suit_pok) != len(poker):    
                            if len(suit_pok) != len(suit_hold):
                                self.setError(currplayer, "ILLEGAL_MOVE")
                        else: # 完全是同种花色
                            count = Counter(suit_hold)
                            if tyfirst == __PAIR__:
                                for v in count.values():
                                    if v == 2:
                                        self.setError(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == __TRACTOR__:
                                trpairs = len(history[0])//2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if pkpairs < trpairs and pkpairs < hdpairs: # 并不是所有对子都用上了
                                    self.setError(currplayer, "ILLEGAL_MOVE")
                        
        return outpok

    # 取一手牌中最强的一张，用于比较大小
    # 牌型合法性已保证同组牌花色/牌型一致，故最强牌即该组牌的代表牌
    # 注意：必须按「最大牌」而非「末位牌」比较，否则拖拉机等连对牌型的胜负会随出牌顺序变化
    def _strongestCard(self, cards, in_major):
        if in_major:
            return max(cards, key=lambda c: self.Major.index(c))
        return max(cards, key=lambda c: self.pointorder.index(c[-1]))

    # 在每轮最后一名玩家行动后触发判定，接收该轮历史行动及玩家本次出牌，判定胜方和分值
    # 对于甩牌，盖毙只判定最大牌型的大小
    # return winner(int: player ID)
    def checkWinner(self, history, currplayer, level, major, banker):
        histo = history + []
        hist = [[self.num2Poker(p) for p in x] for x in histo]
        score = 0 
        
        for his_cards in history:
            for card in his_cards:
                if (card>=16 and card<=19) or (card>=70 and card<=73):
                    score += 5
                    self.get_score_pok.append(card)
                elif (card>=36 and card<=39) or (card>=48 and card<=51) or (card>=90 and card<=93) or (card>=102 and card<=105):
                    score += 10
                    self.get_score_pok.append(card)
        

        win_seq = 0 # 获胜方在本轮行动中的顺位，默认为0
        win_move = hist[0] # 获胜方的出牌，默认为首次出牌
        tyfirst = self.checkPokerType(history[0], level)
        if tyfirst == __SUSPECT__: # 甩牌
            first_parse, _ = self.checkThrow(history[0], [[]], currplayer, level, major, check=False)
            first_parse.sort(key=lambda x: len(x), reverse=True)
            for i in range(1,__PLAYER_COUNT__):
                move_parse, r = self.checkThrow(history[i], [[]], currplayer, level, major, check=False)
                move_parse.sort(key=lambda x: len(x), reverse=True)
                move_cnt = [len(x) for x in move_parse]
                matching = True
                for poktype in first_parse: # 杀毙的前提是牌型相同
                    if move_cnt[0] >= len(poktype):
                        move_cnt[0] -= len(poktype)
                        move_cnt.sort(reverse=True)
                    else:
                        matching = False
                        break
                if not matching:
                    continue
                isMajor = True
                for j in range(len(hist[i])):
                    if hist[i][j] not in self.Major:
                        isMajor = False
                        break
                if not isMajor: # 副牌压主牌，算了吧
                    continue
                if win_move[0] not in self.Major and isMajor: # 主牌压副牌，必须的
                    win_move = hist[i]
                    win_seq = i
                # 两步判断后，只剩下hist[i]和win_move都是主牌的情况
                elif len(first_parse[0]) >= 4: # 有拖拉机再叫我checkThrow来
                    if major == 'n': # 如果这里无主，拖拉机只可能是对大小王，不可能有盖毙
                        continue
                    win_parse, s = self.checkThrow(history[win_seq], [[]], currplayer, level, major, check=False)
                    win_parse.sort(key=lambda x: len(x), reverse=True)
                    # 取最大牌型组里的最强牌比较，与出牌顺序无关
                    if self.Major.index(self._strongestCard(win_parse[0], True)) < \
                       self.Major.index(self._strongestCard(move_parse[0], True)):
                        win_move = hist[i]
                        win_seq = i
                else: 
                    steps = len(first_parse[0])
                    win_count = Counter(win_move)
                    win_max = 0
                    for k,v in win_count.items():
                        if v >= steps and self.Major.index(k) >= win_max: # 这里可以放心地这么做，因为是何种花色的副2不会影响对比的结果
                            win_max = self.Major.index(k)
                    move_count = Counter(hist[i])
                    move_max = 0
                    for k,v in move_count.items():
                        if v >= steps and self.Major.index(k) >= move_max:
                            move_max = self.Major.index(k)
                    if major == 'n': # 无主
                        if self.Major[win_max][1] == level:
                            if self.Major[move_max] == 'jo' or self.Major[move_max] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif move_max > win_max:
                            win_move = hist[i]
                            win_seq = i
                    elif self.Major[win_max][1] == level and self.Major[win_max][0] != major:
                        if (self.Major[move_max][0] == major and self.Major[move_max][1] == level) or self.Major[move_max] == "jo" or self.Major[move_max] == "Jo":
                            win_move = hist[i]
                            win_seq = i
                    elif win_max < move_max:#test code
                    # elif self.Major.index(win_max) < self.Major.index(move_max):
                        win_move = hist[i]
                        win_seq = i


        else: # 常规牌型
            #print("Common: Normal")
            for i in range(1, __PLAYER_COUNT__):
                if self.checkPokerType(history[i], level) != tyfirst: # 牌型不对
                    continue
                #print("check: Normal")
                if (hist[0][0] in self.Major and hist[i][0] not in self.Major) or (hist[0][0] not in self.Major and (hist[i][0] not in self.Major and hist[i][0][0] != hist[0][0][0])):
                # 花色不对，贴
                    continue
                elif win_move[0] in self.Major: # 主牌不会被主牌杀，且该分支内应手均为主牌
                    if hist[i][0] not in self.Major: # 副牌就不用看了
                        continue
                    #print("here")
                    # 取双方最强牌比较，与出牌顺序无关
                    win_best = self._strongestCard(win_move, True)
                    move_best = self._strongestCard(hist[i], True)
                    if major == 'n':
                        if win_best[1] == level:
                            if move_best == 'jo' or move_best == 'Jo': # 目前胜牌是级牌，只有大小王能压
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(move_best) > self.Major.index(win_best):
                            win_move = hist[i]
                            win_seq = i
                    else:
                        if win_best[0] != major and win_best[1] == level:
                            if (move_best[0] == major and move_best[1] == level) or move_best == 'jo' or move_best == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(move_best) > self.Major.index(win_best):
                            win_move = hist[i]
                            win_seq = i
                else: # 副牌存在被主牌压的情况
                    if hist[i][0] in self.Major: # 主牌，正确牌型，必压
                        win_move = hist[i]
                        win_seq = i
                    elif self.pointorder.index(self._strongestCard(win_move, False)[-1]) < \
                         self.pointorder.index(self._strongestCard(hist[i], False)[-1]):
                        win_move = hist[i]
                        win_seq = i
        # 找到获胜方，加分
        win_id = (currplayer - (__PLAYER_COUNT__ - 1) + win_seq) % __PLAYER_COUNT__
        self.Reward(score, win_id, banker)

        return win_id


    def Reward(self, score, currplayer, banker):
        if (currplayer - banker) % 2 != 0: # 非庄家方得分
            self.get_score += score
            

    # return endingScores(dict)
    def EndGame(self, banker, score):
        # 在换庄/升级之前冻结本局庄家；finalend 的提前退出也必须保留它。
        self.globalInfo["banking"]["banker_last"] = banker
        endingScores = [0]*__PLAYER_COUNT__
        bankers = [banker, (banker + 2) % __PLAYER_COUNT__]
        up_level_step = 1#晋升的级数
        bid_score = self.getLeastBidScore()
        banker_win = score < bid_score
        if score == 0: # 大光，庄家得3分
            up_level_step = 3
            for i in range(__PLAYER_COUNT__):
                if i in bankers:
                    self.total_score[i] += 3
                    endingScores[i] = 3
                else: 
                    endingScores[i] = -3
        elif score < 40.: # 小光，庄家得2分
            up_level_step = 2
            for i in range(__PLAYER_COUNT__):
                if i in bankers:
                    self.total_score[i] += 2
                    endingScores[i] = 2
                else:
                    endingScores[i] = -2
        elif score < 80: # 庄家得1分
            up_level_step = 1
            for i in range(__PLAYER_COUNT__):
                if i in bankers:
                    self.total_score[i] += 1
                    endingScores[i] = 1
                else:
                    endingScores[i] = -1
        elif score < 120: # 闲家得1分
            up_level_step = 0
            for i in range(__PLAYER_COUNT__):
                if i in bankers:
                    endingScores[i] = -1
                else:
                    self.total_score[i] += 1
                    endingScores[i] = 1
        elif score < 160: # 闲家得2分
            up_level_step = 1
            for i in range(__PLAYER_COUNT__):
                if i in bankers:
                    endingScores[i] = -2
                else:
                    self.total_score[i] += 2
                    endingScores[i] = 2
        else:  # 闲家得3分
            up_level_step = 2
            for i in range(__PLAYER_COUNT__):
                if i in bankers:
                    endingScores[i] = -3
                else:
                    self.total_score[i] += 3
                    endingScores[i] = 3
            
        # todo
        #print(f"banker_win:{banker_win}, banker:{banker}, score:{score}, endingScores:{endingScores}, level:{self.globalInfo['level']}")
        
        # 点数升级，更新庄家位置
        while True:
            if banker_win:
                banker_lv = self.player_level[banker]
                index = __POINT__.index(banker_lv)
                new_lv_index = index + up_level_step
                if new_lv_index >= len(__POINT__):
                    self.globalInfo["level"] = __POINT__[len(__POINT__)-1]
                    self.globalInfo["stage"] = 'finalend'
                    break
                new_lv_index = min(new_lv_index, len(__POINT__)-1)
                
                for bk in bankers:
                    self.player_level[bk] = __POINT__[new_lv_index]
                self.globalInfo["banking"]["banker"] = (banker+2)%__PLAYER_COUNT__
            else:
                idle_lv = self.player_level[(banker+1)%__PLAYER_COUNT__]
                index = __POINT__.index(idle_lv)
                new_lv_index = index + up_level_step
                if new_lv_index >= len(__POINT__):
                    self.globalInfo["level"] = __POINT__[len(__POINT__)-1]
                    self.globalInfo["stage"] = 'finalend'
                    break
                    
                new_lv_index = min(new_lv_index, len(__POINT__)-1)
                
                for bk in [(banker+1)%__PLAYER_COUNT__, (banker+3)%__PLAYER_COUNT__]:
                    self.player_level[bk] = __POINT__[new_lv_index]
                self.globalInfo["banking"]["banker"] = (banker+1)%__PLAYER_COUNT__
                
            self.globalInfo["level"] = __POINT__[new_lv_index]
            break
        
        #todo
        # print(f"new banker:{self.globalInfo["banking"]["banker"]}, level:{self.globalInfo['level']}")
        
        self.globalInfo["ending_score"] = endingScores
        return endingScores

    #获取玩家手牌
    def getPlayerHandCards(self, pos):
        return self.player_hand_cards[pos][:]
    
    #获取当前叫的主的花色
    def getMajorColor(self):
        return self.globalInfo["banking"]["major"]
    
    #获取当前主牌
    def getMajorCards(self, major=None, level=None):
        if major == None and level == None:
            return self.MajorCards[:]
        
        assert major != None
        assert level != None
        
        MajorCards = __MAJOR__.copy()
        if major != 'n' and major != '': # 非无主
            MajorCards = [major+point for point in __POINT__ if point != level] + [suit + level for suit in __SUITSET__ if suit != major] + [major + level] + MajorCards
        elif major == 'n': # 无主
            MajorCards = [suit + level for suit in __SUITSET__] + MajorCards
        elif major == '': # 不叫主
            MajorCards = [] 
        
        #数字牌
        MajorCards = self.Pokers2Num(MajorCards, list(range(0, 108)))
        #两副牌
        if len(MajorCards) > 0:
            MajorCards = MajorCards + [item + 54 for item in MajorCards]
        return MajorCards
            
    
    #获取当前的庄家
    def getBanker(self):
        return self.globalInfo["banking"]["banker"]
    
    #获取上局的庄家
    def getLastBanker(self):
        return self.globalInfo["banking"]["banker_last"]
    
    #获取当前打第几级
    def getLevel(self):
        return self.globalInfo["level"]
    
    #获取当前轮次的状态
    def getStage(self):
        return self.globalInfo["stage"]
    
    #获取当前发牌状态发的牌
    def getDeliver(self):
        return self.globalInfo["deliver"][:]
    
    #获取目前所有局的总结算分
    def getTotalScore(self, pos):
        return self.total_score[pos]
    
    #获取当局结算分
    def getEndingScore(self, pos):
        return self.globalInfo["ending_score"][pos]
    
    #获取局内分数
    def getGameScore(self):
        return self.globalInfo["game_score"]
    
    # 获取上一轮次的分数
    def getLastRoundScore(self):
        return self.get_score
    
    def getLastRoundScorePoke(self):
        return self.get_score>0 and self.get_score_pok or []
    
    
    #获取上一手的出牌信息
    def getLastPlayCards(self):
        if len(self.play_seq) > 0:
            return self.play_seq[-1]
        # if len(self.globalInfo["history"][1]) > 0:
        #     return self.globalInfo["history"][1][-1][:]
        # else:
        #     if len(self.globalInfo["history"][0]) > 0:
        #         return self.globalInfo["history"][0][-1][:]
        return []
        
    # 获取错误码
    def getErrorCode(self):
        return self.erro_code
    
    # 获取当前轮次最后报主的玩家位置
    def getCalled(self):
        called = self.globalInfo["banking"]["called"]
        return called[0] if len(called)>0 else -1
        
    #获取反主的玩家位置
    def getSnatched(self):
        called = self.globalInfo["banking"]["called"]
        return called[len(called)-1] if len(called)>0 else -1
        # return self.globalInfo["banking"]["snatched"]
    
    def getCalledList(self):
        return self.globalInfo["banking"]["called"]
        
    #获取底牌
    def getPublicCards(self):
        return self.globalInfo["publiccard"][:]
    
    #获取当前权位玩家位置
    def getPlayerPosition(self):
        return self.globalInfo["playerpos"]
    
    #获取玩家剩余手牌数量
    def getPlayerLeftHandCards(self, pos):
        return len(self.player_hand_cards[pos])
    
    #获取当前手牌中的主牌
    def getPlayerLeftMajorCards(self, pos):
        major_cards = [poker for poker in self.player_hand_decks[pos] if poker in self.MajorCards]
        return len(major_cards)
    
    #获取上一轮出牌历史
    def getLastRoundPlayHistory(self):
        return self.globalInfo["history"][0]
    
    #获取上一轮的首出牌人位置
    def getLastRoundPlaySeat(self):
        return self.globalInfo["history"][2]
    
    #获取上一轮的赢家位置
    def getLastRoundWinSeat(self):
        return self.globalInfo["history"][3]
    
    #获取当前轮出牌历史
    def getCurrRoundPlayHistory(self):
        return self.globalInfo["history"][1]
        # play_seq = []
        # for i in range(len(self.globalInfo["history"][1])):
        #     play_seq.append(self.globalInfo["history"][1][i][:])
        # return play_seq
    
    #获取所有出牌历史，返回索引
    def getPlayHistory(self):
        return self.play_seq
    
    #获取当前轮的首出牌人位置
    def getFristPlaySeat(self):
        return self.globalInfo["history"][3]
    
    #获取闲家升级所需的分数线，固定为80分
    def getLeastBidScore(self):
        return 80
    
    #叫主队列
    def getBidSeq(self):
        return self.globalInfo["bid_seq"][:]
    
    #获取最后叫主的人座位
    def getLastBidSeat(self):
        return self.globalInfo["bidsseat"]
    
    #获取当前可叫主的动作
    #return ['', s, h, c, d, n之一]
    def getLegalBidActions(self, seat):
        actions = ['']#'', s, h, c, d, n
        len_bid_qeq = len(self.globalInfo["bid_seq"])#[seat, suit]
        if len_bid_qeq == 0:
            handcards = self.getPlayerHandCards(seat)
            level_int = __CARDSCALE__.index(self.getLevel())*4
            levels_int = [[level_int, level_int + 54, 's'], [level_int+1, level_int+1 + 54, 'h'], [level_int+2, level_int+2 + 54, 'c'], [level_int+3, level_int+3 + 54, 'd']]
            for level_int_card in levels_int:
                if level_int_card[0] in handcards or level_int_card[1] in handcards:
                    actions.append(level_int_card[2])
            levels_int = [[52,52+54,'n'], [53,53+54,'n']]
            for level_int_card in levels_int:
                if level_int_card[0] in handcards and level_int_card[1] in handcards:
                    actions.append(level_int_card[2])
                        
        elif len_bid_qeq == 1:
            suit = self.globalInfo["bid_seq"][-1][1]
            if suit in __SUITSET__:
                handcards = self.getPlayerHandCards(seat)
                level_int = __CARDSCALE__.index(self.getLevel())*4
                levels_int = [[level_int, level_int + 54, 's'], [level_int+1, level_int+1 + 54, 'h'], [level_int+2, level_int+2 + 54, 'c'], [level_int+3, level_int+3 + 54, 'd'], [52,52+54,'n'], [53,53+54,'n']]
                for level_int_card in levels_int:
                    if level_int_card[0] in handcards and level_int_card[1] in handcards:
                        actions.append(level_int_card[2])
                        
        elif len_bid_qeq == 2:
            suit = self.globalInfo["bid_seq"][-1][1]
            if suit in __SUITSET__:
                handcards = self.getPlayerHandCards(seat)
                level_int = __CARDSCALE__.index(self.getLevel())*4
                levels_int = [[52,52+54,'n'], [53,53+54,'n']]
                for level_int_card in levels_int:
                    if level_int_card[0] in handcards and level_int_card[1] in handcards:
                        actions.append(level_int_card[2])
            
        return actions
    

    #分析得到拖拉机牌型（大小王和级牌当然不会参与拖拉机；可选单牌）
    #play_poker 已出牌, 整型    
    def parsePokerWithPlayCard(self, own_poker, play_poker, level, obtain_single=True):
    # poker: 甩牌牌型 list[int]
    # own: 各家持牌 list
    # level & major: 级牌、主花色
        if len(own_poker) != len(play_poker):
            self.setError(own_poker, "UNEQUAL_POKER_NUMBER")
            return []
        
        play_type = self.checkPokerType(play_poker, level)
        own_pok = own_poker if type(own_poker[0]) == str else [self.num2Poker(p) for p in own_poker]
        play_pok = own_poker if type(own_poker[0]) == str else [self.num2Poker(p) for p in own_poker]
        
        suit = play_pok[0][0]
        if play_type == __SINGLE__:
            pok = play_poker
        elif play_type == __PAIR__:
            pok = play_poker
        elif play_type == __TRACTOR__:
            pok = play_poker
        
        
        pok = pok[:]
        pok.sort(key=lambda x: __SUITSET__.index(x[0])*100 + __POINT__.index(x[1]) if x[1] != 'o' else 1000, reverse=False)
        
        tractor = [pok[0]]
        tractor_list = []
        outpok = []
        
        suit = pok[0][0]        
        i,j=1,1
        last_pos = self.pointorder.index(pok[0][1]) if pok[0][1] != level and pok[0][1] != 'o' else -1
        suit = pok[0][0]
        
        while i < len(pok):
            k = pok[i]
            if (k != 'jo' and k != 'Jo' and k[1] != level):
                if suit == k[0] and last_pos + (j//2) == self.pointorder.index(k[1]):
                    j += 1
                    tractor.append(k)
                else:
                    if len(tractor) >= 4 and len(tractor)%2 == 0:
                        outpok.append(tractor)
                        tractor_list.extend(tractor)
                    tractor = [k]
                    suit = k[0]
                    last_pos = self.pointorder.index(k[1])
                    j = 1
            i += 1
        
        # 对牌型作基础的拆分
        if obtain_single == True:
            single_pok = [[p] for p in pok if p not in tractor_list ]       
            outpok.extend(single_pok)
        finalpok = outpok
        
    # #分析得到拖拉机牌型（大小王和级牌当然不会参与拖拉机）
    # def parsePoker(self, pok, level):
    # # poker: 甩牌牌型 list[int]
    # # level & major: 级牌、主花牌
    #     outpok = []
    #     failpok = []
    #     count = Counter(pok)
    #     # 优先检查整牌型（拖拉机）
    #     pos = []
    #     tractor = []
    #     suit = ''
    #     for k, v in count.items():
    #         if v == 2:
    #             if k != 'jo' and k != 'Jo' and k[1] != level: # 大小王和级牌当然不会参与拖拉机
    #                 pos.append(self.pointorder.index(k[1]))
    #                 suit = k[0]
    #     if len(pos) >= 2:
    #         pos.sort()
    #         tmp = []
    #         suc_flag = False
    #         for i in range(len(pos)-1):
    #             if pos[i+1]-pos[i] == 1:
    #                 if not suc_flag:
    #                     tmp = [suit + self.pointorder[pos[i]], suit + self.pointorder[pos[i]], \
    #                            suit + self.pointorder[pos[i+1]], suit + self.pointorder[pos[i+1]]]
    #                     del count[suit + self.pointorder[pos[i]]]
    #                     del count[suit + self.pointorder[pos[i+1]]] # 已计入拖拉机的，从牌组中删去
    #                     suc_flag = True
    #                 else:
    #                     tmp.extend([suit + self.pointorder[pos[i+1]], suit + self.pointorder[pos[i+1]]])
    #                     del count[suit + self.pointorder[pos[i+1]]]
    #             elif suc_flag:
    #                 tractor.append(tmp)
    #                 suc_flag = False
    #         if suc_flag:
    #             tractor.append(tmp)
    #     # 对牌型作基础的拆分 
    #     for k,v in count.items(): 
    #         outpok.append([k for i in range(v)])
    #     outpok.extend(tractor)

    #     finalpok = outpok

    #     return finalpok 
    
    #分析得到拖拉机牌型（大小王和级牌当然不会参与拖拉机）
    def parseTractorPoker(self, poker, level, tractor_len=0):
    # poker: 甩牌牌型 list[int]
    # level & self.Major: 级牌、主花牌
        if len(poker) == 0:
            return []
        

        pospok = []
        tractors = []
        count = Counter(poker)
        for k, v in count.items():
            if v == 2:
                if k != 'jo' and k != 'Jo' and k[1] != level: # 大小王和级牌当然不会参与拖拉机
                    pospok.append([self.pointorder.index(k[1]), k[0]])

        if len(pospok) >= 2:
                pospok.sort(key=lambda x: x[0])
                tmp = []
                suc_flag = False
                for i in range(len(pospok)-1):
                    if pospok[i+1][0] - pospok[i][0] == 1 and pospok[i+1][1] == pospok[i][1]:
                        suit = pospok[i][1]
                        if not suc_flag:
                            tmp = [suit + self.pointorder[pospok[i][0]], suit + self.pointorder[pospok[i][0]], \
                                   suit + self.pointorder[pospok[i+1][0]], suit + self.pointorder[pospok[i+1][0]]]
                            del count[suit + self.pointorder[pospok[i][0]]]
                            del count[suit + self.pointorder[pospok[i+1][0]]] # 已计入拖拉机的，从牌组中删去
                            suc_flag = True
                        else:
                            tmp.extend([suit + self.pointorder[pospok[i+1][0]], suit + self.pointorder[pospok[i+1][0]]])
                            del count[suit + self.pointorder[pospok[i+1][0]]]
                    elif suc_flag:
                        tractors.append(tmp)
                        suc_flag = False
                if suc_flag:
                    tractors.append(tmp)

        outpok = []
        if tractor_len > 0:
            # 過長的拖拉機拆分成連續長度為poker_len的子拖拉机
            for tractor in tractors:
                for idx in range(0, len(tractor), 2):
                    if idx+tractor_len <= len(tractor):outpok.append(tractor[idx:idx+tractor_len])
        else:
            outpok = tractors

        return outpok

    #分析得到可甩牌牌型（大小王和级牌当然不会参与甩牌）
    def parseSuspectPoker(self, poker, level, major):
    # poker: 甩牌牌型 list[int]
    # level & self.Major, banking, major: 级牌、主花牌、主花色
        if len(poker) == 0:
            return []
        
        outpok = []
        suit_pok_list = {suit: [] for suit in __SUITSET__}
        major_pok_list = []
        for k in poker:
            if k != 'jo' and k != 'Jo' and k[1] != level and k[0] != major:
                suit_pok_list[k[0]].append(k)
            else:
                major_pok_list.append(k)
                
        # 为了简化逻辑，这里只取最大张数的甩牌        
        for suit,poks in suit_pok_list.items():
            if len(poks)>1:
                comb_poks = list(poks)
                comb_poks.sort()
                outpok.append(comb_poks)
        if len(major_pok_list)>1:
            comb_poks = list(major_pok_list)
            comb_poks.sort()
            outpok.append(comb_poks)
        return outpok
            
        # suit_comb_pok_list = {suit: [] for suit in __SUITSET__}
        # for suit,poks in suit_pok_list.items():
        #     for n in range(2, len(poks)+1):
        #         comb_poks = list(combinations(poks, n))
        #         suit_comb_pok_list[suit].extend(comb_poks)
        # for suit,poks in suit_comb_pok_list.items():
        #     poks.sort()
        #     outpok.extend(poks)
            
        # for n in range(2, len(major_pok_list)+1):
        #     comb_poks = list(combinations(major_pok_list, n))
        #     comb_poks.sort()
        #     outpok.extend(comb_poks)
        # outpok_final = []
        # element_count = Counter(outpok)
        # #去除重复
        # for k, v in element_count.items():
        #     if v == 1:
        #         outpok_final.append(k)

        # return outpok_final


    def call_Snatch(self, get_card, deck, called, snatched, level):
        # get_card: 本回合新获得的牌 (int)
        # deck: 获得新牌前的手牌 (list[int])
        # called & snatched: 已叫庄/抢庄的玩家ID, -1表示还没有人叫庄/抢庄
        # level: 当前级别
        # return -> list[int] 返回要亮出的牌
        response = []
        get_poker = self.num2Poker(get_card)

        # 不满足报主条件
        if get_poker[1] != level or \
            (called != -1 and (get_card + 54) % 108 in deck):
            return response
       
        # 包含新获得的牌
        deck_pokers = [self.num2Poker(card) for card in deck] + [get_poker]
        # 统计主牌
        major_cards = [poker for poker in deck_pokers if poker in self.Major]        
         # 计算主牌占比
        major_percentage = len(major_cards) / len(deck_pokers)

        #计算主牌牌力值
        count = Counter(major_cards)
        last_cnt = 0
        tractor_len = 0
        power = 0
        for k, v in count.items():
            if count[k] == 1:
                if last_cnt == 2 : 
                    power += tractor_len*5
                    tractor_len = 0
                elif k == 'jo': power += 2
                elif k == 'Jo': power += 3
                elif k[1] == level : power += 1.5
                elif k[1] == '5' or k == k[1] == '0':power += 1
                else:power += 0.5

            elif count[k] == 2:
                if last_cnt == 2 : tractor_len += 1
                elif k == 'jo' or k == 'Jo' or k[1] == level : power += 4                
                elif k[1] == '5' or k == k[1] == '0':power += 3
                else:power += 2

            last_cnt = count[k]

        if (major_percentage >= 0.4 and len(major_cards) > 5) or power >= 8 :
            if called == -1:
                response = [get_card]
            elif snatched == -1:
                if (get_card + 54) % 108 in deck:
                    response = [get_card, (get_card + 54) % 108]
        return response
        
    def cover_Pub(self, old_public, deck):
        # old_public: raw publiccard (list[int])
        ## 直接盖回去
            return old_public
    
    def cover_PubEx(self, old_public, deck, level):
        hand_deck = old_public + deck
        #[__SINGLE__,__PAIR__,__TRACTOR__, __SUSPECT__]
        poker_deck = [self.num2Poker(id) for id in hand_deck]
        rule_deck = self.getTypePoke(poker_deck, level, [__SINGLE__,__PAIR__,__TRACTOR__])
        single = [c for c in rule_deck[__SINGLE__] if c not in self.Major]

        public_card = []
        roll_card = []
        major_single = []
        for card in single:
            if card[0][1] != level and card[0][1] != 'o':
                if card[0][1] == 'A':
                    roll_card.append(card[0])
                elif __CARDSCALE__.index(card[0][1]) < 9:
                    public_card.append(card[0])


            else:
                major_single.append(card[0])
        
        while True:
            if len(public_card) >= 8:
                public_card = random.sample(public_card, 8)
                break
            
            if len(roll_card) >= 8-len(public_card):
                public_card.extend(random.sample(roll_card, 8-len(public_card)))
                if len(public_card) == 8:
                    break
            else:
                public_card.extend(roll_card)
            
            if len(major_single) >= 8-len(public_card):
                public_card.extend(random.sample(major_single, 8-len(public_card)))
                if len(public_card) == 8:
                    break
            else:
                public_card.extend(major_single)
                
            pair = [c[0][0] for c in rule_deck['pair'] if c[0][0] not in self.Major]*2
            if len(pair) >= 8-len(public_card):
                public_card.extend(random.sample(pair, 8-len(public_card)))
                if len(public_card) == 8:
                    break
            else:
                public_card.extend(pair)

            pair = [c[0][0] for c in rule_deck['pair'] if c[0][0] in self.Major]*2
            if len(pair) >= 8-len(public_card):
                public_card.extend(random.sample(pair, 8-len(public_card)))
                if len(public_card) == 8:
                    break
            else:
                public_card.extend(pair)

            for tractor in rule_deck['tractor']:
                if len(tractor) >= 8-len(public_card):
                    public_card.extend(random.sample(tractor, 8-len(public_card)))
                    if len(public_card) == 8:
                        break
                else:
                    public_card.extend(tractor)

            public_card = random.sample(poker_deck, 8)
            break

        return self.Pokers2Num(public_card, hand_deck)
    
    
    def getLegalPlayCard(self, history, deck, level):
        
        poker_deck = [self.num2Poker(id) for id in deck]
        # print("getLegalPlayCard.deck", poker_deck, '\r\n', deck)
        
        all_hands = [[] for _ in range(__WRONG__)]
        
        

        # 首发
        if len(history) == 0: 
            #test code
            # poker_deck = ['d0', 'c4', 'c6', 'cA', 'cA', 'cQ', 'c5', 'd3', 'c5', 'c4', 'd6', 'd3', 'h5', 'h0', 'c2', 'h7', 'hK', 'd6', 'h7', 'dQ', 'hJ', 'hQ', 's3', 'h3', 's8']           
            # level = '6'
            # self.Major = ['s2', 's3', 's4', 's5', 's7', 's8', 's9', 's0', 'sJ', 'sQ', 'sK', 'sA', 'h6', 'c6', 'd6', 's6', 'jo', 'Jo']
            # self.pointorder = ['2', '3', '4', '5', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']

            #为简化游戏环境，去除甩牌牌型
            ret_single = self.getTypePoke(poker_deck, level, type = [__SINGLE__])
            ret_pair = self.getTypePoke(poker_deck, level, type = [__PAIR__])
            ret_tractor = self.getTypePoke(poker_deck, level, type = [__TRACTOR__])
            ret_suspect = self.getTypePoke(poker_deck, level, type = [__SUSPECT__])
                        
            all_hands[__SINGLE__] = ret_single.get(__SINGLE__, [])
            all_hands[__PAIR__] = ret_pair.get(__PAIR__, [])
            all_hands[__TRACTOR__] = ret_tractor.get(__TRACTOR__, [])
            all_hands[__SUSPECT__] = ret_suspect.get(__SUSPECT__, [])
            
            for k,pokers in enumerate(all_hands):
                all_hands[k] = self.PokerList2Num(pokers, deck)
                
            return all_hands
        
        
        
        #test code
        # mj = self.globalInfo['banking']['major']
        # poker_deck = [mj+'5', mj+'5', mj+'0', mj+'6', mj+'7', mj+'9', mj+'8' ]
        # standard_poker = [mj+'3', mj+'3', mj+'6', mj+'6', mj+'5' ]
        # if False:

        standard_move = history[0]
        standard_poker = [self.num2Poker(id) for id in standard_move]
        if self.checkPokerType(standard_move, level) != __SUSPECT__: # 不是甩牌
            pok = [self.num2Poker(p) for p in standard_move] if type(standard_move[0]) == int else standard_move
            own_pok = [self.num2Poker(p) for p in deck]
            all_hands = self.checkResUnSuspect(standard_poker, own_pok, level)
            if sum([len(pokers) for pokers in all_hands]) > 0:
                for i,pokers in enumerate(all_hands):
                    all_hands[i] = self.PokerList2Num(pokers, deck)
                return all_hands
            
            #没有合法牌型，垫牌
            else:
                #出的是主牌
                if standard_poker[0] in self.Major:
                    #print("major")
                    deck_Major = [pok for pok in poker_deck if pok in self.Major]
                    
                    #手上主牌数量不够
                    if len(deck_Major) < len(standard_poker):
                        my_major = deck_Major[:]
                        deck_nMajor = [pok for pok in poker_deck if pok not in self.Major]
                        comb_nMajor = list(combinations(deck_nMajor, len(standard_poker) - len(deck_Major)))
                        for comb in comb_nMajor:
                            all_hands[__DISCARD__].append(my_major + list(comb))
                        
                        
                         
                        # for i in range(len(standard_poker) - len(deck_Major)):
                        #     out.append(deck_nMajor[i])
                        # attach_resp = []
                        # _deck = deck + []
                        # for pok in out:
                        #     cardid = self.Poker2Num(pok, _deck)
                        #     _deck.remove(cardid)
                        #     attach_resp.append(cardid)
                        # return attach_resp
                        
                    #手上主牌够，必须出主牌
                    else:
                        # response = self.checkResUnSuspect(standard_poker, own_pok, level)#test code
                        target_len = len(standard_poker)
                        #单牌
                        if target_len == 1:
                            all_hands[__SINGLE__] = self.getTypePoke(deck_Major, level, [__SINGLE__])[__SINGLE__]
                        #对子
                        elif target_len == 2:
                            all_hands[__PAIR__] = self.getTypePoke(deck_Major, level, ['pair'])['pair']
                            if len(all_hands[__PAIR__]) == 0:
                                single_out = self.getTypePoke(deck_Major, level, [__SINGLE__])[__SINGLE__]
                                all_hands[__PAIR__] = list(combinations(single_out, target_len))
                        #拖拉机
                        else:
                            all_hands[__TRACTOR__] = self.parseTractorPoker(deck_Major, level, target_len)
                            if len(all_hands[__TRACTOR__]) == 0:#没有匹配的牌型
                                pair_out = self.getTypePoke(deck_Major, level, ['pair'])['pair']
                                pair_comb_out = list(combinations(pair_out, target_len//2))
                                all_hands[__TRACTOR__] = [list(itertools.chain.from_iterable(pairs)) for pairs in pair_comb_out]
                                if len(all_hands[__TRACTOR__]) == 0:#没有相同数量的对子组合
                                    pair_out = [pair for pairs in pair_out for pair in pairs]
                                        
                                    single_out = self.getTypePoke(deck_Major, level, [__SINGLE__])[__SINGLE__]
                                    single_out = [item[0] for item in single_out if item[0] not in pair_out]
                                    remain_len = target_len - len(pair_out)
                                    pair_comb_out = list(combinations(single_out, remain_len))
                                    
                                    for single_comb in pair_comb_out:
                                        all_hands[__TRACTOR__].append(pair_out + list(single_comb))
                                            
                                    
                                     
                            
                            # out = []
                            # #先分析同等牌型可出的牌
                            # for poks in target_parse:
                            #     if target_len == 0:
                            #         break
                            #     if len(poks) >= target_len:
                            #         out.extend(poks[:target_len])
                            #         target_len = 0
                            #     else:
                            #         out.extend(poks)
                            #         target_len -= len(poks)
                            # #再分析单牌  
                            # resp = []
                            # _deck = deck + []
                            # for pok in out:
                            #     cardid = self.Poker2Num(pok, _deck)
                            #     _deck.remove(cardid)
                            #     resp.append(cardid)
                            # return resp
                    
                #出的是副牌
                else:
                    #print("not_major")
                    suit = standard_poker[0][0]
                    deck_suit = [pok for pok in poker_deck if pok[0] == suit and pok[1] != level]
                    #副牌数量不够
                    if len(deck_suit) <= len(standard_poker):
                        deck_unSuit = [pok for pok in poker_deck if pok not in deck_suit]
                        comb_unSuit = list(combinations(deck_unSuit, len(standard_poker) - len(deck_suit)))
                        for comb in comb_unSuit:
                            all_hands[__DISCARD__].append(deck_suit + list(comb))
                            
                        # out = deck_suit
                        # deck_nsuit = [pok for pok in poker_deck if pok not in deck_suit]
                        # for i in range(len(standard_poker) - len(deck_suit)):
                        #     out.append(deck_nsuit[i])
                        # attach_resp = []
                        # _deck = deck + []
                        # for pok in out:
                        #     cardid = self.Poker2Num(pok, _deck)
                        #     _deck.remove(cardid)
                        #     attach_resp.append(cardid)
                        # return attach_resp
                    
                    else: #副牌数量够，必须要出副牌
                        # response = self.checkResUnSuspect(standard_poker, own_pok, level)#test code
                        target_len = len(standard_poker)                        
                        if target_len == 1:#单牌
                            all_hands[__SINGLE__] = [[p] for p in deck_suit]
                        else:#对子，拖拉机
                            all_hands[__PAIR__] = self.parseTractorPoker(deck_suit, level, target_len)
                            if len(all_hands[__PAIR__]) == 0:
                                all_hands[__PAIR__] = list(combinations(deck_suit, target_len))

                            # target_parse = self.parseTractorPoker(deck_suit, level, target_len)
                            # target_parse.sort(key=lambda x: len(x), reverse=True)
                            # out = []
                            # if len(target_parse) == 0:
                            #     out = list(combinations(deck_suit, target_len))
                            # else:
                            #     for poks in target_parse:
                            #         if len(poks) == target_len:
                            #             out.append(poks)
                            #         else:
                            #             out.append([poks[i-target_len:i]  for i in range(target_len, len(poks), target_len)])
                                    
        #甩牌
        else:
            #主牌甩牌
            if standard_poker[0] in self.Major:
                #print("major")
                deck_Major = [pok for pok in poker_deck if pok in self.Major]
                
                #手上的主牌数量不够，还需要加上副牌来凑
                if len(deck_Major) < len(standard_poker):
                    deck_unMajor = [pok for pok in poker_deck if pok not in self.Major]
                    comb_unMajor = list(combinations(deck_unMajor, len(standard_poker) - len(deck_Major)))
                    for comb in comb_unMajor:
                        all_hands[__DISCARD__].append(deck_Major + list(comb))
                
                #数量正好相等
                elif len(deck_Major) == len(standard_poker):
                    all_hands[__SUSPECT__] = [deck_Major]

                #手上的主牌数量够，甩牌
                else:
                    all_hands[__SUSPECT__] = self.suspectMatchCard(deck_Major, standard_poker, level)
                    
            #副牌甩牌
            else:
                suit = standard_poker[0][0]
                deck_suit = [pok for pok in poker_deck if pok[0] == suit and pok[1] != level]

                #副牌数不够
                if len(deck_suit) < len(standard_poker):
                    deck_nsuit = [pok for pok in poker_deck if pok not in deck_suit]
                    comb_poks = list(combinations(deck_nsuit, len(standard_poker) - len(deck_suit)))
                    for poks in comb_poks:
                        all_hands[__DISCARD__].append(deck_suit + list(poks))

                #数量正好相等
                elif len(deck_suit) == len(standard_poker):
                    all_hands[__SUSPECT__] = [deck_suit]

                #可出副牌数够
                else:
                    all_hands[__SUSPECT__] = self.suspectMatchCard(deck_suit, standard_poker, level)

        ret = []
        
        # #todo
        # for tp, actions in enumerate(all_hands):
        #     for poks in actions:
        #         _deck = deck[:] + []
        #         for pok in poks:
        #             if type(pok) == list:
        #                 self.getLegalPlayCard(history, deck, level)
        #                 pass
        #             cardid = self.Poker2Num(pok, _deck)
        #             if cardid not in _deck:
        #                 print(self.Pokers2Num(poks,_deck))
        #                 self.getLegalPlayCard(history, deck, level)
        #             _deck.remove(cardid)
        
        
        
        #去除重复
        repeats = dict()
        repeats2 = dict()
        out1 = []
        for tp, actions in enumerate(all_hands):
            for poks in actions:
                outstr = ''.join(poks)
                if outstr not in repeats:
                    out1.append(poks)
                else:
                    repeats2[outstr] = poks
                repeats[outstr] = poks
        
        # todo
        # global __MAX_ACTION_NUM__
        # if __MAX_ACTION_NUM__ < len(out1):
        #     __MAX_ACTION_NUM__ = len(out1)
        #     # print(f'最大动作空间：{__MAX_ACTION_NUM__}, pid={os.getpid()}')
        #     for tp, actions in enumerate(all_hands):
        #         sout = sorted(actions)
        #         if sout != actions:
        #             for subout in actions:
        #                 if subout not in sout:
        #                     pass

        for k, v in enumerate(all_hands):
            all_hands[k] = self.PokerList2Num(v, deck)
        return all_hands
    
    
    
    #own_pok数量>=standard_poker，则直接匹配可出牌型
    def suspectMatchCard(self, own_pok, standard_pok, level):
        match_poks = []
        #甩牌里若有拖拉机，则也要优先匹配拖拉机
        standard_tractor = self.parseTractorPoker(standard_pok, level)
        if len(standard_tractor) > 0:
            for sttractor in standard_tractor:
                standard_pok1 = list(filter(lambda x: x not in sttractor, standard_pok))#去除standard_pok中的sttractor
                sttractor_len = len(sttractor)
                own_tractors = self.parseTractorPoker(own_pok, level, sttractor_len)
                for owtractor in own_tractors:
                    # if len(owtractor) == len(sttractor):
                        own_pok1 = list(filter(lambda x: x not in owtractor, own_pok))
                        remain_match_poks = self.suspectMatchCard(own_pok1, standard_pok1, level)
                        for remain_patch_pok in remain_match_poks:
                            match_poks.append(owtractor + list(remain_patch_pok))
            if len(match_poks)>0:
                return match_poks

        own_counter = sorted(Counter(own_pok).items(), key=lambda item: item[1], reverse=True)
        standard_counter = sorted(Counter(standard_pok).items(), key=lambda item: item[1], reverse=True)
        
        #甩牌里若有对子
        if len(standard_counter)> 0 and standard_counter[0][1] == 2:
            standard_pair_pok = [standard_pok_cnt[0] for standard_pok_cnt in standard_counter if standard_pok_cnt[1] == 2 ]
            own_pair_pok = [own_pok_cnt[0] for own_pok_cnt in own_counter if own_pok_cnt[1] == 2 ]
            
            # 手上对子和出的对子数量
            if len(own_pair_pok) >= len(standard_pair_pok):
                standard_pok1 = [ pok for pok in standard_pok if pok not in standard_pair_pok ]
                comb_pair_list = list(combinations(own_pair_pok, len(standard_pair_pok)))
                for pairs in comb_pair_list:
                    own_pok1 = [pok for pok in own_pok if pok not in pairs]
                    remain_match_poks = self.suspectMatchCard(own_pok1, standard_pok1, level)
                    out_pair_poks = pairs*2
                    for remain_patch_pok in remain_match_poks:
                        match_poks.append(list(out_pair_poks) + list(remain_patch_pok))
            elif len(own_pair_pok) < len(standard_pair_pok) and len(own_pair_pok) > 0:
                own_pok1 = [pok for pok in own_pok if pok not in own_pair_pok]
                standard_pair_pok = standard_pair_pok[:len(own_pair_pok)]
                standard_pok1 = [ pok for pok in standard_pok if pok not in standard_pair_pok ]
                remain_match_poks = self.suspectMatchCard(own_pok1, standard_pok1, level)
                out_pair_poks = own_pair_pok*2
                for remain_patch_pok in remain_match_poks:
                    match_poks.append(out_pair_poks + list(remain_patch_pok))
            if len(match_poks)>0:
                return match_poks
        
        #全部是单牌
        match_poks = list(combinations(own_pok, len(standard_pok)))
        return match_poks
                    
    def _deal_to(self, p):
        idx = self.globalInfo["deal_count"][p]
        card = self.globalInfo["allocation"][p][idx]
        self.player_hand_cards[p].append(card)
        self.player_hand_decks[p].append(self.num2Poker(card))
        self.globalInfo["deal_count"][p] += 1
        self.globalInfo["dealt"] += 1
        self.globalInfo["deliver"] = [card]

    def _process_bid(self, response):
        p = self.globalInfo["playerpos"]
        if response is None:
            seat, suit = p, ""
        else:
            seat = response[0]
            suit = response[1] if len(response) > 1 else ""
        if seat != p:
            self.setError(seat, "ILLEGAL_MOVE")
            return

        # 叫主/反主资格校验：必须命中当前阶段公开的合法报主动作
        # （无叫时可报单张级牌/对子/双王；已报主后只能对子反主；已反主后只能双王无主）
        if suit not in self.getLegalBidActions(seat):
            self.setError(seat, "ILLEGAL_MOVE")
            return

        banking = self.globalInfo["banking"]
        if suit in __SUITSET__:
            # 叫主或反主：直接指定主花色
            banking["major"] = suit
            if self.globalInfo["first_round"]:
                banking["banker"] = seat
                banking["banker_last"] = seat
        elif suit == "n":
            # 叫无主
            banking["major"] = "n"
            if self.globalInfo["first_round"]:
                banking["banker"] = seat
                banking["banker_last"] = seat
        # 空字符串表示不叫，不改变任何状态
        else:
            return
        
        self.globalInfo["bid_seq"].append([seat, suit])
        self.globalInfo["banking"] = banking
        
        if hasattr(self, '_game_log'):
            print(f"玩家{seat}叫主, 花色:{suit}, 级数:{self.getLevel()}")

    def _advance_deal(self):
        total = __PLAYER_COUNT__ * __HAND_CARD_NUM__
        if self.globalInfo["dealt"] < total:
            nextplayer = (self.globalInfo["playerpos"] + 1) % __PLAYER_COUNT__
            self._deal_to(nextplayer)
            self.globalInfo["playerpos"] = nextplayer
            self.globalInfo["stage"] = "bid"
        else:
            self._finalize_deal()

    def _finalize_deal(self):
        banking = self.globalInfo["banking"]
        # 发牌结束仍无人叫主花色，本局打无主
        if banking["major"] == "":
            banking["major"] = "n"
        if banking["banker"] == -1:
            banking["banker"] = self.globalInfo.get("first_dealer", 0)
            banking["banker_last"] = banking["banker"]
        banker = banking["banker"]
        major = banking["major"]
        self.globalInfo["banking"] = banking
        self.setMajor(major, self.globalInfo["level"])
        self.player_hand_cards[banker].extend(self.globalInfo["publiccard"])
        self.player_hand_decks[banker].extend([self.num2Poker(card) for card in self.globalInfo["publiccard"]])
        self.globalInfo["bidsseat"] = self.globalInfo["bid_seq"][-1][0] if len(self.globalInfo["bid_seq"]) > 0 else banker#最后叫牌的人
        self.globalInfo["stage"] = "cover"
        self.globalInfo["playerpos"] = banker#首轮从庄家开始出牌
        
        if hasattr(self, '_game_log'):
            print(f"叫牌阶段结束, 花色:{self.getMajorColor()}, 级数:{self.getLevel()}, 庄家:{self.getBanker()}")

    # initdata里包含了庄家、牌型、级牌
    # global里包含了主花色、级牌、庄家（覆盖）、报主情况
    def step(self, response=None):
        self.get_score = 0
        self.get_score_pok = []
        # 每步重置错误记录：否则一次非法动作会让 getError 永久非空
        self.errored = [[] for _ in range(__PLAYER_COUNT__)]
        self.erro_code = 0
        self.step_count += 1
        
        # try: 
        # 叫主/反主/无主/不叫，处理完后发下一张牌或结束发牌
        if self.globalInfo["stage"] == "bid":
            self._process_bid(response)
            # 非法叫主/反主已记账，直接中止本步，避免继续发牌推进状态
            if self.erro_code:
                return
            self._advance_deal()

        #埋牌
        elif self.globalInfo["stage"] == "cover":
            cover_seat = response[0]
            cover_cards = list(response[1])
            banker = self.globalInfo["banking"]["banker"]
            # 只有庄家能埋底，且必须埋满底牌张数（8 张）
            if cover_seat != banker:
                self.setError(cover_seat, "ILLEGAL_MOVE")
                return
            if len(cover_cards) != 8:
                self.setError(cover_seat, "INVALID_MOVE")
                return

            my_hand_cards = self.player_hand_cards[cover_seat][:]
            # 先校验（含重复牌），校验通过后再改状态，避免半途失败留下脏数据
            remain = Counter(my_hand_cards)
            for c in cover_cards:
                if remain[c] <= 0:
                    self.setError(cover_seat, "NOT_YOUR_POKER")
                    return
                remain[c] -= 1
            for c in cover_cards:
                self.player_hand_cards[cover_seat].remove(c)
                self.player_hand_decks[cover_seat].remove(self.num2Poker(c))
            self.globalInfo["publiccard"] = cover_cards

            self.globalInfo["history"] = [[],[], cover_seat, cover_seat]
            self.globalInfo["game_score"] = 0
            
            self.globalInfo["stage"] = "play"
            
            if hasattr(self, '_game_log'):
                print(f"埋牌阶段, 埋牌玩家:{cover_seat}, 埋牌:{[self.num2Poker(p) for p in cover_cards]}, 手牌:{[self.num2Poker(p) for p in my_hand_cards]}")
        
        # #开始比赛
        elif self.globalInfo["stage"] == 'ready':
            self.globalInfo["stage"] = 'play'
            
        #一回合结束
        elif self.globalInfo["stage"] == 'roundend':
            self.globalInfo["stage"] = 'play'
            
        # 正式出牌
        elif self.globalInfo["stage"] == "play":
            banker = self.globalInfo["banking"]["banker"]
            major = self.globalInfo["banking"]["major"]
            # self.setMajor(major, self.globalInfo["level"])
            # old_publiccard = full_input["initdata"]["publiccard"]
            # new_publiccard = full_input["log"][201][str(banker)]["response"]
            # big_hold = old_alloc[banker] + old_publiccard
            old_score = self.globalInfo["game_score"]
            # for pok in new_publiccard:
            #     big_hold.remove(pok)
            # new_alloc = old_alloc.copy() + []
            # new_alloc[banker] = big_hold
            
            history = self.globalInfo["history"]
            currplayer = response[0]
            curr_move = response[1]
            # 校验行动者：必须轮到 playerpos 出牌
            if currplayer != self.globalInfo["playerpos"]:
                self.setError(currplayer, "ILLEGAL_MOVE")
                return
            # if type(curr_move) is not list:
            #     self.setError(currplayer, "INVALID_FORMAT")
            # 空出牌属于非法动作，直接受控判非法（避免后续 checkPokerType 越界）
            if curr_move is None or len(curr_move) == 0:
                self.setError(currplayer, "INVALID_MOVE")
                return
            # latest_request = full_input["log"][-2]
            
            for pok in curr_move:
                if pok not in self.player_hand_cards[currplayer]:
                    self.setError(currplayer, "PLAY CARD NOT YOUR POKER")
                    return
            
            play_move = [self.num2Poker(p) for p in curr_move]        
            outpok = self.checkLegalMove(play_move, self.globalInfo["level"], major, currplayer, history[1], self.player_hand_cards, banker)
            
            # collect history
            my_hand_cards = self.player_hand_cards[currplayer][:]
            outid = []
            for pok in outpok:
                id = self.Poker2Num(pok, self.player_hand_cards[currplayer])
                outid.append(id)
                self.player_hand_cards[currplayer].remove(id)
                self.player_hand_decks[currplayer].remove(pok)
            
            #出牌序列
            self.play_seq.append([currplayer, outid, self.checkPokerType(play_move, self.globalInfo["level"])])
            
            if hasattr(self, '_game_log'):
                action = self.play_seq[-1]
                round = max(0, len(self.play_seq)-1)//__PLAYER_COUNT__
                print(f"出牌阶段, 第{round}轮, 玩家:{action[0]}, 牌型{action[2]}, 出牌:{outpok}, 手牌:{[self.num2Poker(p) for p in my_hand_cards]}")
            
            new_history = history[1]            
            if len(new_history) == 0:
                history[3] = currplayer
            if len(new_history) < __PLAYER_COUNT__:#len(new_history) < 3
                nextplayer = (currplayer + 1) % __PLAYER_COUNT__
            new_history.append(outid)
            self.player_played_cards[currplayer].extend(outid)
            old_history = history[0]
        
            if len(new_history) == __PLAYER_COUNT__: # 本回合为该轮最后一次出牌
                winner = self.checkWinner(history[1], currplayer, self.globalInfo["level"], major, banker)
                nextplayer = winner
                old_history = new_history
                new_history = []
                history[2] = history[3] + 0
                history[3] = winner

                if len(self.player_hand_cards[currplayer]) == 0: # 本局结束
                    # 扣底
                    if self.checkPokerType(history[1][0], self.globalInfo["level"]) != __SUSPECT__:
                        mult = len(history[1][0])
                    else:
                        divided, _ = self.checkThrow(history[1][0], [[]], (currplayer - (__PLAYER_COUNT__ - 1)) % __PLAYER_COUNT__, self.globalInfo["level"], major, check=False)
                        divided.sort(key=lambda x: len(x), reverse=True)
                        if len(divided[0]) >= 4:
                            mult = len(divided[0]) * 2
                        elif len(divided[0]) == 2:
                            mult = 4
                        else: 
                            mult = 2

                    publicscore = 0
                    for pok in self.globalInfo["publiccard"]: 
                        p = self.num2Poker(pok)
                        if p[1] == "5":
                            publicscore += 5
                        elif p[1] == "0" or p[1] == "K":
                            publicscore += 10
                    
                    self.Reward(publicscore*mult, winner, banker)
                    new_score = old_score + self.get_score
                    self.globalInfo["game_score"] = new_score
                    
                    history[0] = old_history
                    history[1] = new_history
                    self.globalInfo["history"] = history
                    self.globalInfo["playerpos"] = winner
                    self.globalInfo["stage"] = 'gameend'
                    
                    if hasattr(self, '_game_log'):
                        print(f'游戏结束：庄家:{self.getBanker()}, 闲家本局得分:{new_score}')
                        
                    self.EndGame(banker, new_score)
                    return


                # 非终止回合出现分数变动，说明甩牌失败
                if self.get_score != 0: 
                    pass
                new_score = old_score + self.get_score# get_score 为负数就是罚分
                self.globalInfo["game_score"] = new_score
                if hasattr(self, '_game_log'):
                    print(f"得分更新, 本轮得分:{self.get_score}, 总得分:{new_score}")

            history[0] = old_history
            history[1] = new_history
            self.globalInfo["history"] = history
            self.globalInfo["playerpos"] = nextplayer
            self.globalInfo["stage"] = len(new_history)==0 and "roundend" or "play"

        #test code
        # except Exception as e:
        #     traceback.print_exc()
        #     raise e

    #小局结束
    def isInningEnd(self):
        return self.step_count == -1
    
    #获取玩家的已出牌
    def getPlayedCards(self, play_pos):
        return self.player_played_cards[play_pos][:]
    
    #打印出牌过程
    def print_game_log(self, b):
        self._game_log = b    

__MAX_ACTION_NUM__ = 0
def run_random(env):
    response = None
    level = env.getLevel()
    stage = env.getStage()
    play_pos = env.getPlayerPosition()

    if stage == "bid":
        # 随机叫主：多数不叫，偶尔叫一个花色或反主/无主
        r = random.random()
        if r < 0.6:
            suit = ""
        elif r < 0.9:
            suit = random.choice(__SUITSET__)
        else:
            suit = "n"
        response = [play_pos, suit]
    elif stage == "cover":
        publiccard = env.getPublicCards()
        hold = env.getPlayerHandCards(env.getBanker())
        # cover_PubEx 的 deck 参数为不含底牌的 25 张手牌
        deck = [c for c in hold if c not in publiccard]
        # response = [env.getBanker(), env.cover_Pub(publiccard, deck)]
        response = [env.getBanker(), env.cover_PubEx(publiccard, deck, level)]
    elif stage == "play":
        history_curr = env.getCurrRoundPlayHistory()
        hold = env.getPlayerHandCards(play_pos)
        
        
        playedCards = env.getLegalPlayCard(history_curr, hold, level)
        
        playedCards = playedCards[random.randint(0, len(playedCards)-1)]
        # if type(playedCards) is not list:#todo
        #     playedCards = env.getLegalPlayCard(history_curr, hold, level)
        response = [play_pos, playedCards]
    elif stage == "roundend":
        pass
    elif stage == "finish":
        pass

    return response

class TractorBot:
    def __init__(self, env):
        self.env = env  # tractorGame 实例

    # 叫主/反主决策（改进自原 call_Snatch）
    def call_snatch(self, get_card, deck, called, snatched, level):
        get_poker = self.env.num2Poker(get_card)
        if get_poker[1] != level:
            return []  # 不是级牌，不能叫
        if called != -1 and (get_card + 54) % 108 in deck:
            return []  # 已没对子可反

        deck_pokers = [self.env.num2Poker(c) for c in deck] + [get_poker]
        major_cards = [p for p in deck_pokers if p in self.env.Major]
        major_percent = len(major_cards) / len(deck_pokers)

        # 计算主牌强度（包含大小王、级牌、对子、拖拉机等）
        count = Counter(major_cards)
        strength = 0
        for k, v in count.items():
            if k == 'jo':
                strength += 3 if v >= 2 else 2
            elif k == 'Jo':
                strength += 4 if v >= 2 else 3
            elif k[1] == level:
                strength += 2 if v >= 2 else 1.5
            elif k[1] in ('5', '0'):
                strength += 1.5 if v >= 2 else 1
            else:
                strength += 1 if v >= 2 else 0.5

        # 叫牌条件：主牌比例高 或 强度够大
        if (major_percent >= 0.35 and len(major_cards) > 5) or strength >= 9:
            if called == -1:
                return [get_card]
            elif snatched == -1:
                if (get_card + 54) % 108 in deck:
                    return [get_card, (get_card + 54) % 108]
        return []

    # 盖底牌决策（改进自原 cover_PubEx）
    def cover_pub(self, publiccard, deck, level):
        hand = publiccard + deck
        poker_hand = [self.env.num2Poker(c) for c in hand]
        major_color = self.env.getMajorColor()
        if major_color == 'n':
            major_cards = [p for p in poker_hand if p in self.env.Major]
        else:
            major_cards = [p for p in poker_hand if p in self.env.Major and (p[0] == major_color or p[1] == level or p in __MAJOR__)]

        # 统计副牌各花色及分牌
        suit_cards = {s: [] for s in __SUITSET__ if s != major_color}
        for p in poker_hand:
            if p not in major_cards and p[0] != major_color:
                suit_cards.setdefault(p[0], []).append(p)

        chosen = []
        # 优先扣分（5,10,K）并且尽可能绝一门
        points_order = ['5', '0', 'K']  # 分牌
        for suit, cards in suit_cards.items():
            cards.sort(key=lambda x: (x[1] in points_order, self.env.pointorder.index(x[1]) if x[1] != 'o' else 99), reverse=True)
            # 扣到只剩少量控制牌或绝门
            keep_count = max(1, len(cards) - 2) if len(cards) > 2 else 0
            for c in cards:
                if len(chosen) >= 8:
                    break
                if c[1] in points_order or len(cards) - len(chosen) > keep_count:
                    chosen.append(c)
            if len(chosen) >= 8:
                break

        # 如果还不够8张，再从主牌中选弱牌
        if len(chosen) < 8:
            major_cards.sort(key=lambda x: (x[1] == level, self.env.Major.index(x) if x in self.env.Major else 999))
            for c in major_cards:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) == 8:
                    break

        # 转换为数字编号
        result = self.env.Pokers2Num(chosen[:8], hand)
        return result

    # 出牌主函数
    def play_card(self, history, deck, level):
        legal_plays = self.env.getLegalPlayCard(history, deck, level)
        if not legal_plays:
            return []
        if len(history) == 0:
            # 首发：评估每种出法的“保守度”
            scored = [(self.eval_lead(play, deck, level), play) for play in legal_plays]
        else:
            scored = [(self.eval_follow(play, history, deck, level), play) for play in legal_plays]
        scored.sort(key=lambda x: x[0], reverse=True)  # 得分越高越好
        return scored[0][1]

    def eval_lead(self, play, deck, level):
        poker_play = [self.env.num2Poker(c) for c in play]
        # 鼓励出小单张或小对子，避免拆对子/拖拉机，除非能甩
        score = 0
        # 如果是甩牌，判断安全性
        if self.env.checkPokerType(play, level) == __SUSPECT__:
            # 环境会自动校验甩牌是否成功，我们这里倾向得分高但实际出牌仍会被校验
            return 100  # 甩牌总是好的
        for card in poker_play:
            if card[1] == level or card in __MAJOR__:
                score -= 0.5  # 不轻易出级牌/大小王
            elif card[1] in ('A', 'K'):
                score += 2
            elif card[1] in ('5', '0'):
                score -= 3  # 少出分
            else:
                score += 1  # 小牌加分，逼迫对手出大牌
        # 对子/拖拉机额外加分
        typ = self.env.checkPokerType(play, level)
        if typ == __PAIR__:
            score += 2
        elif typ == __TRACTOR__:
            score += 5
        return score

    def eval_follow(self, play, history, deck, level):
        lead_move = history[0]
        lead_typ = self.env.checkPokerType(lead_move, level)
        lead_pokers = [self.env.num2Poker(c) for c in lead_move]
        play_pokers = [self.env.num2Poker(c) for c in play]
        my_deck = [self.env.num2Poker(c) for c in deck]

        # 有同花色或主牌时：
        if lead_pokers[0] in self.env.Major:  # 主牌局
            required_suit = self.env.Major
        else:
            required_suit = [lead_pokers[0]]

        # 1. 必跟情况：尽量出最小的合法组合
        if set(play_pokers).issubset(my_deck):  # 其实是合法的
            # 如果有杀牌（毙牌），得分看情况
            if lead_pokers[0] not in self.env.Major and set(play_pokers).issubset(self.env.Major):
                return 100  # 杀牌，鼓励最小主牌毙
            # 垫牌：尽量垫无分的小牌
            score = 0
            for c in play_pokers:
                if c[1] in ('5', '0', 'K'):
                    score -= 5
                elif c[1] == 'A':
                    score += 1
                else:
                    score += 0.5
            return score
        return 0
    
    
    
def run(env):
    bot = TractorBot(env)
    play_pos = env.getPlayerPosition()
    stage = env.getStage()
    level = env.getLevel()

    if stage == "bid":
        # 简单叫主策略：随机叫花色/无主/不叫
        r = random.random()
        if r < 0.6:
            suit = ""
        elif r < 0.9:
            suit = random.choice(__SUITSET__)
        else:
            suit = "n"
        return [play_pos, suit]
    elif stage == "cover":
        publiccard = env.getPublicCards()
        hold = env.getPlayerHandCards(env.getBanker())
        deck = [c for c in hold if c not in publiccard]
        response = bot.cover_pub(publiccard, deck, level)
        return [env.getBanker(), response]
    elif stage == "play":
        history = env.getCurrRoundPlayHistory()
        hold = env.getPlayerHandCards(play_pos)
        response = bot.play_card(history, hold, level)
        return [play_pos, response]
    else:
        return []
    
def runGame():
    for _ in range(100000):
        # print(f'start round,{_}, pid={os.getpid()}\r\n')
        env = tractorGame()
        envs[os.getpid()] = env
        env.reset()
        response = None
        while True:
            if response is None:
                response = run_random(env)

            env.step(response)

            if env.isFinalEnd():
                # print(env.globalInfo)
                break
            if env.getStage() == "gameend":
                env.reset()
                response = None
                continue

            response = run_random(env)



if __name__ == '__main__':
    # matrix = np.arange(54*2, dtype=np.int8)
    # arr = [0,2,4,8,52]
    # matrix[arr] = -1
    # matrix = np.insert(matrix, 54, [-2,-2])
    # matrix = np.insert(matrix, 110, [-2,-2])
    # matrix = matrix.reshape(2,14,4)
    # matrix = np.transpose(matrix, (0,2,1))
    
    # # 创建一个新的列顺序数组
    # new_order = list(range(matrix.shape[2]))
    # new_order.remove(2)
    # new_order.insert(12, 2)

    # # 使用新的列顺序重新排列矩阵
    # print(matrix)
    # matrix = matrix[:, :, new_order]
    # print(matrix)

    import multiprocessing
    # scores = os.cpu_count()
    scores=1
    print(f"Number of CPU cores: {scores}")
    processes = []
    for _ in range(scores):
        p = multiprocessing.Process(target=runGame)
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()
    pass
