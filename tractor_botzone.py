import random, os, signal
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

###############################################################
# 牌面表示：数字
# h:红桃 d:方片 s:黑桃 c:草花 
# (0-h1 1-d1 2-s1 3-c1) (4-h2 5-d2 6-s2 7-c2) ... 52-joker 53-Joker (54-h1 55-d1 56-s1 57-c1) ... 106-joker 107-Joker
# 请注意：10记为0
# 共2副108张
###############################################################
__CARDSCALE__ = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
__SUITSET__ = ['s','h','c','d']# h:红桃 d:方片 s:黑桃 c:草花 
__MAJOR__ = ['jo', 'Jo']#小王 大王
__POINT__ = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
__CARDSCALE_COUNT__ = 14 #点数
__PLAYER_COUNT__ = 4
__CARDS_NUM__ = (54*2)

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

class tractorGame():
    def __init__(self):
        self.errored = [[] for _ in range(__PLAYER_COUNT__)]
        self.Major = copy.deepcopy(__MAJOR__)
        self.get_score = 0 # 闲家得分，每次step前清空一下该值
        self.get_score_pok=[]# 得分牌，每次step前清空一下该值
        self.globalInfo = {} # 未确定主花色前为空
        
        self.step_count = -1 # 一局里面的步数, -1是未在对局中
        self.player_hand_cards = [[] for _ in range(__PLAYER_COUNT__)] # 每位玩家手牌
        self.player_played_cards = [[] for _ in range(__PLAYER_COUNT__)] # 各玩家已经出过的牌
        self.player_level = ['2' for _ in range(__PLAYER_COUNT__)] # 各玩家当前的级数
        self.round_score = [0 for _ in range(__PLAYER_COUNT__)] # 玩家的局分数
        self.pointorder = copy.deepcopy(__POINT__)
        self.logs = []
        self.erro_code = 0
        self._final_ended = False
    
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

    def Poker2Num(self, poker, deck): # poker: str
        NumInDeck = -1
        if poker[0] == "j":
            NumInDeck = 52
        elif poker[0] == "J":
            NumInDeck = 53
        else:
            NumInDeck = __CARDSCALE__.index(poker[1])*4 + __SUITSET__.index(poker[0])
        return NumInDeck if NumInDeck in deck else NumInDeck + 54
        
        
    #pokers是个二维数组，每个元素是个list，list是个扑克牌列表
    def PokerList2Num(self, pokers, deck): # poker: str        
        numDesk = []
        for _pokers in pokers:
            rule_poker = []
            _deck = copy.deepcopy(deck)
            for poker in _pokers:
                NumInDeck = -1
                if poker[0] == "j":
                    NumInDeck = 52
                elif poker[0] == "J":
                    NumInDeck = 53
                else:
                    NumInDeck = __CARDSCALE__.index(poker[1])*4 + __SUITSET__.index(poker[0])
                
                p = NumInDeck if NumInDeck in _deck else NumInDeck + 54
                rule_poker.append(p)
                if p not in _deck:
                    pass
                _deck.remove(p)
            numDesk.append(rule_poker)
        return numDesk

    #pokers是个list，list是个扑克牌列表
    def Pokers2Num(self, pokers, deck): # poker: str        
        numDesk = []
        for poker in pokers:
            NumInDeck = -1
            if poker[0] == "j":
                NumInDeck = 52
            elif poker[0] == "J":
                NumInDeck = 53
            else:
                NumInDeck = __CARDSCALE__.index(poker[1])*4 + __SUITSET__.index(poker[0])
            numDesk.append(NumInDeck if NumInDeck in deck else NumInDeck + 54)
            
        return numDesk
    
    # 确定主牌
    def setMajor(self, major, level):
        self.Major = copy.deepcopy(__MAJOR__)
        self.pointorder = copy.deepcopy(__POINT__)
        if major != 'n': # 非无主
            self.Major = [major+point for point in self.pointorder if point != level] + [suit + level for suit in __SUITSET__ if suit != major] + [major + level] + self.Major
        else: # 无主
            self.Major = [suit + level for suit in __SUITSET__] + self.Major
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
        
        
        endingScores = {}
        for i in range(4):
            if i == player:
                endingScores[str(i)] = -3 # 出错会被额外扣分
            elif i == (player + 2) % 4:
                endingScores[str(i)] = 0
            else:
                endingScores[str(i)] = 1
        
        self.errored[player].append([reason, endingScores])
        self.erro_code = 1
        print(json.dumps({
            "command": "finish",
            "content": endingScores,
            "display": {
                "currplayer": player,
                "score": endingScores,
                "error": self.errored
            }
        }))
        
        
        #test code
        pass
        for pid,_ in envs.items():
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)

        response = run(self)
        self.step(response)

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
        poker = [self.num2Poker(p) for p in poker] if type(poker[0])==int else poker
        if len(poker) == 1:
            return "single" #一张牌必定为单牌
        if len(poker) == 2:
            if poker[0] == poker[1]:
                return "pair" #同点数同花色才是对子
            else:
                return "suspect" #怀疑是甩牌
        if len(poker) % 2 == 0: #其他情况下只有偶数张牌可能是整牌型（连对）
        # 连对：每组两张；各组花色相同；各组点数在大小上连续(需排除大小王和级牌)
            count = Counter(poker)
            if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2 and len(poker) == 4:
                return "tractor"
            elif "jo" in count.keys() or "Jo" in count.keys(): # 排除大小王
                return "suspect"
            for v in count.values(): # 每组两张
                if v != 2:
                    return "suspect"
            pointpos = []
            suit = list(count.keys())[0][0] # 花色相同
            for k in count.keys():
                if k[0] != suit or k[1] == level: # 排除级牌
                    return "suspect"
                pointpos.append(self.pointorder.index(k[1])) # 点数在大小上连续
            pointpos.sort()
            for i in range(len(pointpos)-1):
                if pointpos[i+1] - pointpos[i] != 1:
                    return "suspect"
            return "tractor" # 说明是拖拉机
        
        return "suspect"

    # 甩牌判定功能函数
    # return: ExistBigger(True/False)
    # 给定一组常规牌型，鉴定其他三家是否有同花色的更大牌型
    def checkBigger(self, poker, own, currplayer, level, major):
    # poker: 给定牌型 list
    # own: 各家持牌 list
        tyPoker = self.checkPokerType(poker, level)
        poker = [self.num2Poker(p) for p in poker] if type(poker[0])==int else poker
        assert tyPoker != "suspect", "Type 'throw' should contain common types"
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
        if "allocation" in full_input["initdata"]:
            allocation = full_input["initdata"]["allocation"]
        else: # 产生大家各自有什么牌
            allo = [i for i in range(108)]
            random.shuffle(allo)
            allocation = [allo[8:33], allo[33:58], allo[58:83], allo[83:108]]

        if "publiccard" in full_input["initdata"]:
            publiccard = full_input["initdata"]["publiccard"]
        else:
            publiccard = allo[0:8]
        
        return full_input, seedRandom, allocation, publiccard

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
            if  len(banking["called"]) == 0 or len(banking["called"]) == 3: # 还未报主或已经反主
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
    def getTypePoke(self, own_pok, level, type = ["single","pair","tractor", "suspect"]):        
        rule_poks = []#poker格式
        
        # own_pok = ['d4','d4','d5','d5','d7','d7','d8','d8']#test
        if 'suspect' not in type:
            count = Counter(own_pok)
            if 'single' in type:
                rule_poks.extend([[p] for p in own_pok])
                
            if 'pair' in type:
                for k, v in count.items():
                    if v == 2:# and k != 'jo' and k != 'Jo':
                        rule_poks.append([k,k])
                        
            if 'tractor' in type:
                tr = self.parseTractorPoker(own_pok, level)
                rule_poks.extend(tr)

        #甩牌的牌堆里肯定包含了单，对，连对
        else:
            tr = self.parseSuspectPoker(own_pok, level, self.globalInfo["banking"]["major"])
            rule_poks.extend(tr)

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

    #play_pok不是甩牌牌型
    def checkResUnSuspect(self, play_pok, own_pok, level): # poker: list[int]
        poker_len = len(play_pok)
        suit = play_pok[0][0]
        typoker = self.checkPokerType(play_pok, level)
        #出的是主牌
        if play_pok[0] in self.Major:
            major_pok = [pok for pok in own_pok if pok in self.Major]
            my_pok_count = Counter(major_pok)   

            #单张或对子         
            if poker_len <= 2:
                ret = []
                for k,v in my_pok_count.items():
                    if v >= poker_len:
                        ret.append([k]*poker_len)
                if len(major_pok) < poker_len:
                    unmajor_pok = [pok for pok in own_pok if pok not in major_pok]
                    ret = [major_pok+list(pok) for pok in list(combinations(unmajor_pok, poker_len-len(major_pok)))]
                elif len(ret) == 0:              
                    ret.extend(list(combinations(major_pok, poker_len)))
                return ret
                
            # 主牌拖拉机
            else: 
                deck_Major = [pok for pok in own_pok if pok in self.Major]
                tractors = self.parseTractorPoker(deck_Major, level)
                ret = [tractor for tractor in tractors if len(tractor) == poker_len]

                #没有拖拉机，看有没有对子
                if len(ret) == 0:                  
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
                            
                        ret = [fixed_pok + list(poks) for poks in combinations(singlepok, poker_len-len(fixed_pok))]
                return ret
                
                
        #出的是副牌
        else:
            suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
            # print('suit_pok:',suit_pok)
            my_pok_count = Counter(suit_pok)

            #单张或对子
            if poker_len <= 2:
                ret = []
                for k,v in my_pok_count.items():
                    if v >= poker_len:
                        ret.append([k]*poker_len)
                
                if len(suit_pok) < poker_len:
                    unsuit_pok = [pok for pok in own_pok if pok[0] != suit or pok[1] == level]
                    ret = [suit_pok+list(pok) for pok in list(combinations(unsuit_pok, poker_len-len(suit_pok)))]
                elif len(ret) == 0:
                    ret.extend(list(combinations(suit_pok, poker_len)))
                return ret
            
            #副牌拖拉机
            else:
                tractors = self.parseTractorPoker(suit_pok, level)
                ret = [tractor for tractor in tractors if len(tractor) == poker_len]
                #没有副拖拉机
                if len(ret) == 0:
                    #看有没有主牌拖拉机
                    deck_Major = [pok for pok in own_pok if pok in self.Major]
                    tractors = self.parseTractorPoker(deck_Major, level)
                    ret_tr = [tractor for tractor in tractors if len(tractor) == poker_len]
                    ret.extend(ret_tr)

                    #其他牌
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
                            
                        ret.extend([fixed_pok + list(poks) for poks in combinations(singlepok, poker_len-len(fixed_pok))])
                return ret
            
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
            if typoker == "suspect":
                outpok_s, ilcnt = self.checkThrow(poker, own, currplayer, level, major, True)
                if ilcnt > 0:
                    self.Punish(currplayer, banker, ilcnt*10)
                outpok = [p for poktype in outpok_s for p in poktype] # 符合交互模式，把甩牌展开
        else:
            tyfirst = self.checkPokerType(history[0], level)
            if len(poker) != len(history[0]):
                self.setError(currplayer, "ILLEGAL_MOVE")
            if tyfirst == "suspect": # 这里own不一样了，但是可以不需要check
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
                        if self.checkPokerType(outhis[0], level) == "tractor": # 拖拉机来喽
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
                        if self.checkPokerType(outhis[0], level) == "tractor": # 拖拉机来喽
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
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self.setError(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
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
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self.setError(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
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
        if tyfirst == "suspect": # 甩牌
            first_parse, _ = self.checkThrow(history[0], [[]], currplayer, level, major, check=False)
            first_parse.sort(key=lambda x: len(x), reverse=True)
            for i in range(1,4):
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
                    if self.Major.index(win_parse[0][-1]) < self.Major.index(move_parse[0][-1]):
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
                        elif self.Major.index(move_max) > self.Major.index(win_max):
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
            for i in range(1, 4):
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
                    if major == 'n':
                        if win_move[-1][1] == level:
                            if hist[i][-1] == 'jo' or hist[i][-1] == 'Jo': # 目前胜牌是级牌，只有大小王能压
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                    else:
                        if win_move[-1][0] != major and win_move[-1][1] == level:
                            if (hist[i][-1][0] == major and hist[i][-1][1] == level) or hist[i][-1] == 'jo' or hist[i][-1] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                else: # 副牌存在被主牌压的情况
                    if hist[i][0] in self.Major: # 主牌，正确牌型，必压
                        win_move = hist[i]
                        win_seq = i
                    elif self.pointorder.index(win_move[0][-1]) < self.pointorder.index(hist[i][0][-1]):
                        win_move = hist[i]
                        win_seq = i
        # 找到获胜方，加分
        win_id = (currplayer - 3 + win_seq) % 4
        self.Reward(score, win_id, banker)

        return win_id


    def Reward(self, score, currplayer, banker):
        if (currplayer - banker) % 2 != 0: # 非庄家得分
            self.get_score += score

    # return endingScores(dict)
    def EndGame(self, banker, score):
        endingScores = {}
        bankers = [banker, (banker + 2) % __PLAYER_COUNT__]
        up_level_step = 1#晋升的级数
        banker_win = score < 80
        if score <= 0: # 大光，庄家得3分
            up_level_step = 3
            for i in range(4):
                if i in bankers:
                    self.round_score[i] += 3
                    endingScores[str(i)] = 3
                else: 
                    endingScores[str(i)] = 0
        elif score < 40: # 小光，庄家得2分
            up_level_step = 2
            for i in range(4):
                if i in bankers:
                    self.round_score[i] += 2
                    endingScores[str(i)] = 2
                else:
                    endingScores[str(i)] = 0
        elif score < 80: # 庄家得1分
            up_level_step = 1
            for i in range(4):
                if i in bankers:
                    self.round_score[i] += 1
                    endingScores[str(i)] = 1
                else:
                    endingScores[str(i)] = 0
        elif score < 120: # 闲家得1分
            up_level_step = 0
            for i in range(4):
                if i in bankers:
                    endingScores[str(i)] = 0
                else:
                    self.round_score[i] += 1
                    endingScores[str(i)] = 1
        elif score < 160: # 闲家得2分
            up_level_step = 1
            for i in range(4):
                if i in bankers:
                    endingScores[str(i)] = 0
                else:
                    self.round_score[i] += 2
                    endingScores[str(i)] = 2
        else: 
            up_level_step = 2
            for i in range(4):
                if i in bankers:
                    endingScores[str(i)] = 0
                else:
                    self.round_score[i] += 3
                    endingScores[str(i)] = 3
        
        
        
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
        
            
        
        
        return endingScores

    #获取玩家手牌
    def getPlayerHoldCards(self, pos):
        return copy.deepcopy(self.player_hand_cards[pos])
    
    #获取当前叫的主的花色
    def getMajor(self):
        return self.globalInfo["banking"]["major"]
    
    #获取当前的庄家
    def getBanker(self):
        return self.globalInfo["banking"]["banker"]
    
    #获取当前打第几级
    def getLevel(self):
        return self.globalInfo["level"]
    
    #获取当前轮次的状态
    def getStage(self):
        return self.globalInfo["stage"]
    
    #获取当前发牌状态发的牌
    def getDeliver(self):
        return self.globalInfo["deliver"]
    
    # 获取上一轮次的分数
    def getLastRoundScore(self):
        return self.get_score
    def getLastRoundScorePoke(self):
        return self.get_score>0 and self.get_score_pok or []
    
    
    # 获取错误码
    def getErrorCode(self):
        return self.erro_code
    
    # 获取当前轮次的序号
    def getStepCount(self):
        return self.step_count
        
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
        return self.globalInfo["publiccard"]
    
    #获取当前权位玩家位置
    def getPlayerPosition(self):
        return self.globalInfo["playerpos"]
    
    #获取上一轮出牌历史
    def getLastRoundPlayHistory(self):
        return self.globalInfo["history"][0]
    #获取上一轮的首出牌人位置
    def getLastRoundPlaySeat(self):
        return self.globalInfo["history"][2]
    #获取当前轮出牌历史
    def getCurrRoundPlayHistory(self):
        return self.globalInfo["history"][1]
    #获取当前轮的首出牌人位置
    def getCurrRoundPlaySeat(self):
        return self.globalInfo["history"][3]
    

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
        if play_type == "single":
            pok = play_poker
        elif play_type == "pair":
            pok = play_poker
        elif play_type == "tractor":
            pok = play_poker
        
        
        pok = copy.deepcopy(pok)
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
        
    #分析得到拖拉机牌型（大小王和级牌当然不会参与拖拉机）
    def parsePoker(self, pok, level):
    # poker: 甩牌牌型 list[int]
    # level & major: 级牌、主花牌
        outpok = []
        failpok = []
        count = Counter(pok)
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
                        tmp = [suit + self.pointorder[pos[i]], suit + self.pointorder[pos[i]], \
                               suit + self.pointorder[pos[i+1]], suit + self.pointorder[pos[i+1]]]
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

        finalpok = outpok

        return finalpok 
    
    #分析得到拖拉机牌型（大小王和级牌当然不会参与拖拉机）
    def parseTractorPoker(self, poker, level):
    # poker: 甩牌牌型 list[int]
    # level & self.Major: 级牌、主花牌
        if len(poker) == 0:
            return []
        

        pospok = []
        tractor = []
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
                        tractor.append(tmp)
                        suc_flag = False
                if suc_flag:
                    tractor.append(tmp)

        outpok = tractor
        # def find_long_subarrays(arr, r=4):
        #     subarrays = []
        #     n = len(arr)            
        #     for start in range(0, n - r + 1, 2):
        #         subarrays.append(arr[start:start+r])            
        #     return subarrays
        
        #找出超过3拖中的子拖拉机
        # sub_tractors = []
        # for _opok in outpok :
        #     _opok_len = len(_opok)        
        #     sub_tractors = []
        #     for i in range(2, 24, 2):
        #         if _opok_len == 4+i:
        #             for j in range(4, _opok_len-1, 2):
        #                 sub_tractor = find_long_subarrays(_opok[:], j)
        #                 sub_tractors.extend(sub_tractor)
        #             break
        # outpok.extend(sub_tractors)
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
                
        
        suit_comb_pok_list = {suit: [] for suit in __SUITSET__}
        for suit,poks in suit_pok_list.items():
            for n in range(1, len(poks)+1):
                comb_poks = list(combinations(poks, n))
                suit_comb_pok_list[suit].extend(comb_poks)
        for suit,poks in suit_comb_pok_list.items():
            poks.sort()
            outpok.extend(poks)
            
        for n in range(1, len(major_pok_list)+1):
            comb_poks = list(combinations(major_pok_list, n))
            comb_poks.sort()
            outpok.extend(comb_poks)

        outpok_final = []
        element_count = Counter(outpok)
        #去除重复
        for k, v in element_count.items():
            if v == 1:
                outpok_final.append(k)

        return outpok_final

    def call_Snatch(self, get_card, deck, called, snatched, level):
        # get_card: new card in this turn (int)
        # deck: your deck (list[int]) before getting the new card
        # called & snatched: player_id, -1 if not called/snatched
        # level: level
        # return -> list[int]
        response = []
        ## 目前的策略是一拿到牌立刻报/反，之后不再报/反
        ## 不反无主
        # deck_poker = [self.num2Poker(id) for id in deck]
        get_poker = self.num2Poker(get_card)
        if get_poker[1] == level:
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
    
    def getLegalPlayCard(self, history, deck, level):
        
        poker_deck = [self.num2Poker(id) for id in deck]
        # print("getLegalPlayCard.deck", poker_deck, '\r\n', deck)
        
        

        # 首发
        if len(history) == 0: 
            #test code
            # poker_deck = ['d0', 'c4', 'c6', 'cA', 'cA', 'cQ', 'c5', 'd3', 'c5', 'c4', 'd6', 'd3', 'h5', 'h0', 'c2', 'h7', 'hK', 'd6', 'h7', 'dQ', 'hJ', 'hQ', 's3', 'h3', 's8']           
            # level = '6'
            # self.Major = ['s2', 's3', 's4', 's5', 's7', 's8', 's9', 's0', 'sJ', 'sQ', 'sK', 'sA', 'h6', 'c6', 'd6', 's6', 'jo', 'Jo']
            # self.pointorder = ['2', '3', '4', '5', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']

            ret = self.getTypePoke(poker_deck, level)
            ret = self.PokerList2Num(ret, deck)
            # return ret[random.randint(0, len(ret))]
            return ret
        
        
        
        #test code
        # mj = self.globalInfo['banking']['major']
        # poker_deck = [mj+'5', mj+'5', mj+'0', mj+'6', mj+'7', mj+'9', mj+'8' ]
        # standard_poker = [mj+'3', mj+'3', mj+'6', mj+'6', mj+'5' ]
        # if False:

        standard_move = history[0]
        standard_poker = [self.num2Poker(id) for id in standard_move]
        if self.checkPokerType(standard_move, level) != "suspect": # 不是甩牌
            pok = [self.num2Poker(p) for p in standard_move] if type(standard_move[0]) == int else standard_move
            own_pok = [self.num2Poker(p) for p in deck]
            response = self.checkResUnSuspect(standard_poker, own_pok, level)
            if response:
                response = self.PokerList2Num(response, deck)
                return response
            else:
                response = self.checkResUnSuspect(standard_poker, own_pok, level)#test code
                #出的是主牌
                if standard_poker[0] in self.Major:
                    #print("major")
                    deck_Major = [pok for pok in poker_deck if pok in self.Major]
                    
                    if len(deck_Major) < len(standard_poker):#手上主牌数少于出牌数
                        my_major = copy.deepcopy(deck_Major)
                        deck_nMajor = [pok for pok in poker_deck if pok not in self.Major]
                        comb_nMajor = list(combinations(deck_nMajor, len(standard_poker) - len(deck_Major)))
                        out = []
                        for comb in comb_nMajor:
                            out.append(my_major + list(comb))
                        
                        
                         
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
                        response = self.checkResUnSuspect(standard_poker, own_pok, level)#test code
                        target_len = len(standard_poker)
                        #单牌
                        if target_len == 1:
                            out = self.getTypePoke(deck_Major, level, ['single'])
                        #对子
                        elif target_len == 2:
                            out = self.getTypePoke(deck_Major, level, ['pair'])
                            if len(out) == 0:
                                single_out = self.getTypePoke(deck_Major, level, ['single'])
                                out = list(combinations(single_out, target_len))
                        #拖拉机
                        else:
                            target_parse = self.parseTractorPoker(deck_Major, level)
                            out = [poks for poks in target_parse if len(poks) == target_len]
                            if len(out) == 0:#没有匹配的牌型
                                pair_out = self.getTypePoke(deck_Major, level, ['pair'])
                                pair_comb_out = list(combinations(pair_out, target_len//2))
                                out = [list(itertools.chain.from_iterable(pairs)) for pairs in pair_comb_out]
                                if len(out) == 0:#没有相同数量的对子组合
                                    pair_out = [pair for pairs in pair_out for pair in pairs]
                                        
                                    single_out = self.getTypePoke(deck_Major, level, ['single'])
                                    single_out = [item[0] for item in single_out if item[0] not in pair_out]
                                    remain_len = target_len - len(pair_out)
                                    pair_comb_out = list(combinations(single_out, remain_len))
                                    
                                    for single_comb in pair_comb_out:
                                        out.append(pair_out + list(single_comb))
                                            
                                    
                                     
                            
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
                    if len(deck_suit) <= len(standard_poker):#副牌数量不够
                        deck_unSuit = [pok for pok in poker_deck if pok not in deck_suit]
                        comb_unSuit = list(combinations(deck_unSuit, len(standard_poker) - len(deck_suit)))
                        out = []
                        for comb in comb_unSuit:
                            out.append(deck_suit + list(comb))
                            
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
                        response = self.checkResUnSuspect(standard_poker, own_pok, level)#test code
                        target_len = len(standard_poker)                        
                        if target_len == 1:#单牌
                            out = [[p] for p in deck_suit]
                        else:#对子，拖拉机
                            target_parse = self.parseTractorPoker(deck_suit, level)
                            target_parse.sort(key=lambda x: len(x), reverse=True)
                            out = []
                            if len(target_parse) == 0:
                                out = list(combinations(deck_suit, target_len))
                            else:
                                for poks in target_parse:
                                    if len(poks) == target_len:
                                        out.append(poks)
                                    else:
                                        out.append([poks[i-target_len:i]  for i in range(target_len, len(poks), target_len)])
                                    
        #甩牌
        else:
            #主牌甩牌
            if standard_poker[0] in self.Major:
                #print("major")
                deck_Major = [pok for pok in poker_deck if pok in self.Major]
                
                #手上的主牌不够，还需要加上副牌来凑数量
                if len(deck_Major) < len(standard_poker):
                    deck_unMajor = [pok for pok in poker_deck if pok not in self.Major]
                    comb_unMajor = list(combinations(deck_unMajor, len(standard_poker) - len(deck_Major)))
                    out = []
                    for comb in comb_unMajor:
                        out.append(deck_Major + list(comb))
                
                #数量正好相等
                elif len(deck_Major) == len(standard_poker):
                    out = [deck_Major]

                #手上的主牌数量够，甩牌
                else:
                    out = self.suspectMatchCard(deck_Major, standard_poker, level)
                    
            #副牌甩牌
            else:
                suit = standard_poker[0][0]
                deck_suit = [pok for pok in poker_deck if pok[0] == suit and pok[1] != level]

                #可出副牌数不够
                if len(deck_suit) < len(standard_poker):
                    deck_nsuit = [pok for pok in poker_deck if pok not in deck_suit]
                    out = []
                    comb_poks = list(combinations(deck_nsuit, len(standard_poker) - len(deck_suit)))
                    for poks in comb_poks:
                        out.append(deck_suit + list(poks))

                #数量正好相等
                elif len(deck_suit) == len(standard_poker):
                    out = [deck_suit]

                #可出副牌数够
                else:
                    out = self.suspectMatchCard(deck_suit, standard_poker, level)

        ret = []
        #test code
        for poks in out:
            _deck = copy.deepcopy(deck) + []
            for pok in poks:
                if type(pok) == list:
                    self.getLegalPlayCard(history, deck, level)
                    pass
                cardid = self.Poker2Num(pok, _deck)
                if cardid not in _deck:
                    print(self.Pokers2Num(poks,_deck))
                    self.getLegalPlayCard(history, deck, level)
                _deck.remove(cardid)
        
        
        
        #去除重复
        repeats = dict()
        repeats2 = dict()
        out1 = []
        for poks in out:
            outstr = ''.join(poks)
            if outstr not in repeats:
                out1.append(poks)
            else:
                repeats2[outstr] = poks
            repeats[outstr] = poks
        
        global __MAX_ACTION_NUM__#test code
        if __MAX_ACTION_NUM__ < len(out1):
            __MAX_ACTION_NUM__ = len(out1)
            print(f'最大动作空间：{__MAX_ACTION_NUM__}, pid={os.getpid()}')
            sout = sorted(out)
            if sout != out:
                for subout in out:
                    if subout not in sout:
                        pass


        ret = self.PokerList2Num(out1, deck)
        return ret
    
    #own_pok数量>=standard_poker，则直接匹配可出牌型
    def suspectMatchCard(self, own_pok, standard_pok, level):
        match_poks = []
        #甩牌里若有拖拉机，则也要优先匹配拖拉机
        standard_tractor = self.parseTractorPoker(standard_pok, level)
        if len(standard_tractor) > 0:
            for sttractor in standard_tractor:
                standard_pok1 = list(filter(lambda x: x not in sttractor, standard_pok))#去除standard_pok中的sttractor
                own_tractor = self.parseTractorPoker(own_pok, level)
                for owtractor in own_tractor:
                    if len(owtractor) == len(sttractor):
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
                    
    # initdata里包含了庄家、牌型、级牌
    # global里包含了主花色、级牌、庄家（覆盖）、报主情况
    def step(self, response=None):
        self.get_score = 0
        self.get_score_pok = []
        self.erro_code = 0
        self.step_count += 1
        
        try:
            if self.step_count == 0: # 刚开局
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
                
                
                
                #第一轮有机会写入initdata
                if "banking" not in self.globalInfo: # 没有规定摸牌方
                    first = 0
                    banking = {
                    "called": [],
                    "major": "",
                    "banker": -1
                    } # 初始化banking
                else:
                    first = self.globalInfo["banking"]["banker"]
                    # initdata["banker"] = first
                    banking = {
                    "called": [],
                    "major": "",
                    "banker": first
                    } # 初始化banking

                
                self.globalInfo["stage"] = "deal"
                self.globalInfo["deliver"] = [allocation[first][0]]#发的牌
                self.globalInfo["banking"] = banking
                self.globalInfo["playerpos"] = first
                
                self.player_hand_cards = [[] for _ in range(__PLAYER_COUNT__)]
                self.player_hand_cards[first].append(allocation[first][0])
                
            elif self.step_count <= 200: # 仍在发牌中，回合199是该阶段玩家最后一次反馈（lenLog = 200
                eventpoker = []
                # raw_response = full_input["log"][-1]
                currplayer = response[0]# int(list(raw_response.keys())[0])
                # = raw_response[str(currplayer)]["response"]
                # self.globalInfo = list(full_input["log"][-2]["output"]["content"].values())[0]["global"]
                banking = self.globalInfo["banking"] # 上回合发出的banking
                level = self.globalInfo["level"] #full_input["initdata"]["level"]
                repo = response[1] or [] # 这里有所改动，若玩家报主只需要发送报牌，若不报发送空列表[]即可
                if len(repo) > 0:
                    for poker in repo:
                        if poker not in self.player_hand_cards[currplayer]:                            
                            self.setError(currplayer, "NOT_YOUR_POKER")
                    eventpoker = repo
                    new_ = True
                    newbanking = self.checkBanker(repo, level, currplayer, banking, self.globalInfo["first_round"])
                    self.setMajor(newbanking["major"], self.globalInfo["level"])
                
                else: # 不报就不管
                    eventpoker = []
                    new_ = False
                    newbanking = banking
            
                if self.step_count < 200: # 给下一名玩家发牌
                    self.globalInfo["stage"] = "deal"                    
                    self.globalInfo["banking"] = newbanking
                    
            
                else: # 发牌的最后一轮，要确定庄家，下回合盖底牌
                    seedRandom = self.globalInfo["seed"]
                    random.seed(seedRandom)
                    if newbanking["banker"] == -1:
                        new_ = True
                        newbanking["banker"] = random.randint(0, __PLAYER_COUNT__)
                    if newbanking["major"] == "":
                        newbanking["major"] = __SUITSET__[random.randint(0, len(__SUITSET__))]
                        self.setMajor(newbanking["major"], self.globalInfo["level"])
                        
                    self.globalInfo["banking"] = newbanking
                    self.globalInfo["stage"] = "cover"
                    self.globalInfo["deliver"] = self.globalInfo["publiccard"]
                    self.globalInfo["playerpos"] = newbanking["banker"]
                
                if self.step_count%2 == 0 and self.step_count < 200:
                    player = self.globalInfo["playerpos"]
                    player = (player + 1) % __PLAYER_COUNT__
                    allocation = self.globalInfo["allocation"]
                    self.globalInfo["deliver"] = [allocation[player][self.step_count // (__PLAYER_COUNT__*2)]]
                    self.globalInfo["playerpos"] = player
                    
                    self.player_hand_cards[player].extend(self.globalInfo["deliver"])
                    self.step_count += 1
            
            elif self.step_count == 201: # 第101回合，庄家返回盖底牌
                allocation = self.globalInfo["allocation"]
                global_banker = self.globalInfo["banking"]["banker"]
                old_publiccard = self.globalInfo["publiccard"]
                banking = self.globalInfo["banking"]
                currplayer = response[0]
                if currplayer != global_banker: # 你的庄家在哪里？让他上前来！
                    self.setError(currplayer, "NOT_YOUR_TURN")
                repo = response[1]
                if type(repo) is not list:
                    self.setError(currplayer, "INVALID_FORMAT")
                if len(repo) != 8:
                    self.setError(currplayer, "INVALID_MOVE")
                big_hold = old_publiccard + allocation[currplayer]
                for pok in repo:
                    if pok not in big_hold:
                        self.setError(currplayer, "NOT_YOUR_POKER")
                    big_hold.remove(pok)
                
                self.player_hand_cards[currplayer] = big_hold
                self.globalInfo["publiccard"] = repo#更新压的底牌

                # 仍然给庄家发request
                self.globalInfo["stage"] = "play"
                self.globalInfo["history"] = [[],[], currplayer, currplayer]
                self.globalInfo["total_score"] = 0
                

            else: # 正式出牌（101回合开始, self.step_count=204）
                # old_alloc = full_input["initdata"]["allocation"]
                banker = self.globalInfo["banking"]["banker"]
                major = self.globalInfo["banking"]["major"]
                # self.setMajor(major, self.globalInfo["level"])
                # old_publiccard = full_input["initdata"]["publiccard"]
                # new_publiccard = full_input["log"][201][str(banker)]["response"]
                # big_hold = old_alloc[banker] + old_publiccard
                old_score = self.globalInfo["total_score"]
                # for pok in new_publiccard:
                #     big_hold.remove(pok)
                # new_alloc = copy.deepcopy(old_alloc) + []
                # new_alloc[banker] = big_hold
                
                '''for turn_id in range(204, self.step_count - 1, 2): # 还原本轮手牌（刚刚接收到的一回合信息不处理）
                    request = full_input["log"][turn_id]
                    #player = int(list(response.keys())[0])
                    #move = response[str(player)]["response"] 
                    # 由于甩牌有失败的可能性，这里必须使用记录的history恢复局面
                    player_prov = int(list(request["output"]["content"].keys())[0])
                    hist_prov = request["output"]["content"][str(player_prov)]["history"]
                    if len(hist_prov[1]) > 0:
                        move = hist_prov[1][-1] #此处功能未测试
                        player = (player_prov-1) % 4
                    else:
                        move = hist_prov[0][-1]
                        player = (hist_prov[2]+3) % 4
                    
                    for pok in move:
                        new_alloc[player].remove(pok)'''
                
                #print("recover: Normal")

                # latest_response = full_input["log"][-1]
                if None == response:
                    pass
                currplayer = response[0]
                curr_move = response[1]
                # if type(curr_move) is not list:
                #     self.setError(currplayer, "INVALID_FORMAT")
                # if len(curr_move) == 0:
                #     self.setError(currplayer, "INVALID_MOVE")
                # latest_request = full_input["log"][-2]
                history = self.globalInfo["history"]
                for pok in curr_move:
                    if pok not in self.player_hand_cards[currplayer]:
                        self.setError(currplayer, "NOT_YOUR_POKER")
                play_move = [self.num2Poker(p) for p in curr_move]
                outpok = self.checkLegalMove(play_move, self.globalInfo["level"], major, currplayer, history[1], self.player_hand_cards, banker)
                # collect history
                outid = []
                for pok in outpok:
                    id = self.Poker2Num(pok, self.player_hand_cards[currplayer])
                    outid.append(id)
                    self.player_hand_cards[currplayer].remove(id)
                    # del self.player_hand_cards[currplayer][self.player_hand_cards[currplayer].index(id)]
                
                #print("move: Normal")

                new_history = history[1]
            
                if len(new_history) == 0:
                    history[3] = currplayer
                if len(new_history) < 3:
                    nextplayer = (currplayer + 1) % __PLAYER_COUNT__
                new_history.append(outid)
                self.player_played_cards[currplayer].extend(outid)
                old_history = history[0]
            
                if len(new_history) == 4: # 本回合为该轮最后一次出牌
                    winner = self.checkWinner(history[1], currplayer, self.globalInfo["level"], major, banker)
                    nextplayer = winner
                    old_history = new_history
                    new_history = []
                    history[2] = history[3] + 0
                    history[3] = winner

                    if len(self.player_hand_cards[currplayer]) == 0: # 本局结束
                        # 扣底
                        if self.checkPokerType(history[1][0], self.globalInfo["level"]) != "suspect":
                            mult = len(history[1][0])
                        else:
                            divided, _ = self.checkThrow(history[1][0], [[]], (currplayer-3)%4, self.globalInfo["level"], major, check=False)
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

                        endingScores = self.EndGame(banker, new_score)
                        
                        self.globalInfo["stage"] = "finish"
                        return self.__EndRound__()

                
                    if self.get_score != 0: # 非终止回合出现分数变动，说明甩牌失败
                        pass
                    new_score = old_score + self.get_score#get_score为负数就是罚分
                    self.globalInfo["total_score"] = new_score
                    if new_score < 0:#test code
                        pass

                history[0] = old_history
                history[1] = new_history
                self.globalInfo["history"] = history
                self.globalInfo["playerpos"] = nextplayer
                self.globalInfo["stage"] = len(new_history)==0 and "roundend" or "play"

        except Exception as e:
            traceback.print_exc()
            raise e

        

    def __EndRound__(self):
        self.step_count = -1

    #小局结束
    def isInningEnd(self):
        return self.step_count == -1
    
    #大局结束
    def isFinalEnd(self):
        return self.globalInfo["stage"] == 'finalend'
     
    
    def getPlayedCards(self):
        return self.player_played_cards
        

__MAX_ACTION_NUM__ = 0
def run(env):
    response = None
    level = env.getLevel()
    stage = env.getStage()
    if stage == "finish":
        pass
    elif stage == "deal":
        get_card = env.getDeliver()[0]
        called = env.getCalled()
        snatched = env.getSnatched()        
        hold = env.getPlayerHoldCards(env.getPlayerPosition())
        response = [env.getPlayerPosition(), env.call_Snatch(get_card, hold, called, snatched, level)]
    elif stage == "cover":
        publiccard = env.getPublicCards()
        hold = env.getPlayerHoldCards(env.getBanker())
        response = [env.getBanker(), env.cover_Pub(publiccard, hold)]
    elif stage == "play" or stage == "roundend":
        history_curr = env.getCurrRoundPlayHistory()
        hold = env.getPlayerHoldCards(env.getPlayerPosition())
        
        
        playedCards = env.getLegalPlayCard(history_curr, hold, level)
        if len(playedCards) == 0:#test code
            playedCards = env.getLegalPlayCard(history_curr, hold, level)
            pass
        
        playedCards = playedCards[random.randint(0, len(playedCards)-1)]#test code
        if type(playedCards) is not list:#test code
            playedCards = env.getLegalPlayCard(history_curr, hold, level)
        response = [env.getPlayerPosition(), playedCards]
    
    return response

def runGame():
    for _ in range(100000):
        # print(f'start round,{_}, pid={os.getpid()}\r\n')
        env = tractorGame()
        envs[os.getpid()] = env
        response = None     
        while True:
            env.step(response)
            if env.isFinalEnd():
                # print(env.globalInfo)
                break
            response = run(env)



if __name__ == '__main__':
    matrix = np.arange(54*2, dtype=np.int8)
    arr = [0,2,4,8,52]
    matrix[arr] = -1
    matrix = np.insert(matrix, 54, [-2,-2])
    matrix = np.insert(matrix, 110, [-2,-2])
    matrix = matrix.reshape(2,14,4)
    matrix = np.transpose(matrix, (0,2,1))
    
    # 创建一个新的列顺序数组
    new_order = list(range(matrix.shape[2]))
    new_order.remove(2)
    new_order.insert(12, 2)

    # 使用新的列顺序重新排列矩阵
    print(matrix)
    matrix = matrix[:, :, new_order]
    print(matrix)

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

