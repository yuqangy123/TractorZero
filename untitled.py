import unittest
from collections import Counter
from itertools import combinations
from tractor_botzone import tractorGame


def checkResUnSuspect_repsect(self, play_pok, own_pok, level): # poker: list[int]
    poker_len = len(play_pok)
    suit = play_pok[0][0]
    # typoker = self.checkPokerType(play_pok, level)
    ret = {'fixedcard':[], 'discard':[]}#分为固定牌和垫牌
    #出的是主牌
    if play_pok[0] in self.Major:
        major_pok = [pok for pok in own_pok if pok in self.Major or pok[1] == level]
        my_pok_count = Counter(major_pok)

        #手上没有主牌
        if len(my_pok_count) == 0:
            ret['discard'] = own_pok

        #单张
        elif poker_len == 1:
            ret['discard'] = major_pok
            return ret
        #对子
        elif poker_len == 2:
            ret['fixedcard'] = [[k,k] for k,v in my_pok_count.items() if v == 2]
            #没有对子，则用固定牌+垫牌组合
            if len(ret['fixedcard']) == 0:
                combpairs = list(combinations(major_pok, 2))
                for pairs_pok in combpairs: ret['fixedcard'].append(list(pairs_pok))
                #主牌不够
                if len(ret['fixedcard']) == 0:
                    ret['fixedcard'] = [major_pok]
                    ret['discard'] = [k for k,v in Counter(own_pok).items() if k not in self.Major]
    
            
        # 主牌拖拉机
        else:
            deck_Major = major_pok
            ret['fixedcard'] = self.parseTractorPoker(deck_Major, level, poker_len)

            #没有拖拉机，看有没有对子
            if len(ret['fixedcard']) == 0:                  
                pairspok = [[p,p] for p,v in my_pok_count.items() if v == 2]
                #对子数>=出牌数
                if len(pairspok) >= poker_len//2:
                    combpairs = list(combinations(pairspok, poker_len//2))
                    for pairs_pok in combpairs: ret['fixedcard'].append([pairs_pok[0][0]]*2+[pairs_pok[1][0]]*2)
                        
                #对子不够，单张来凑
                else:
                    singlepok = [p for p,v in my_pok_count.items() if v == 1]
                    # if len(singlepok) + len(pairspok)*2 >= poker_len:
                    singleCombpairs = list(combinations(singlepok, poker_len-len(pairspok)*2))
                    pairspok = [p for p,v in my_pok_count.items() if v == 2]*2
                    for pairs_pok in singleCombpairs: ret['fixedcard'].append(pairspok+list(pairs_pok))
                    #对子加单张都不够
                    if len(ret['fixedcard']) == 0:
                        ret['fixedcard'] = [deck_Major]
                        ret['discard'] = [pok for pok in own_pok if pok not in self.Major]
            
            
            
    #出的是副牌
    else:
        suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
        my_pok_count = Counter(suit_pok)

        #手上没有副牌
        if len(my_pok_count) == 0:
            ret['discard'] = own_pok

        #单张
        elif poker_len == 1:
            ret['discard'] = suit_pok

        #对子
        elif poker_len == 2:
            ret['fixedcard'] = [[k,k] for k,v in my_pok_count.items() if v == 2]
            #没有对子，则用固定牌+垫牌组合
            if len(ret['fixedcard']) == 0:
                combpairs = list(combinations(suit_pok, 2))
                for pairs_pok in combpairs: ret['fixedcard'].append(list(pairs_pok))
                #主牌不够
                if len(ret['fixedcard']) == 0:
                    ret['fixedcard'] = [suit_pok]
                    ret['discard'] = [k for k,v in Counter(own_pok).items() if k not in self.Major]
            
            
        # 拖拉机
        else:
            deck_suit = suit_pok
            ret['fixedcard'] = self.parseTractorPoker(deck_suit, level, poker_len)

            #没有拖拉机，看有没有对子
            if len(ret['fixedcard']) == 0:                  
                pairspok = [[p,p] for p,v in my_pok_count.items() if v == 2]
                #对子数>=出牌数
                if len(pairspok) >= poker_len//2:
                    combpairs = list(combinations(pairspok, poker_len//2))
                    for pairs_pok in combpairs: ret['fixedcard'].append([pairs_pok[0][0]]*2+[pairs_pok[1][0]]*2)
                        
                #对子不够，单张来凑
                else:
                    singlepok = [p for p,v in my_pok_count.items() if v == 1]
                    # if len(singlepok) + len(pairspok)*2 >= poker_len:
                    singleCombpairs = list(combinations(singlepok, poker_len-len(pairspok)*2))
                    pairspok = [p for p,v in my_pok_count.items() if v == 2]*2
                    for pairs_pok in singleCombpairs: ret['fixedcard'].append(pairspok+list(pairs_pok))
                    #对子加单张都不够
                    if len(ret['fixedcard']) == 0:
                        ret['fixedcard'] = [deck_suit]
                        ret['discard'] = [pok for pok in own_pok if pok not in deck_suit]
    return ret
        

# 假设 tractorGame 类和相关常量已经定义

class TestCheckResUnSuspectRepsect(unittest.TestCase):

    def setUp(self):
        self.game = tractorGame()
        # 模拟一个简单的 Major 列表和 pointorder
        self.game.Major = ['S2', 'H2', 'D2', 'C2', 'S3', 'H3', 'D3', 'C3', 'Jo', 'jo']
        self.game.pointorder = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        # 注意：在实际测试中，可能需要根据函数内部逻辑调整 Major 和 pointorder

    def test_major_no_major_in_hand(self):
        """场景1: 打出主牌，手牌中没有主牌"""
        play_pok = ['S2']
        own_pok = ['H4', 'D5']
        level = '4'
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertEqual(result['fixedcard'], [])
        self.assertEqual(sorted(result['discard']), sorted(['H4']))

    def test_major_single(self):
        """场景2: 打出主牌单张，手牌中有主牌x"""
        play_pok = ['S2']
        own_pok = ['S2', 'H4', 'D5']
        level = '4'
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertEqual(result['fixedcard'], [])
        self.assertEqual(sorted(result['discard']), sorted(['S2','H4']))

    def test_major_pair(self):
        """场景3: 打出主牌对子，手牌中有主牌对子"""
        play_pok = ['S2', 'S2']
        own_pok = ['S2', 'S2', 'H4', 'D5']
        level = '4'
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertIn(['S2', 'S2'], result['fixedcard'])

    def test_major_tractor(self):
        """场景4: 打出主牌拖拉机，手牌中有主牌拖拉机"""
        # 假设 S2S2S3S3 是一个拖拉机
        play_pok = ['S2', 'S2', 'S3', 'S3']
        own_pok = ['S2', 'S2', 'S3', 'S3', 'H4', 'D5']
        level = '4'
        # 需要模拟 parseTractorPoker 的返回值
        original_parse = self.game.parseTractorPoker
        self.game.parseTractorPoker = lambda deck, lvl: [['S2', 'S2', 'S3', 'S3']] if 'S2' in deck and 'S3' in deck else []
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertIn(['S2', 'S2', 'S3', 'S3'], result['fixedcard'])
        self.game.parseTractorPoker = original_parse # 恢复原始函数

    def test_major_tractor_no_tractor_has_pairs(self):
        """场景5: 打出主牌拖拉机，手牌中没有拖拉机但有对子"""
        play_pok = ['S2', 'S2', 'S3', 'S3']
        own_pok = ['S2', 'S2', 'H4', 'H4', 'D5']
        level = '5'
        original_parse = self.game.parseTractorPoker
        self.game.parseTractorPoker = lambda deck, lvl: []
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        # 检查是否生成了由对子组成的牌型x
        self.assertTrue(any(len(set(cards)) == 2 and len(cards) == 4 for cards in result['fixedcard']))
        self.game.parseTractorPoker = original_parse

    def test_major_tractor_no_tractor_no_pairs(self):
        """场景6: 打出主牌拖拉机，手牌中没有拖拉机和对子"""
        play_pok = ['S2', 'S2', 'S3', 'S3']
        own_pok = ['S2', 'H4', 'D5', 'C6']
        level = '5'
        original_parse = self.game.parseTractorPoker
        self.game.parseTractorPoker = lambda deck, lvl: []
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        # fixedcard 应该包含所有主牌，discard 包含非主牌x
        self.assertEqual(sorted([card for sublist in result['fixedcard'] for card in sublist]), sorted(['S2']))
        self.assertEqual(sorted(result['discard']), sorted(['H4', 'D5', 'C6']))
        self.game.parseTractorPoker = original_parse

    def test_suit_no_suit_in_hand(self):
        """场景7: 打出副牌，手牌中没有同花色牌"""
        play_pok = ['H4']
        own_pok = ['S2', 'D5']
        level = '4'
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertEqual(result['fixedcard'], [])
        self.assertEqual(sorted(result['discard']), sorted(own_pok))

    def test_suit_single(self):
        """场景8: 打出副牌单张，手牌中有同花色牌"""
        play_pok = ['H4']
        own_pok = ['H4', 'H5', 'S2', 'D5']
        level = '6'
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertEqual(result['fixedcard'], [])
        self.assertEqual(sorted(result['discard']), sorted(['H4', 'H5']))

    def test_suit_pair(self):
        """场景9: 打出副牌对子，手牌中有同花色对子"""
        play_pok = ['H4', 'H4']
        own_pok = ['H4', 'H4', 'H5', 'S2', 'D5']
        level = '6'
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertIn(['H4', 'H4'], result['fixedcard'])

    def test_suit_tractor(self):
        """场景10: 打出副牌拖拉机，手牌中有同花色拖拉机"""
        play_pok = ['H4', 'H4', 'H5', 'H5']
        own_pok = ['H4', 'H4', 'H5', 'H5', 'H6', 'S2', 'D5']
        level = '6'
        original_parse = self.game.parseTractorPoker
        self.game.parseTractorPoker = lambda deck, lvl: [['H4', 'H4', 'H5', 'H5']] if 'H4' in deck and 'H5' in deck else []
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        self.assertIn(['H4', 'H4', 'H5', 'H5'], result['fixedcard'])
        self.game.parseTractorPoker = original_parse

    def test_suit_tractor_no_tractor_has_pairs(self):
        """场景11: 打出副牌拖拉机，手牌中没有拖拉机但有对子"""
        play_pok = ['H4', 'H4', 'H5', 'H5']
        own_pok = ['H4', 'H4', 'D4', 'D4', 'S2', 'D5']
        level = '6'
        original_parse = self.game.parseTractorPoker
        self.game.parseTractorPoker = lambda deck, lvl: []
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        # 检查是否生成了由对子组成的牌型
        self.assertTrue(any(len(set(cards)) == 1 and len(cards) == 2 for cards in result['fixedcard']))
        self.assertEqual(sorted(result['discard']), sorted(['D4', 'D4', 'S2', 'D5']))
        self.game.parseTractorPoker = original_parse

    def test_suit_tractor_no_tractor_no_pairs(self):
        """场景12: 打出副牌拖拉机，手牌中没有拖拉机和对子"""
        play_pok = ['H4', 'H4', 'H5', 'H5']
        own_pok = ['H4', 'H6', 'S2', 'D5']
        level = '6'
        original_parse = self.game.parseTractorPoker
        self.game.parseTractorPoker = lambda deck, lvl: []
        result = self.game.checkResUnSuspect_repsect(play_pok, own_pok, level)
        # fixedcard 应该包含所有同花色牌，discard 包含非同花色牌
        self.assertEqual(sorted([card for sublist in result['fixedcard'] for card in sublist]), sorted(['H4']))
        self.assertEqual(sorted(result['discard']), sorted(['S2', 'D5', 'H6']))
        self.game.parseTractorPoker = original_parse

if __name__ == '__main__':
    unittest.main()
