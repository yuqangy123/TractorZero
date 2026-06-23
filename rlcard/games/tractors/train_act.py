from rlcard.games.tractors.env.env import Env
from .env.env import Env
from .env.env_utils import Environment#*


def create_env(flags):
    return Env(flags.objective)

def act(i, device, batch_queues, model, flags):
    positions = ['banker', 'idler_up', 'idler_down', 'bid']
    try:
        T = flags.unroll_length
        T_C = flags.unroll_length_coach
        print('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_adp_buf = {p: [] for p in positions}
        target_wp_buf = {p: [] for p in positions}
        target_wp_bid_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        obs_x_batch_buf = {p: [] for p in positions}
        # coach_landlord_cards_buf = []
        # coach_landlord_down_cards_buf = []
        # coach_landlord_up_cards_buf = []
        # coach_target_buf = []


        position, obs, env_output = env.initial(model, device, flags=flags)
        # pid = threading.get_ident()
        
        while True:
            init_site_cards = [env_output['game_infoset'].all_handcards['first'], env_output['game_infoset'].all_handcards['second'], env_output['game_infoset'].all_handcards['third']]
            three_landlord_cards = env_output['game_infoset'].three_landlord_cards
            while True:
                if len(obs['legal_actions']) > 1:
                    with torch.no_grad():
                        agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags, game_infoset=env_output['game_infoset'])                        
                    _action_idx = int(agent_output['action'].cpu().detach().numpy())
                    action = obs['legal_actions'][_action_idx]

                    if position in ['first', 'second', 'third']:
                        obs_z_buf[position].append(
                            torch.vstack((torch.full((1, 54), action[0]), env_output['obs_z'])).float())
                    else:
                        obs_z_buf[position].append(
                            torch.vstack((_cards2tensor(action).unsqueeze(0), env_output['obs_z'])).float())
                else:
                    action = obs['legal_actions'][0]
                    if position in ['first', 'second', 'third']:
                        obs_z_buf[position].append(
                            torch.vstack((torch.full((1, 54), action[0]), env_output['obs_z'])).float())
                    else:
                        obs_z_buf[position].append(
                            torch.vstack((_cards2tensor(action).unsqueeze(0), env_output['obs_z'])).float())

                x_batch = env_output['obs_x_no_action'].float()
                obs_x_batch_buf[position].append(x_batch)
                size[position] += 1
                
                position, obs, env_output = env.step(action, model, device, flags=flags)
                if env_output['done'] or env_output['draw']:
                    for p in positions:
                        diff = size[p] - len(target_adp_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)
                            if env_output['draw']:
                                episode_return = 0.
                                wp_return = 0.
                                wp_bid = [0, 0, 1]
                            else:
                                episode_return = env_output['episode_return']["play"][p]
                                wp_return = 1. if episode_return > 0. else -1.
                                wp_bid = [1, 0, 0] if episode_return > 0. else [0, 1, 0]
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_adp_buf[p].extend([episode_return for _ in range(diff)])
                            target_wp_buf[p].extend([wp_return for _ in range(diff)])
                            target_wp_bid_buf[p].extend([torch.tensor(wp_bid) for _ in range(diff)])
                    
                    if not env_output['draw']:
                        landlord_site = 0
                        win_count = 0
                        for index, site in enumerate(['first', 'second', 'third']):
                            if env_output['episode_return']["play"][site]>0.:
                                win_count += 1
                                landlord_site = index
                        target = 1.
                        if win_count == 2:
                            target = 0.
                            for index, site in enumerate(['first', 'second', 'third']):
                                if env_output['episode_return']["play"][site]<0.:
                                    landlord_site = index
                                    break
                        init_site_cards[landlord_site] += three_landlord_cards
                        # coach_landlord_cards_buf.append(cards2array(init_site_cards[landlord_site]))
                        # coach_landlord_down_cards_buf.append(cards2array(init_site_cards[(landlord_site+1)%3]))
                        # coach_landlord_up_cards_buf.append(cards2array(init_site_cards[(landlord_site+2)%3]))
                        # coach_target_buf.append(target)
                        # print(landlord_site, target, \
                        #       env_output['episode_return']["play"]['first'], \
                        #         env_output['episode_return']["play"]['second'], \
                        #             env_output['episode_return']["play"]['third'],\
                        #                 init_site_cards,  '\n')
                    break

            for p in positions:
                if size[p] > T:
                    batch_queues[p].put({
                        "done": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in done_buf[p][:T]]),
                        "episode_return": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in episode_return_buf[p][:T]]),
                        "target_adp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_adp_buf[p][:T]]),
                        "target_wp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_wp_buf[p][:T]]),
                        "target_wp_bid": torch.stack(
                            [ndarr.clone().detach() for ndarr in target_wp_bid_buf[p][:T]]),
                        "obs_z": torch.stack([ndarr.clone().detach() for ndarr in obs_z_buf[p][:T]]),
                        "obs_x_batch": torch.stack(
                            [ndarr.clone().detach() for ndarr in obs_x_batch_buf[p][:T]]),
                    })
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_adp_buf[p] = target_adp_buf[p][T:]
                    target_wp_buf[p] = target_wp_buf[p][T:]
                    target_wp_bid_buf[p] = target_wp_bid_buf[p][T:]
                    obs_x_batch_buf[p] = obs_x_batch_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T
            # if len(coach_target_buf) > T_C:
            #     batch_queues['coach'].put({
            #         "target": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in coach_target_buf[:T_C]]),
            #         "landlord": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in coach_landlord_cards_buf[:T_C]]),
            #         "landlord_down": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in coach_landlord_down_cards_buf[:T_C]]),
            #         "landlord_up": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in coach_landlord_up_cards_buf[:T_C]]),
            #     })
            #     coach_target_buf = coach_target_buf[T_C:]
            #     coach_landlord_cards_buf = coach_landlord_cards_buf[T_C:]
            #     coach_landlord_down_cards_buf = coach_landlord_down_cards_buf[T_C:]
            #     coach_landlord_up_cards_buf = coach_landlord_up_cards_buf[T_C:]            

    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    except Exception as e:
        print('Exception in worker process %i', i)
        log.error('Exception in worker process %i', i)
        traceback.print_exc()        
        raise e    
    print('act over')

def step(i, device, actor, batch_queues, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    
    try:
        T = flags.unroll_length
        print('(TractorsEnv)Device %s Actor %i started.', str(device), i)

        env = TractorsEnv(flags)
        # env = Environment(env, device)

        '''逐步迭代:
        1.从残局预测开始训练（可见信息最丰富），然后逐步加入更多的不可见信息。'''
        threshold_handcards = 5
        
        '''最终的回放buff，形状[T,15,4,2,4,15]，
        每个buff元素是一个形状为[15,4,2,4,15]的矩阵，15是轮数(card_play_action_seq)，后面是一轮的出牌'''
        
        positions = ['banker', 'idler']
        
        #经验缓存
        hand_cards_buf = {p: [] for p in positions}
        major_cards_buf = {p: [] for p in positions}
        public_cards_buf = {p: [] for p in positions}
        played_score_cards_buf = {p: [] for p in positions}
        remain_score_cards_buf = {p: [] for p in positions}
        card_play_action_seq_buf = {p: [] for p in positions}
        round_play_cards_buf = {p: [] for p in positions}
        last_play_cards_buf = {p: [] for p in positions}
        played_cards_buf = {p: [] for p in positions}
        mask_cards_buf = {p: [] for p in positions}
        my_seat_buf = {p: [] for p in positions}
        play_rights_seat_buf = {p: [] for p in positions}
        banker_seat_buf = {p: [] for p in positions}
        score_buf = {p: [] for p in positions}
        win_score_buf = {p: [] for p in positions}
        num_cards_left_buf = {p: [] for p in positions}
        predict_action_buf = {p: [] for p in positions}
        reward_buf = {p: [] for p in positions}
               

        while True:
            response = []
            
            #新一局开始
            env.reset()
            
            '''出牌阶段的回放经验'''
            #历史出牌序列信息
            card_play_action_seq = []

            #当前回合出牌序列信息
            round_play_cards = []
            
            #最后一次出牌
            last_round_play_cards = []
            
            hand_cards = []

            ######################################################################################
            #报主缓存
            bid_trajectory = []
            bid_score = 0
            
            #出牌阶段的缓存            
            played_cards = []
            played_score_cards = []
            remain_score_cards = []

            #底牌缓存
            public_cards = None

            #当前得分
            round_score = 0
            game_score = 0
            
            #当前牌权位置
            play_rights = 0            

            mask_cards = []
            ######################################################################################

            #回合出牌緩存
            round_cnt = 0
            play_counts = 0

            # round_player_remain_card_num = [0,0,0,0]
            record_index = False#是否开始记录轨迹
            inning_major = None
            inning_level = None
            
            #主牌
            major_cards_mtx = None
            
            infoset = [{} for _ in __PLAYER_COUNT__]

            
            while True:
                env.step(response)
                
                err = env.getError()
                if len(err)>0:
                    print(err[len(err)-1])
                    env.reset()
                    env.step(response)

                stage = env.getStage()
                #叫分阶段
                if stage == "bid":
                    play_pos = env.getPlayerPosition()                    
                    
                    if np.random.rand() < 0.5:
                        response = [play_pos, 0]
                    else:
                        bid_opt = math.max(0, (80/5 - len(bid_trajectory)))
                        response = [play_pos, random.randint(0, bid_opt*5)]
                        if response[1] > 0:
                            major_color = random.sample(__SUITSET__, 1)
                            response[2] = major_color
                            bid_score = response[1]
                    bid_trajectory.append(response)
                    
                    
                    # get_card = env.getDeliver()[0]
                    # called = env.getCalled()
                    # snatched = env.getSnatched()
                    # level = env.getLevel()
                    # play_pos = env.getPlayerPosition()                    
                    # hold = env.getPlayerHandCards(play_pos)
                    # ret = env.call_Snatch(get_card, hold, called, snatched, level)
                    # response = [play_pos, ret]
                    # if len(ret) > 0:
                    #     bid_trajectory.append(response)
                    
                #埋牌阶段(无埋牌阶段)
                elif stage == "cover":
                    # cover_seat = response[0]
                    # cover_cards = response[1]
                    # major_color = response[2]
                    
                    banker = env.getBanker()
                    hold_cards = env.getPlayerHandCards(banker)
                    cover_seat = banker                    
                    cover_cards = random.sample(hold_cards, 8)
                    response = [cover_seat, cover_cards]
                
                
                elif stage == 'startplay':
                    #self, public_cards, hold_card, own_seat, bid_history, level, major
                    # agent_output = actor.coverCard(publiccard, hold_cards, bid_trajectory, inning_major, inning_level)
                    
                    public_cards = cards2matrix(env.getPublicCards())
                    banker = env.getBanker()
                    
                    #桌面分
                    played_score_cards = cards2matrix([])
                    #隐藏信息分
                    remain_score_cards = [s + '5' for s in __SUITSET__] + [s + '0' for s in __SUITSET__] + [s + 'K' for s in __SUITSET__]
                    remain_score_cards = env.Pokers2Num(remain_score_cards,[i for i in range(54)])
                    remain_score_cards.extend([c+54 for c in remain_score_cards])
                    remain_score_cards = cards2matrix(remain_score_cards)
                    #已出牌
                    played_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    
                    #叫主轨迹信息
                    # history_bid_card = [cards2matrix([]) for _ in range(2)]
                    # history_bid_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(2)]
                    # if len(bid_trajectory)>2:
                    #     KeyError('len(bid_trajectory)>2')
                    # for i,traj in enumerate(bid_trajectory):
                    #     history_bid_seat[i][traj[0]-1] = 1.0
                    #     history_bid_card[i] = cards2matrix(traj[1])
                    
                    #回合出牌序列信息
                    round_play_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    round_play_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)]
                    
                    #上回合出牌序列信息
                    last_round_play_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    
                    card_play_action_seq = []
                    
                    #手牌
                    hand_cards = [cards2matrix([env.getPlayerHandCards(i)]) for i in range(__PLAYER_COUNT__)]
                    #mask隐蔽牌
                    mask_cards = [cards2matrix([1 for _ in range(__CARDS_NUM__)]) for _ in range(__PLAYER_COUNT__)]
                    #去掉自己的手牌
                    for i in range(len(mask_cards)):
                        mask_cards[i][hand_cards[i] == 1] = 0
                    mask_cards[banker][public_cards == 1] = 0
                    
                    #极牌
                    major_cards_mtx = cards2matrix(env.getMajorCards())

                    #牌权
                    play_rights = banker
                    
                    round_cnt = 0
                    play_counts = 0

                    #场面数据
                    # hand_cards 			[2,4,15]                #我的手牌
                    # major_cards			[2,4,15]                #历史级牌
                    # public_cards		[2,4,15]                    #底牌，只有banker可见
                    # played_score_cards	[2,4,15]                #已出分数牌
                    # remain_score_cards	[2,4,15]                #剩余分数牌
                    # card_play_action_seq	[60,PLAYER_COUNT,2,4,15]#历史出牌序列
                    # round_play_cards	[PLAYER_COUNT,2,4,15]       #当前回合出牌序列
                    # last_round_play_cards		[PLAYER_COUNT,2,4,15]   #上次回合出牌序列                    
                    # played_cards		[PLAYER_COUNT,2,4,15]       #已出牌
                    # mask_cards			[PLAYER_COUNT,2,4,15]   #当前玩家未知牌mask
                    # my_seat				[PLAYER_COUNT]          #我的座位号
                    # banker_seat			[PLAYER_COUNT]          #庄家座位号
                    # score				[40]                        #当前捡到的分数（1个占位为5分）
                    # win_score_limit         [40]                        #赢的分数线
                    # num_cards_left    []
                    banker = env.getBanker()
                    
                    for p in __PLAYER_COUNT__:
                        infoset[p]['hand_cards'] = hand_cards[p]
                        infoset[p]['major_cards'] = major_cards_mtx
                        infoset[p]['public_cards'] = public_cards
                        infoset[p]['played_score_cards'] = played_score_cards
                        infoset[p]['remain_score_cards'] = remain_score_cards
                        infoset[p]['card_play_action_seq'] = card_play_action_seq
                        infoset[p]['round_play_cards'] = round_play_cards
                        infoset[p]['last_round_play_cards'] = last_round_play_cards
                        infoset[p]['played_cards'] = played_cards[p]
                        infoset[p]['mask_cards'] = mask_cards[p]
                        infoset[p]['my_seat'] = get_one_hot_array(p+1, __PLAYER_COUNT__)
                        infoset[p]['play_rights_seat'] = get_one_hot_array(play_rights+1, __PLAYER_COUNT__)
                        infoset[p]['banker_seat'] = get_one_hot_array(banker+1, __PLAYER_COUNT__)
                        infoset[p]['score'] = get_one_hot_array(game_score//5, __MAX_SCORE__)
                        infoset[p]['win_score_limit'] = get_one_hot_array(bid_score//5, __MAX_SCORE__)
                        infoset[p]['num_cards_left'] = [get_one_hot_array(env.getPlayerLeftHandCards(i)+1, __HAND_CARD_NUM__) for i in __PLAYER_COUNT__]
                    
                    
                #出牌阶段
                elif stage == "play":
                    p = env.getPlayerPosition()
                    banker = env.getBanker()
                    hold = env.getPlayerHandCards(p)
                    infoset[p]['hand_cards'] = cards2matrix(hold)
                    infoset[p]['my_seat'] = get_one_hot_array(p+1, __PLAYER_COUNT__)
                    infoset[p]['played_score_cards'] = played_score_cards
                    infoset[p]['remain_score_cards'] = remain_score_cards
                    infoset[p]['card_play_action_seq'] = card_play_action_seq
                    infoset[p]['round_play_cards'] = round_play_cards
                    infoset[p]['played_cards'] = played_cards[p]
                    infoset[p]['mask_cards'] = mask_cards[p]
                    infoset[p]['play_rights_seat'] = get_one_hot_array(play_rights+1, __PLAYER_COUNT__)
                    infoset[p]['score'] = get_one_hot_array(game_score//5, __MAX_SCORE__)
                    history_curr = env.getCurrRoundPlayHistory()
                    infoset[p]['legal_actions'] = env.getLegalPlayCard(history_curr, hold, env.getLevel())
                    infoset[p]['num_cards_left'] = [get_one_hot_array(env.getPlayerLeftHandCards(i)+1, __HAND_CARD_NUM__) for i in __PLAYER_COUNT__]
                    
                    obs = get_obs(infoset)
                    
                    
                    #数据合法性验证 test code
                    # for trj in range(len(card_play_action_seq)):
                    #     count = 0
                    #     for seat in range(4):
                    #         count += np.sum(card_play_action_seq[trj][seat])
                    #     if count%2 != 0:
                    #         raise ValueError(card_play_action_seq[trj][seat])
                    
                    
                   
                    #执行游戏出牌
                    history_curr = env.getCurrRoundPlayHistory()
                    hold = env.getPlayerHandCards(play_pos)
                    playedCards = env.getLegalPlayCard(history_curr, hold, inning_level)
                    response = [play_pos, playedCards[random.randint(0, len(playedCards)-1)]]
                    playcard_mtrx = cards2matrix(response[1])
                    
                    '''存储当前进度的经验回放 回合内的动态buf'''
                    if record_index:
                        role = 'banker' if banker == p else 'idler'
                        hand_cards_buf[role].append(np.copy(infoset[p]['hand_cards']))
                        major_cards_buf[role].append(np.copy(infoset[p]['major_cards']))
                        public_cards_buf[role].append(np.copy(infoset[p]['public_cards']))
                        played_score_cards_buf[role].append(np.copy(infoset[p]['played_score_cards']))
                        remain_score_cards_buf[role].append(np.copy(infoset[p]['remain_score_cards']))
                        card_play_action_seq_buf[role].append([np.copy(actions) for actions in infoset[p]['card_play_action_seq']])
                        round_play_cards_buf[role].append([np.copy(actions) for actions in infoset[p]['round_play_cards']])
                        last_play_cards_buf[role].append(np.copy(infoset[p]['last_round_play_cards']))
                        played_cards_buf[role].append(np.copy(infoset[p]['played_cards']))
                        mask_cards_buf[role].append(np.copy(infoset[p]['mask_cards']))
                        my_seat_buf[role].append(np.copy(infoset[p]['my_seat']))
                        play_rights_seat_buf[role].append(np.copy(infoset[p]['play_rights_seat']))
                        banker_seat_buf[role].append(np.copy(infoset[p]['banker_seat']))
                        score_buf[role].append(np.copy(infoset[p]['score']))
                        win_score_buf[role].append(np.copy(infoset[p]['win_score_limit']))
                        num_cards_left_buf[role].append(np.copy(infoset[p]['num_cards_left']))
                        predict_action_buf[role].append(np.copy(playcard_mtrx))
                    
                    '''更新回合动态buf'''
                    #mask隐蔽牌
                    for i in range(__PLAYER_COUNT__):
                        mask_cards[i][playcard_mtrx == 1] = 0
                    
                    if play_counts == 0 :
                        first_play_mtx = np.copy(playcard_mtrx)
                        first_play_suit_mask = np.any(first_play_mtx == 1, axis=(0, 2))#找出 first_play_mtx 中值为 1 的位置位于第二维（Axis 1）的哪一行（即哪个花色）
                        suit_indices = np.where(first_play_suit_mask)[0]# 获取具体的花色索引 (例如: array([1]) 表示第1个花色，即红桃等，取决于具体定义) np.where 返回元组，取第一个元素 [0] 得到索引数组
                        play_suit_index = suit_indices[0]
                        
                        played_indices = playcard_mtrx == 1
                        is_all_major = np.all(major_cards_mtx[played_indices] == 1) if np.any(played_indices) else True
                    else:
                        #首出的牌是主牌，看有没有跟主牌
                        if is_all_major:
                            played_indices = playcard_mtrx == 1
                            follow_play_is_major = np.all(major_cards_mtx[played_indices] == 1) if np.any(played_indices) else True
                            if not follow_play_is_major:
                                mask_cards[play_pos][major_cards_mtx == 1] = 0
                                
                        #首出的不是主牌，看有没有全部出花牌
                        else:
                            play_card_num = np.sum(playcard_mtrx == 1)#出牌的数量
                            cards_in_play_suit = playcard_mtrx[:, play_suit_index, :]#提取 playcard_mtrx 在目标花色行的数据 
                            play_suit_card_num = np.sum(cards_in_play_suit == 1)
                            is_all_in_same_suit = (play_card_num == play_suit_card_num)
                            if not is_all_in_same_suit:
                                mask_cards[play_pos][:, play_suit_index, :] = 0
                        
                    # 出牌序列
                    round_play_cards[play_counts] = np.copy(playcard_mtrx)
                    # round_play_seat[play_counts][play_pos] = 1.0
                    
                    card_play_action_seq.append(np.copy(playcard_mtrx))
                    
                    #分牌
                    play_score_card = playcard_mtrx * remain_score_cards
                    played_score_cards = played_score_cards + play_score_card
                    remain_score_cards = remain_score_cards - play_score_card
                        
                    #已出牌
                    played_cards = played_cards + playcard_mtrx    
                    play_counts += 1
                            
                    
                    def checkRule():
                        pass
                        # if np.sum(playcard_mtrx[:,1:4, 13:14]) > 0:
                        #     raise ValueError("卡牌矩阵非法")
                        
                        # if np.sum(playcard_mtrx)%2 != 0 and np.sum(playcard_mtrx) != 1:
                        #     playedCards = env.getLegalPlayCard(history_curr, hold, inning_level)
                        #     raise ValueError("出牌报错，该出牌为空")
                        
                        
                        # for seat in range(play_counts):
                        #     if round_play_cards[seat].sum() != playcard_mtrx.sum():
                        #         raise ValueError("出牌报错，该出牌与历史出牌不一致")

                        

                        

                        
                        # # #test code 错误检验
                        # all_cards = played_score_cards + remain_score_cards
                        # card_num = np.sum(all_cards)
                        # if card_num != 24:
                        #     raise ValueError('分数牌不一致')
                        # for k in range(len(played_score_card_buf)):
                        #     score_cards = played_score_card_buf[k]
                        #     remain_score_cards = remain_score_card_buf[k]
                        #     all_cards = score_cards + remain_score_cards
                        #     card_num = np.sum(all_cards)
                        #     if card_num != 24:
                        #         raise ValueError('分数牌不一致')
                            
                    
                    

                #一回合结束
                elif stage == 'roundend' or stage == 'gameend':
                    if len(env.getPlayerHandCards(env.getPlayerPosition())) <= threshold_handcards:
                        record_index = True
                    
                    round_score = env.getLastRoundScore()
                    game_score = env.getTotalScore()
                    
                    #根据回合结束后的分数给reward
                    if stage == 'roundend':
                        #获得回合胜利
                        banker = env.getBanker()
                        #这里最好看有没有分牌，有分牌就调整reward
                        reward = round_score/10. + 0.05
                        mult = 1 if banker == play_rights else -1
                        reward_buf['banker'].append(torch.tensor(mult*reward))
                        reward_buf['idler'].append(torch.tensor(-mult*reward))
                        reward_buf['idler'].append(torch.tensor(-mult*reward))
                        # if banker == play_rights:
                        #     reward_list = [reward, -reward, -reward]
                        # else:
                        #     reward_list = []
                        #     for i in __PLAYER_COUNT__:
                        #         reward_list.append(-reward if (play_rights+i)%__PLAYER_COUNT__ == banker else reward)
                        # reward_list = np.array(reward_list, dtype=np.float32)
                        # reward_buf.extend(reward_list)
                        
                        play_rights = env.getPlayerPosition()
                        infoset['play_rights_seat'] = get_one_hot_array(play_rights+1, __PLAYER_COUNT__)
                        last_round_play_cards = [np.copy(play_cards) for play_cards in round_play_cards]
                        
                        play_counts = 0
                        round_cnt += 1
                        
                        #存储训练用的经验回放 即时奖励
                        '''role = 'banker' if banker == p else 'idler'
                        hand_cards_buf[role].append(np.copy(infoset[p]['hand_cards']))
                        major_cards_buf[role].append(np.copy(infoset[p]['major_cards']))
                        public_cards_buf[role].append(np.copy(infoset[p]['public_cards']))
                        played_score_cards_buf[role].append(np.copy(infoset[p]['played_score_cards']))
                        remain_score_cards_buf[role].append(np.copy(infoset[p]['remain_score_cards']))
                        card_play_action_seq_buf[role].append([np.copy(actions) for actions in infoset[p]['card_play_action_seq']])
                        round_play_cards_buf[role].append([np.copy(actions) for actions in infoset[p]['round_play_cards']])
                        last_play_cards_buf[role].append(np.copy(infoset[p]['last_round_play_cards']))
                        played_cards_buf[role].append(np.copy(infoset[p]['played_cards']))
                        mask_cards_buf[role].append(np.copy(infoset[p]['mask_cards']))
                        my_seat_buf[role].append(np.copy(infoset[p]['my_seat']))
                        play_rights_seat_buf[role].append(np.copy(infoset[p]['play_rights_seat']))
                        banker_seat_buf[role].append(np.copy(infoset[p]['banker_seat']))
                        score_buf[role].append(np.copy(infoset[p]['score']))
                        win_score_buf[role].append(np.copy(infoset[p]['win_score_limit']))
                        num_cards_left_buf[role].append(np.copy(infoset[p]['num_cards_left']))
                        predict_action_buf[role].append(np.copy(playcard_mtrx))'''
                        role ='banker'
                        while len(hand_cards_buf[role]) > T:
                            batch_queues.put({
                                "hand_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in hand_cards_buf[role][:T]]),
                                "major_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in major_cards_buf[role][:T]]),
                                "public_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in public_cards_buf[role][:T]]),
                                "played_score_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in played_score_cards_buf[role][:T]]),
                                "remain_score_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in remain_score_cards_buf[role][:T]]),
                                "card_play_action_seq_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in card_play_action_seq_buf[role][:T]]),
                                "round_play_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in round_play_cards_buf[role][:T]]),
                                "last_play_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in last_play_cards_buf[role][:T]]),
                                "played_cards_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in played_cards_buf[role][:T]]),
                                "my_seat_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in my_seat_buf[role][:T]]),
                                "play_rights_seat_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in play_rights_seat_buf[role][:T]]),
                                "banker_seat_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in banker_seat_buf[role][:T]]),
                                "score_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in score_buf[role][:T]]),
                                "win_score_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in win_score_buf[role][:T]]),
                                "num_cards_left_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in num_cards_left_buf[role][:T]]),
                                "predict_action_buf": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in predict_action_buf[role][:T]]),
                                "reward_buf": torch.stack([ndarr.clone().detach() for ndarr in reward_buf[role][:T]]),
                            })
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            hand_cards_buf[role] = hand_cards_buf[role][T:]
                            
                            
                            episode_return_buf[p] = episode_return_buf[p][T:]
                            target_adp_buf[p] = target_adp_buf[p][T:]
                            target_wp_buf[p] = target_wp_buf[p][T:]
                            target_wp_bid_buf[p] = target_wp_bid_buf[p][T:]
                            obs_x_batch_buf[p] = obs_x_batch_buf[p][T:]
                            obs_z_buf[p] = obs_z_buf[p][T:]
                            size[p] -= T
                            
                            index = free_queue.get()#bug 这里容易卡死 batchszie要小于num_buffers，batchszie不够就会一直等待足够的num_buffers，num_buffers又会等待batchsize训练数据释放
                            if index is None:
                                break
                            for t in range(T):
                                # buffers是tensor history_play_card_buff是array
                                buffers['card_play_action_seq'][index][t, ...] = torch.tensor(history_play_card_buf[t])
                                buffers['history_play_seat'][index][t, ...] = torch.tensor(history_play_seat_buf[t])
                                buffers['played_cards'][index][t, ...] = torch.tensor(history_played_card_buf[t])
                                buffers['history_bid_card'][index][t, ...] = torch.tensor(history_bid_card_buf[t])
                                buffers['history_bid_seat'][index][t, ...] = torch.tensor(history_bid_seat_buf[t])
                                buffers['round_play_cards'][index][t, ...] = torch.tensor(round_play_card_buf[t])
                                buffers['round_play_seat'][index][t, ...] = torch.tensor(round_play_seat_buf[t])
                                buffers['score_card'][index][t, ...] = torch.tensor(played_score_card_buf[t])
                                buffers['remain_score_cards'][index][t, ...] = torch.tensor(remain_score_card_buf[t])
                                buffers['my_seat'][index][t, ...] = torch.tensor(my_seat_buf[t])
                                buffers['banker_seat'][index][t, ...] = torch.tensor(banker_seat_buf[t])
                                buffers['public_cards'][index][t, ...] = torch.tensor(public_card_buf[t])
                                buffers['hand_card'][index][t, ...] = torch.tensor(hand_cards_buf[t])
                                buffers['mask_card'][index][t, ...] = torch.tensor(mask_cards_buf[t])#'''特征工程 规则层特征'''    
                                
                            
                            full_queue.put(index)
                            history_play_card_buf = history_play_card_buf[T:]
                            history_play_seat_buf = history_play_seat_buf[T:]
                            history_played_card_buf = history_played_card_buf[T:]
                            history_bid_card_buf = history_bid_card_buf[T:]
                            history_bid_seat_buf = history_bid_seat_buf[T:]
                            round_play_card_buf = round_play_card_buf[T:]
                            round_play_seat_buf = round_play_seat_buf[T:]
                            played_score_card_buf = played_score_card_buf[T:]
                            remain_score_card_buf = remain_score_card_buf[T:]
                            my_seat_buf = my_seat_buf[T:]
                            banker_seat_buf = banker_seat_buf[T:]
                            public_card_buf = public_card_buf[T:]
                            hand_cards_buf = hand_cards_buf[T:]
                            mask_cards_buf = mask_cards_buf[T:]
                    
                    # #数据合法性验证 test code
                    # for trj in range(len(card_play_action_seq)):
                    #     count = 0
                    #     for seat in range(4):
                    #         count += np.sum(card_play_action_seq[trj][seat])
                    #     if count%2 != 0:
                    #         raise ValueError(card_play_action_seq[trj][seat])
                    
                        

                    #重置回合信息
                    round_play_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    round_play_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)]                    
                    
                        
                    response = None
                    
                    
                    
                elif stage == "finalend":
                    break
            
            
            

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
    

