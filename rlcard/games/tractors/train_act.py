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

