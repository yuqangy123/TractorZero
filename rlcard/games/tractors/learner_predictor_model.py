import torch
import torch.nn as nn
from torch.nn import functional as F

def create_optimizer(
    lr,
    learner_model
):
    optimizer = torch.optim.Adam(learner_model.parameters(), lr)
    return optimizer

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch
    #使用PPOClip的generate_batches


def learn(actor_model,
          learn_model,
          batch,
          optimizer,
          device,
          flags,
          mean_episode_return_buf,
          lock):
    
    history_play_card = batch['history_play_card'].to(device)
    history_play_seat = batch['history_play_seat'].to(device)
    history_bid_card = batch['history_bid_card'].to(device)
    history_bid_seat = batch['history_bid_seat'].to(device)

    label_hand_card = batch['play_hand_card'].to(device)
    label_public_card = batch['public_card'].to(device)

    
    with lock:
        opp_probs, bottom_prob = learn_model({'history_play_card':history_play_card, 'history_play_seat':history_play_seat,
                                       'history_bid_card':history_bid_card, 'history_bid_seat':history_bid_seat}
                                       )
        loss_opp = sum(F.binary_cross_entropy(probs, lbl) for probs,lbl in zip(opp_probs, label_hand_card))
        loss_bottom = F.binary_cross_entropy(bottom_prob, label_public_card)
        loss = loss_opp + loss_bottom
        
        stats = {
            'mean_episode_return': torch.mean(torch.stack([_r for _r in mean_episode_return_buf['cover']])).item(),
            'loss': loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(learn_model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_model.values():
            actor_model.load_state_dict(learn_model.state_dict())
        return stats
