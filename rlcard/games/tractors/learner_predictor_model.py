import torch
import torch.nn as nn
from torch.nn import functional as F
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=True
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
        
    T = flags.unroll_length
    batch_size = flags.batch_size
    loss_total, loss_opp_total, loss_bottom_total = torch.zeros(25).to(device), torch.zeros(25).to(device), torch.zeros(25).to(device)
    card_num_cnt = [0]*25
    with lock:
        for t in range(T):
            one_batch = {
                key: value[t] for key, value in batch.items()
            }
            label_hand_card = one_batch['hand_card'].to(device)
            label_public_card = one_batch['public_card'].to(device)

            
            opp_probs, bottom_prob = learn_model.predictCard(one_batch)
            
            #每个玩家的手牌数, 归一化
            hand_card_num = label_hand_card.view(label_hand_card.size(0),label_hand_card.size(1),-1).sum(-1)
            hand_card_num_norm = (25.0 - hand_card_num)/25.0

            loss_opp = F.binary_cross_entropy( opp_probs,  label_hand_card, reduction='none')
            # loss_opp = sum(F.binary_cross_entropy(prob, lbl) for prob,lbl in zip(opp_probs, label_hand_card))
            loss_bottom = F.binary_cross_entropy(bottom_prob, label_public_card, reduction='none')

            hand_card_entropy = hand_card_num_norm.view(hand_card_num_norm.size(0), hand_card_num_norm.size(1), 1, 1, 1)
            hand_card_entropy = hand_card_entropy.expand_as(loss_opp)
            hand_card_entropy.requires_grad = False
            loss_opp = loss_opp*hand_card_entropy

            hand_card_num_norm = hand_card_num_norm.mean(1)
            hand_card_entropy = hand_card_num_norm.view(hand_card_num_norm.size(0), 1, 1, 1)
            hand_card_entropy = hand_card_entropy.expand_as(loss_bottom)
            hand_card_entropy.requires_grad = False
            loss_bottom = loss_bottom*hand_card_entropy

            loss_opp_mean = loss_opp.view(loss_opp.size(0),loss_opp.size(1),-1).sum(-1).mean(1)
            loss_bottom_mean = loss_bottom.view(loss_bottom.size(0),-1).mean(1)
            hand_card_num = hand_card_num.mean(1).long()
            for i in range(hand_card_num.shape[0]):
                loss_opp_total[hand_card_num[i]] += loss_opp_mean[i]
                loss_bottom_total[hand_card_num[i]] += loss_bottom_mean[i]                
                card_num_cnt[hand_card_num[i]] += 1

            loss = loss_opp.mean() + loss_bottom.mean()
            loss_total += loss.item()
            
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(learn_model.get_model('predictor').parameters(), flags.max_grad_norm)
            optimizer.step()

            actor_model.get_model('predictor').load_state_dict(learn_model.get_model('predictor').state_dict())

        stats = {
            # 'mean_episode_return': torch.mean(torch.stack([_r for _r in mean_episode_return_buf['predictor']])).item(),
            'loss': loss_total/T,
        }
        print(f'opp loss:{loss_opp_total/T:.4f},bottom loss:{loss_bottom_total/T:.4f},total loss:{loss_total/T:.4f},')
        
        return stats
