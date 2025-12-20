import torch
import torch.nn as nn
from torch.nn import functional as F
import threading, math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# _writer_loss_opp = SummaryWriter('./logs/loss_opp')
_writer_loss_public_score_cards = SummaryWriter('./logs/loss_pub')
_writer_variance_opp = SummaryWriter('./logs/variance_opp')
_writer_variance_public_score_cards = SummaryWriter('./logs/variance_public_score_cards')

_writer_loss_opp_1 = SummaryWriter('./logs/loss_opp_1')
_writer_loss_opp_2 = SummaryWriter('./logs/loss_opp_2')
_writer_loss_opp_3 = SummaryWriter('./logs/loss_opp_3')
_writer_loss_opp_4 = SummaryWriter('./logs/loss_opp_4')
_writer_loss_opp_5 = SummaryWriter('./logs/loss_opp_5')

g_learn_count = [0]*5

CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=True
def create_optimizer(
    lr,
    learner_model
):
    optimizer = torch.optim.AdamW(learner_model.parameters(), lr)
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
    tid = threading.get_ident()
    print('get_batch_1', tid)
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    print('get_batch_2', tid)
    return batch
    #使用PPOClip的generate_batches



def learn(actor_model,
          learn_model,
          batch,
          optimizer,
          device,
          flags,
          mean_episode_return_buf,
          lock,
          learn_count):
    tid = threading.get_ident()
        
    T = flags.unroll_length
    B = flags.batch_size

    with torch.no_grad():
        # #统计各类手牌数量时的loss
        # loss_opp_total, loss_public_score_cards_total = torch.zeros(25).to(device), torch.zeros(25).to(device)

        # #统计对手牌模型的手牌预测准确率
        # variance_opp,variance_public_score_cards = torch.zeros(25).to(device), torch.zeros(25).to(device)

        # #不同手牌数量的训练次数统计
        # card_num_cnt, opp_card_num_cnt = [0]*25, [0]*25


        loss_opp_totalx, loss_public_score_cards_totalx = torch.zeros(25).to(device), torch.zeros(25).to(device)
        card_num_cntx, opp_card_num_cntx = torch.zeros(25).to(device), torch.zeros(25).to(device)
        variance_oppx,variance_public_score_cardsx = torch.zeros(25).to(device), torch.zeros(25).to(device)


    
    with lock:
        for t in range(T):
            one_batch = {#[T, batch_size, ...]
                key: value[t] for key, value in batch.items()
            }
            opp_logits, public_card_logits = learn_model.predictCard(one_batch, True)#[batch_size, 4, 4]， #[batch_size]
            
            label_hand_card = one_batch['hand_card'].to(device)
            label_public_card = one_batch['public_card'].to(device)
            
            my_seat_idx = one_batch['my_seat'].argmax(dim=1)

            label_hand_card_color_num = []
            hand_card_num = []
            other_seat = [ (my_seat_idx + k) % 4 for k in (1, 2, 3) ]
            for i in range(len(other_seat)):            
                # 假设取第 0 个玩家
                player_idx = other_seat[i]  # [32]
                batch_idx = torch.arange(B)  # [32]
                # 先进制索引取出
                other_hand_card = label_hand_card[batch_idx, player_idx, :, :, :-3]
                color_num = other_hand_card.sum(dim=(1,3))#手牌花色数量分布, bug，没有排除大小王列
                other_majorhand_card = label_hand_card[batch_idx, player_idx, :, :, -3:]
                color_num[:, :, 0] += other_majorhand_card[:]
                label_hand_card_color_num.append(color_num)
                hand_card_num.append(color_num.sum(dim=-1).long())

            label_hand_card_color_num = torch.stack(label_hand_card_color_num, dim=0).to(device=device)
            label_hand_card_color_num = label_hand_card_color_num.transpose(1,0)
            hand_card_num = torch.stack(hand_card_num, dim=0).to(device=device)
            hand_card_num = hand_card_num.transpose(1,0)
            label_public_card_socre = ((one_batch['remain_score_card']+one_batch['score_card']).clamp(min=0.0, max=1.0)*label_public_card).sum(dim=(1,2,3))
            #数据校验 ：((label_hand_card[0][1] + one_batch['mask_card'][0][1]) > 1).sum()
            
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True) 
            # start_event.record()

            
            
            # end_event.record()
             # torch.cuda.synchronize()
            # print("Forward time (ms):", start_event.elapsed_time(end_event))
            
            loss_opp = F.mse_loss(opp_logits, label_hand_card_color_num, reduction='none')#[batch_size, 4, 4]
            loss_pub = F.mse_loss(public_card_logits, label_public_card_socre, reduction='none')#[batch_size]

            #当前4个玩家真实的手牌数, 归一化，用于乘以loss，做置信度
            # hand_card_num = label_hand_card.view(B,label
            # _hand_card.size(1),-1).sum(-1).long() #【batch_size,4】
            hand_card_num_norm = (25.0 - hand_card_num)/25.0#【batch_size,4】

            
            loss_opp *= hand_card_num_norm.unsqueeze(-1).expand(-1, -1, 4)#【batch_size,4,4】
            # loss_pub *= hand_card_num_norm[torch.arange(B), one_batch['my_seat'].argmax(dim=1)]#[batch_size]
            
            
            
            # with torch.no_grad():
            #     # hand_card_num: [B,4]
            #     idx = (hand_card_num - 1)  # [B,4]
            #     var_opp = torch.abs( label_hand_card_color_num - opp_logits ).mean(dim=-1)
            #     var_pub = torch.abs(label_public_card_socre - public_card_logits)  # [B]

            #     loss_opp_mean = loss_opp.mean(dim=-1)       # [B,4]

            #     my_seat = one_batch['my_seat'].argmax(dim=1)#我的座位号
            #     my_hand_card_num = hand_card_num[torch.arange(B), my_seat] - 1#获取每个样本中对应my_seat位置的手牌数                
                
            #     for k in range(0,25):
            #         #统计对手牌模型指标
            #         mask = (idx == k)
            #         loss_opp_totalx[k] += loss_opp_mean[mask].sum()
            #         variance_oppx[k] += var_opp[mask].sum()
            #         opp_card_num_cntx[k] += mask.sum().item()

            #         #统计底牌分数预测模型指标
            #         mask = (my_hand_card_num == k)
            #         card_num_cntx[k] += mask.sum()
            #         loss_public_score_cards_totalx[k] += loss_pub[mask].sum()
            #         variance_public_score_cardsx[k] += var_pub[mask].sum()

                    
                
                
            
                ############################
                # for i in range(hand_card_num.shape[0]):
                #     handcardNum = hand_card_num[i]-1
                #     for j in range(handcardNum.shape[0]):                    
                #         #统计对手牌模型各类手牌数量时的loss
                #         #loss total [hand_card_num]=[loss]
                #         if handcardNum[j] > 0:
                #             loss_opp_total[handcardNum[j]] += loss_opp[i][j].mean()
                #             variance_opp[handcardNum[j]] += torch.abs(label_hand_card_color_num[i][j]-opp_logits[i][j]).mean()
                #             opp_card_num_cnt[handcardNum[j]] += 1#计算对手手牌数量统计
                                  
                #     my_seat = one_batch['my_seat'][i].argmax()#我的座位号
                #     my_hand_card_num = hand_card_num[i][my_seat]-1#我的手牌数
                #     if my_hand_card_num > 0:
                #         card_num_cnt[my_hand_card_num] += 1#只计算我的手牌数量统计
                #         loss_public_score_cards_total[my_hand_card_num] += loss_pub[i]
                #         variance_public_score_cards[my_hand_card_num] += torch.abs(label_public_card_socre[i] - public_card_logits[i])
                

            
            

            #梯度更新
            # loss_total = loss_opp.mean() + loss_pub.mean()
            loss_total = loss_opp.mean()
            cardnum_m = label_hand_card_color_num.sum().long().item()/B#平均牌预测数量
            l = loss_total.item()
            print(f"loss_opp_:{l:.3f}, {cardnum_m}")

            # if cardnum_m == 1: 
            #     _writer_loss_opp_1.add_scalar(f'loss_opp_1', l, g_learn_count[0]) 
            #     g_learn_count[0]+=1
            # elif cardnum_m == 2: 
            #     _writer_loss_opp_2.add_scalar(f'loss_opp_2', l, g_learn_count[1])
            #     g_learn_count[1]+=1
            # elif cardnum_m == 3: 
            #     _writer_loss_opp_3.add_scalar(f'loss_opp_3', l, g_learn_count[2])
            #     g_learn_count[2]+=1
            # elif cardnum_m == 4: 
            #     _writer_loss_opp_4.add_scalar(f'loss_opp_4', l, g_learn_count[3])
            #     g_learn_count[3]+=1
            # elif cardnum_m == 5: 
            #     _writer_loss_opp_5.add_scalar(f'loss_opp_5', l, g_learn_count[4])
            #     g_learn_count[4]+=1

            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(learn_model.get_model('predictor').parameters(), flags.max_grad_norm)
            optimizer.step()

            
                
            
        actor_model.get_model('predictor').load_state_dict(learn_model.get_model('predictor').state_dict())
            

        stats = {
            # 'mean_episode_return': torch.mean(torch.stack([_r for _r in mean_episode_return_buf['predictor']])).item(),
            # 'loss': loss_total/T,
        }
        # batch_size = 1#test code
        # loss_opp_total /= (T*batch_size)
        # loss_public_score_cards_total /= (T*batch_size)
        # variance_opp /= (T*batch_size)
        # variance_public_score_cards /= (T*batch_size)
        
        # print("loss_opp:")
        # print([ f'{loss.item()/opp_card_num_cntx[i].item():.4f} ' if opp_card_num_cntx[i].item() > 0 else 'none' for i, loss in enumerate(loss_opp_totalx)])
        # print('\n')
        # print("loss_pub:")
        # print([ f'{loss.item()/card_num_cntx[i].item():.4f} ' if card_num_cntx[i].item() > 0 else 'none' for i, loss in enumerate(loss_public_score_cards_totalx)])
        # print('\n')
        # print('variance_opp:')
        # print([f'{(acc).item()/opp_card_num_cntx[i].item():.4f}'  if opp_card_num_cntx[i].item() > 0 else 'none' for i, acc in enumerate(variance_oppx)])
        # print('\n')
        # print('variance_public_score_cards:')
        # print([f'{(acc).item()/card_num_cntx[i].item():.4f}' if card_num_cntx[i] > 0 else 'none' for i, acc in enumerate(variance_public_score_cardsx)])
        # print('\n-----------------------------------------------------------------------------------------------------------------------')
        
        # _writer_loss_opp.add_scalars('loss_opp', {f'loss_opp_{i}': v.item()/opp_card_num_cntx[i] if opp_card_num_cntx[i] > 0 else 0.0 for i, v in enumerate(loss_opp_totalx)}, g_learn_count)
        # _writer_loss_public_score_cards.add_scalars('loss_pub', {f'loss_public_score_cards_{i}': v.item()/card_num_cntx[i] if card_num_cntx[i] > 0 else 0.0 for i, v in enumerate(loss_public_score_cards_totalx)}, g_learn_count)
        # _writer_variance_opp.add_scalars('variance_opp', {f'variance_opp_{i}': v.item()/opp_card_num_cntx[i] if opp_card_num_cntx[i] > 0 else 0.0 for i, v in enumerate(variance_oppx)}, g_learn_count)
        # _writer_variance_public_score_cards.add_scalars('variance_public_score_cards', {f'variance_public_score_cards_{i}': v.item()/card_num_cntx[i] if card_num_cntx[i] > 0 else 0.0 for i, v in enumerate(variance_public_score_cardsx)}, g_learn_count)
        
       
       
        return stats
    

# def learn_card_distribute(actor_model,
#           learn_model,
#           batch,
#           optimizer,
#           device,
#           flags,
#           mean_episode_return_buf,
#           lock):
#     tid = threading.get_ident()
        
#     T = flags.unroll_length
#     batch_size = flags.batch_size

#     #统计各类手牌数量时的loss
#     loss_total, loss_opp_total, loss_bottom_total = torch.zeros(25).to(device), torch.zeros(25).to(device), torch.zeros(25).to(device)

#     #统计对手牌模型的手牌预测准确率
#     accuracy,accuracy_zero,accuracy_one = torch.zeros(25).to(device), torch.zeros(25).to(device), torch.zeros(25).to(device)

#     #不同手牌数量的训练次数统计
#     card_num_cnt = torch.zeros(25).to(device)
#     with lock:
#         for t in range(T):
#             one_batch = {
#                 key: value[t] for key, value in batch.items()
#             }
#             label_hand_card = one_batch['hand_card'].to(device)
#             label_public_card = one_batch['public_card'].to(device)

            
#             opp_probs, bottom_prob = learn_model.predictCard(one_batch)
            
#             #每个玩家的手牌数, 归一化
#             hand_card_num = label_hand_card.view(label_hand_card.size(0),label_hand_card.size(1),-1).sum(-1)
#             hand_card_num_norm = (25.0 - hand_card_num)/25.0

#             loss_opp = F.binary_cross_entropy( opp_probs,  label_hand_card, reduction='none')
#             # loss_opp = sum(F.binary_cross_entropy(prob, lbl) for prob,lbl in zip(opp_probs, label_hand_card))
#             loss_bottom = F.binary_cross_entropy(bottom_prob, label_public_card, reduction='none')

#             hand_card_entropy = hand_card_num_norm.view(hand_card_num_norm.size(0), hand_card_num_norm.size(1), 1, 1, 1)
#             hand_card_entropy = hand_card_entropy.expand_as(loss_opp)
#             hand_card_entropy.requires_grad = False
#             loss_opp = loss_opp*hand_card_entropy

#             hand_card_num_norm = hand_card_num_norm.mean(1)
#             hand_card_entropy = hand_card_num_norm.view(hand_card_num_norm.size(0), 1, 1, 1)
#             hand_card_entropy = hand_card_entropy.expand_as(loss_bottom)
#             hand_card_entropy.requires_grad = False
#             loss_bottom = loss_bottom*hand_card_entropy

#             loss_opp_mean = loss_opp.view(loss_opp.size(0),loss_opp.size(1),-1).sum(-1).mean(1)
#             loss_bottom_mean = loss_bottom.view(loss_bottom.size(0),-1).mean(1)
#             hand_card_num = hand_card_num.mean(1).long()

            

            
#             B = opp_probs  # rename for clarity
#             # 1. 展平最后三个维度：120 = 2*4*15
#             flat = B.view(B.size(0), B.size(1), -1)   # [batch_size,4,120]
#             for i in range(hand_card_num.shape[0]):

#                 #统计对手牌模型各类手牌数量时的loss
#                 loss_opp_total[hand_card_num[i]-1] += loss_opp_mean[i]
#                 loss_bottom_total[hand_card_num[i]-1] += loss_bottom_mean[i]                
#                 card_num_cnt[hand_card_num[i]-1] += 1

#                 #统计对手牌模型的手牌预测准确率
#                 values, indices = torch.topk(flat[i], hand_card_num[i], dim=1)
#                 mask_flat = torch.zeros_like(flat[i])
#                 mask_flat.scatter_(1, indices, 1)# 把 top-k 的位置置为 1
#                 opp_probs_mask = mask_flat.view_as(B[i])   # [4,2,4,15]
#                 # 计算：在最后几个维度 (2,3,4) 上分别求和，结果形状 [batch_size]
#                 same = (label_hand_card[i] == opp_probs_mask).sum(dim=(1,2,3))        # [batch_size]
#                 both_zero = ((label_hand_card[i] == 0) & (opp_probs_mask == 0)).sum(dim=(1,2,3)) 
#                 both_one  = ((label_hand_card[i] == 1) & (opp_probs_mask == 1)).sum(dim=(1,2,3))

#                 accuracy[hand_card_num[i]-1] += same.sum()
#                 accuracy_zero[hand_card_num[i]-1] += both_zero.sum()
#                 accuracy_one[hand_card_num[i]-1] += both_one.sum()
                



#             #梯度更新
#             loss = loss_opp.mean() + loss_bottom.mean()
#             loss_total += loss.item()
#             optimizer.zero_grad()
#             loss.backward()
#             nn.utils.clip_grad_norm_(learn_model.get_model('predictor').parameters(), flags.max_grad_norm)
#             optimizer.step()

#             actor_model.get_model('predictor').load_state_dict(learn_model.get_model('predictor').state_dict())

#         stats = {
#             # 'mean_episode_return': torch.mean(torch.stack([_r for _r in mean_episode_return_buf['predictor']])).item(),
#             'loss': loss_total/T,
#         }
#         print("loss_opp_total:")
#         print([ f'{(loss/card_num_cnt[i]).item():.4f} ' if card_num_cnt[i] > 0 else 0.0 for i, loss in enumerate(loss_opp_total)])
#         print('\n')
#         print('opp model accuracy:')
#         print([f'{(acc/card_num_cnt[i]/4/(120)).item():.4f}' if card_num_cnt[i] > 0 else 0.0 for i, acc in enumerate(accuracy)])
#         print('\n')
#         print('opp model accuracy_zero:')
#         print([f'{(acc_zero/card_num_cnt[i]/4/(120-i+1)).item():.4f}' if card_num_cnt[i] > 0 else 0.0 for i, acc_zero in enumerate(accuracy_zero)])
#         print('\n')
#         print('opp model accuracy_one:')
#         print([f'{(acc_one/card_num_cnt[i]/4/(i+1)).item():.4f}' if card_num_cnt[i] > 0 else 0.0 for i, acc_one in enumerate(accuracy_one)])
#         print('\n---------------------------------------------------------------------------------------------\n')
#         # print(f'opp loss:{loss_opp_total/T:.4f},bottom loss:{loss_bottom_total/T:.4f},total loss:{loss_total/T:.4f},')
        
        

#         return stats
