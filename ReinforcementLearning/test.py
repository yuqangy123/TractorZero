import torch

for _ in range(10):
    dist = torch.Tensor([1, 2, 3, 8])
    
    ## 显式指定这是 logits ， PyTorch 会自动执行 Softmax 归一化 在强化学习（PPO）中，Actor 网络输出的是 Logits，应该使用
    dist = torch.distributions.Categorical(logits=dist)
    
    #使用熵正则化（训练损失层面  Loss = Actor_Loss - Coefficient * Entropy  ）
    # 防止策略过早收敛到局部最优，保持一定的探索能力。随着训练进行，你可以逐渐减小 c_entropy 系数。
    entropy = dist.entropy().mean()
    print('entropy:', entropy)
    
    # # 内部等效于: [1/10, 2/10, 3/10, 4/10] = [0.1, 0.2, 0.3, 0.4]    
    probs_raw = torch.Tensor([1, 2, 3, 8])
    dist = torch.distributions.Categorical(probs=probs_raw) 


    action = dist.sample()
    log_prob = dist.log_prob(action)
    print(action, log_prob)