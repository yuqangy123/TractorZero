import torch
import torch.nn as nn


def create_optimizer(
    learning_rate,
    momentum,
    epsilon,
    alpha,
    learner_model
):
    learner_model.get_model('player')
    optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic}
            ]
        )
    
    optimizer = torch.optim.RMSprop(
        learner_model.parameters('cover'),
        lr=learning_rate,
        momentum=momentum,
        eps=epsilon,
        alpha=alpha)
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

def __compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

#(actors, learner_playModel, batch, optimizers['cover'], device, flags, mean_episode_return_buf, position_lock)
def learn(actor_model,
          learn_model,
          batch,
          optimizer,
          device,
          flags,
          mean_episode_return_buf,
          lock):    
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch[''][batch['done']]
    mean_episode_return_buf['cover'].append(torch.mean(episode_returns).to(device))
        
    with lock:
        learner_outputs = learn_model(obs_z, obs_x, return_value=True)
        loss = __compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return': torch.mean(torch.stack([_r for _r in mean_episode_return_buf['cover']])).item(),
            'loss': loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(learn_model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_model.values():
            actor_model.load_state_dict('cover', learn_model.state_dict())
        return stats
