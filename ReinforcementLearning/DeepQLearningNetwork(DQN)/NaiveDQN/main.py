import gym
import model
import torch as t
# from tensorboardX import SummaryWriter
if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    # writer = SummaryWriter("DeepQLearningNetwork(DQN)\\Naive DQN\\logs")
    n_games = 1000
    """ scores=[]
    eps_history=[]
    losses=[]"""
    agent = model.DQN(
        env.observation_space.shape[0], env.action_space.n, 1e-3, 0.99, 1.0, 1e-5, 1e-2)
    # writer.add_graph(agent, t.randn((1, env.observation_space.shape[0])))
    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()
        state = state[0]
        loss_ep = 0
        step = 0
        while not done:

            action = agent.choose_action(t.FloatTensor(state).unsqueeze(0))
            state_, reward, done, truncated, info = env.step(action)#ObsType, float, bool, bool, dict
            score += reward
            loss = agent.learn(state, action, reward, state)
            loss_ep += loss
            step += 1
            state = state_
            print(f'action:{action}, \treward:{reward:.4f}, \t\tdone:{done}, \tstep:{step}')

        loss_ep /= step
        """scores.append(score)
        eps_history.append(agent.epsilon)
        losses.append(loss_ep)"""
        # writer.add_scalar("SCORES", score, i)
        # writer.add_scalar("epsilon", agent.epsilon, i)
        # writer.add_scalar("loss_ep", loss_ep, i)
